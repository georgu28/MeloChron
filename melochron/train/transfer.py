"""Re-point a pretrained encoder at a different catalog.

This is the piece that makes "pretrain on other people, serve to you" a real
operation rather than an intention. A checkpoint trained on lastfm-1K knows
171,904 items; a personal Spotify export contains a different 18,452, and the
two overlap by about 6%. Loading the checkpoint the ordinary way rebuilds the
model at the *old* catalog size and there is nowhere to put the new items.

What makes the transplant possible is a property of the text variants and only
of them. ``ProjectedTextEmbedding`` stores an ``[n_items, d_text]`` matrix and a
``[d_text, d_model]`` projection, and **only the matrix is catalog-shaped**. The
projection --- the part that actually learned how sentence-embedding space maps
into the model's space --- is the same size for any catalog. So the encoder, the
position embeddings, the time encoding and that projection all transfer, and the
new catalog arrives as a new text matrix.

An ``id`` checkpoint cannot do this, and that is not a limitation to work around
but the entire finding. Its item table *is* a lookup indexed by the old
vocabulary; row 40,112 means one specific lastfm track and nothing else. There
is no operation that turns it into a representation of a track it never saw.
Phase 2 argued transfer would matter for exactly this case; this module is where
the argument becomes checkable, so it raises rather than silently degrading.

One detail that will otherwise bite: the scoring head's bias is ``[n_items]``.
It is catalog-shaped, it cannot come along, and it must not be quietly reused.
The transplanted head is built without one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from melochron.data.vocab import Vocab
from melochron.models.heads import TiedItemScorer
from melochron.models.sasrec import SASRec
from melochron.models.scorer import SASRecScorer, build_scorer

#: Variants whose item representation is a function of text, and therefore
#: defined for items the model never saw.
TRANSFERABLE = ("text_frozen", "text_finetuned", "hybrid")


@dataclass
class Transplant:
    """A pretrained encoder wearing a new catalog."""

    model: SASRec
    head: TiedItemScorer
    scorer: SASRecScorer
    vocab: Vocab
    config: dict
    #: Weights that came from the checkpoint rather than from initialization.
    transferred: list[str]
    source: str

    def card(self) -> dict:
        return {
            "source": self.source,
            "variant": self.config.get("variant"),
            "n_items": int(self.model.n_items),
            "d_model": int(self.model.d_model),
            "max_len": int(self.model.max_len),
            "transferred_tensors": len(self.transferred),
        }


def load_for_catalog(
    path: str,
    text_vectors: torch.Tensor,
    vocab: Vocab,
    device: torch.device | str = "cpu",
    name: str = "zero-shot",
    freeze_encoder: bool = False,
) -> Transplant:
    """Load a pretrained checkpoint and attach it to ``text_vectors``.

    ``text_vectors`` is the new catalog's ``[n_items, d_text]`` matrix, row
    ``i`` being vocabulary id ``i``, built by ``features/embed.py`` against
    ``vocab``. It must have the same ``d_text`` as the checkpoint, since the
    projection being transferred consumes exactly that width.

    ``freeze_encoder`` leaves only the item representation trainable, which is
    the cheap per-user adaptation Phase 5 describes. It is off by default: a
    full fine-tune is the honest thing to measure against zero-shot first.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    variant = config.get("variant", "id")

    if variant not in TRANSFERABLE:
        raise ValueError(
            f"checkpoint variant {variant!r} cannot be re-pointed at another catalog: "
            f"its item table is indexed by the training vocabulary, so items it never "
            f"saw have no representation. Transferable variants are {list(TRANSFERABLE)}."
        )

    state = payload["model"]
    # The text buffer sits at items.text_vectors under the text-only variants
    # and at items.text.text_vectors under hybrid, which nests the text module.
    # Matched by suffix so a further nesting change does not silently exclude a
    # variant that TRANSFERABLE claims to support.
    text_keys = sorted(k for k in state if k.endswith("text_vectors"))
    if not text_keys:
        raise ValueError(
            f"checkpoint declares variant {variant!r} but carries no *text_vectors buffer"
        )
    key = text_keys[0]

    d_text_old = state[key].shape[1]
    if text_vectors.shape[1] != d_text_old:
        raise ValueError(
            f"text vectors are {text_vectors.shape[1]}-dimensional but the checkpoint's "
            f"projection consumes {d_text_old}; they must come from the same encoder"
        )
    if text_vectors.shape[0] != len(vocab):
        raise ValueError(
            f"text vectors have {text_vectors.shape[0]} rows but the vocabulary has "
            f"{len(vocab)}; row i must be vocabulary id i"
        )

    model, head, scorer = build_scorer(
        n_items=len(vocab),
        device=device,
        name=name,
        variant=variant,
        text_vectors=text_vectors,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_blocks=config["n_blocks"],
        max_len=config["max_len"],
        dropout=config["dropout"],
        use_time=config["use_time"],
    )

    # Everything except the catalog-shaped tensors. The new text matrix was
    # already installed by build_scorer and must not be overwritten by the old
    # one, and hybrid adds a second one: its residual is [n_items, d_model],
    # indexed by the *old* vocabulary, so row 40,112 is a correction learned
    # about one specific lastfm track.
    #
    # Dropping it is not a compromise, it is the designed behaviour. The
    # residual is zero-initialized precisely so that a row carrying no learned
    # signal falls back to its pure text vector, which is exactly the right
    # prior for a catalog this model has never seen. So a zero-shot hybrid
    # transplant arrives with every residual row at zero and no per-item
    # correction at all.
    #
    # It is tempting to conclude that this makes it equivalent to a zero-shot
    # text_frozen transplant. It does not, and the difference is the reason the
    # hybrid variant works: _combine L2-normalizes every row and applies a
    # learned scale, so its rows all have equal length, while text_frozen's
    # vary by ~3x. Ranking is a dot product, so that changes the order. It is
    # the same correction as commit 8109042, where unequal norms alone were
    # keeping cold items out of the top 10 while their directions were fine.
    #
    # Which sets up a prediction worth measuring rather than assuming: on a new
    # catalog *every* item is cold from the residual's point of view, which is
    # the regime normalization exists to fix, so a zero-shot hybrid should beat
    # a zero-shot text_frozen. tests/test_transfer.py pins the mechanism; the
    # personal-corpus table is where the size of it gets reported.
    catalog_shaped = {key} | {k for k in state if k.endswith("residual.weight")}
    portable = {k: v for k, v in state.items() if k not in catalog_shaped}
    missing, unexpected = model.load_state_dict(portable, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint carries tensors this model has no place for: {unexpected}")
    if set(missing) != catalog_shaped:
        raise ValueError(
            f"expected exactly {sorted(catalog_shaped)} to be missing, got {sorted(missing)}"
        )

    # The head bias is [n_items] and belongs to the old catalog. Rebuilt empty
    # rather than carried over, because a per-item prior learned from someone
    # else's listening is not a prior about these items.
    head = TiedItemScorer(model.items, use_bias=False, pad_id=model.pad_id)
    scorer = SASRecScorer(model, head, device=device, name=name, use_time=config["use_time"])

    if freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.items.parameters():
            parameter.requires_grad_(True)

    return Transplant(
        model=model,
        head=head,
        scorer=scorer,
        vocab=vocab,
        config=config,
        transferred=sorted(portable),
        source=str(path),
    )
