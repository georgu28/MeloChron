"""Item representations: the swappable half of the transfer ablation.

Phase 2 compares three ways of representing an item, and the comparison is the
project's headline claim:

===========================  ==========================================
Random-init ID embeddings    no transfer
Frozen text embeddings       pure transfer
Text embeddings, fine-tuned  transfer plus adaptation
===========================  ==========================================

Those are three *representations* behind one encoder, not three models, so the
encoder must not know which it has. Everything here exposes the same two
operations --- look up some ids, and hand over the whole ``[n_items, d_model]``
matrix so the output head can tie to it --- and :class:`~melochron.models.sasrec.SASRec`
takes whichever it is given.

Why this matters beyond tidiness: the pretraining corpus (lastfm-1K, 2005-2009)
and the personal Spotify history (2020s) have near-zero item-ID overlap, so an
ID table transfers nothing across that gap by construction. All real transfer
has to flow through text. Wiring the encoder directly to an ``nn.Embedding``
would make the frozen-text variant, the unseen-item evaluation slice, and the
Phase 5 cold-start path all unimplementable at once.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn


class ItemRepresentation(nn.Module):
    """Maps item ids to vectors, and exposes the full table for weight tying.

    Subclasses set ``n_items`` and ``d_model`` and implement
    :meth:`item_vectors`; lookup is defined in terms of it so the two can never
    disagree, which is what keeps the output head genuinely tied.
    """

    n_items: int
    d_model: int
    pad_id: int

    @abstractmethod
    def item_vectors(self) -> Tensor:
        """The whole table, ``[n_items, d_model]``."""

    def forward(self, item_ids: Tensor) -> Tensor:
        """Look up ``item_ids`` of any shape, returning ``[..., d_model]``."""
        return self.item_vectors()[item_ids]


class IdEmbedding(ItemRepresentation):
    """A learned lookup table. The no-transfer control in the ablation."""

    def __init__(self, n_items: int, d_model: int, pad_id: int = 0):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(n_items, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()

    def item_vectors(self) -> Tensor:
        return self.embedding.weight

    def forward(self, item_ids: Tensor) -> Tensor:
        # Goes through nn.Embedding rather than the base class's index-select so
        # that padding_idx keeps its gradient behaviour.
        return self.embedding(item_ids)


class ProjectedTextEmbedding(ItemRepresentation):
    """Sentence-transformer vectors projected into the model's width.

    ``text_vectors`` is ``[n_items, d_text]``, aligned to vocabulary ids and
    produced offline by ``features/embed.py``. Rows for PAD and OOV are
    expected to be zero and are re-zeroed here rather than trusted.

    ``freeze`` selects between the two transfer variants: frozen keeps the
    text matrix fixed and trains only the projection (pure transfer), unfrozen
    lets the whole thing move (transfer plus adaptation). The projection is
    always trainable --- a frozen 384-dim space and a 128-dim model need *some*
    learned map between them, and freezing that too would test nothing.

    Cold-start works because an item never seen in training still has a text
    vector: its representation comes from what it *is*, not from an id that was
    never trained.
    """

    def __init__(
        self,
        text_vectors: Tensor,
        d_model: int,
        pad_id: int = 0,
        oov_id: int = 1,
        freeze: bool = True,
    ):
        super().__init__()
        if text_vectors.dim() != 2:
            raise ValueError(
                f"text_vectors must be [n_items, d_text], got {tuple(text_vectors.shape)}"
            )

        self.n_items, d_text = text_vectors.shape
        self.d_model = d_model
        self.pad_id = pad_id
        self.oov_id = oov_id
        self.frozen = freeze

        vectors = text_vectors.detach().clone().float()
        vectors[pad_id].zero_()
        vectors[oov_id].zero_()

        if freeze:
            # A buffer, not a parameter: it must ride along in the checkpoint
            # (serving needs the same vectors) without ever taking a gradient.
            self.register_buffer("text_vectors", vectors)
        else:
            self.text_vectors = nn.Parameter(vectors)

        self.projection = nn.Linear(d_text, d_model, bias=False)
        nn.init.normal_(self.projection.weight, std=0.02)

    def item_vectors(self) -> Tensor:
        """``[n_items, d_model]``.

        Recomputed on every call. That is a real cost at full-catalog eval
        (``n_items x d_text x d_model`` per batch), which is why
        :meth:`melochron.models.heads.TiedItemScorer.full_logits` accepts a
        precomputed matrix and the scorer hoists this out of its batch loop.
        """
        vectors = self.projection(self.text_vectors)
        # Keep the reserved rows at exactly zero: the projection is unbiased, so
        # zero in stays zero out, but this is asserted rather than assumed
        # because a later bias=True would break it silently.
        return vectors

    def extra_repr(self) -> str:
        return f"n_items={self.n_items}, d_model={self.d_model}, frozen={self.frozen}"


class HybridItemRepresentation(ItemRepresentation):
    """Frozen text prior plus a learned per-item residual.

    ``item_vectors = projection(text) + residual[id]``

    This exists because the ablation made the answer obvious. Learned ID
    embeddings won every slice where the item had been seen in training
    (HR@10 0.2553 overall) and scored a structural **zero** on items that had
    not, because a per-ID table has no row for them. Frozen text vectors were
    the mirror image: 0.2745 on those same cold items and far worse everywhere
    else. Neither representation is wrong; each is missing what the other has.

    So text supplies the prior and the interaction data supplies a correction on
    top of it. Text says what an item *is*; the residual learns what listeners
    actually *do* with it, which no amount of metadata can supply. Artist-level
    tags in particular give every track by an artist identical text, so text
    alone cannot distinguish tracks within an artist, and choosing among tracks
    by an artist already playing is a large share of next-track prediction. The
    residual is exactly where that distinction can be learned.

    **The residual is zero-initialized, and that is the whole trick.** A row
    that never receives gradient stays exactly zero, so the item falls back to
    pure text with nothing corrupting it. Cold items never receive gradient:
    they are never positives (absent from the training period by definition) and
    never negatives (popularity sampling weights by training count, so a count
    of zero is drawn with probability zero). Random initialization would instead
    add noise to precisely the items with no signal to override it, which is the
    one case this design exists to protect.
    """

    def __init__(
        self,
        text_vectors: Tensor,
        d_model: int,
        pad_id: int = 0,
        oov_id: int = 1,
        freeze_text: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.text = ProjectedTextEmbedding(
            text_vectors, d_model, pad_id=pad_id, oov_id=oov_id, freeze=freeze_text
        )
        self.n_items = self.text.n_items
        self.d_model = d_model
        self.pad_id = pad_id
        self.oov_id = oov_id
        self.residual_scale = residual_scale

        self.residual = nn.Embedding(self.n_items, d_model, padding_idx=pad_id)
        nn.init.zeros_(self.residual.weight)

    def item_vectors(self) -> Tensor:
        return self.text.item_vectors() + self.residual_scale * self.residual.weight

    def forward(self, item_ids: Tensor) -> Tensor:
        return self.text(item_ids) + self.residual_scale * self.residual(item_ids)

    def pure_text_rows(self) -> Tensor:
        """Ids whose residual is still exactly zero, i.e. pure-text items.

        Diagnostic rather than decoration. If this count does not roughly match
        the number of items absent from the training period, gradient is leaking
        into rows that should have none and the cold-start guarantee is quietly
        broken.
        """
        return (self.residual.weight.abs().sum(dim=-1) == 0).nonzero(as_tuple=True)[0]

    def extra_repr(self) -> str:
        return (
            f"n_items={self.n_items}, d_model={self.d_model}, residual_scale={self.residual_scale}"
        )


def build_item_representation(
    variant: str,
    n_items: int,
    d_model: int,
    text_vectors: Tensor | None = None,
    pad_id: int = 0,
) -> ItemRepresentation:
    """Construct the representation named by a config string.

    ``variant`` is one of ``"id"``, ``"text_frozen"``, ``"text_finetuned"``,
    ``"hybrid"`` --- the rows of the ablation table, so a config file selects a
    row by name and the whole table regenerates from one command.

    ``"hybrid"`` is the one intended for deployment. The other three are the
    controls that establish why: ``id`` cannot score an unseen item at all, and
    text alone gives up most of the accuracy on seen ones.
    """
    if variant == "id":
        return IdEmbedding(n_items, d_model, pad_id=pad_id)

    if variant in ("text_frozen", "text_finetuned"):
        if text_vectors is None:
            raise ValueError(f"variant {variant!r} needs text_vectors")
        if len(text_vectors) != n_items:
            raise ValueError(
                f"text_vectors has {len(text_vectors)} rows but vocabulary has {n_items}; "
                "they must be aligned by item id"
            )
        return ProjectedTextEmbedding(
            text_vectors, d_model, pad_id=pad_id, freeze=variant == "text_frozen"
        )

    if variant == "hybrid":
        if text_vectors is None:
            raise ValueError("variant 'hybrid' needs text_vectors")
        if len(text_vectors) != n_items:
            raise ValueError(
                f"text_vectors has {len(text_vectors)} rows but vocabulary has {n_items}; "
                "they must be aligned by item id"
            )
        return HybridItemRepresentation(text_vectors, d_model, pad_id=pad_id)

    raise ValueError(
        f"unknown item representation {variant!r}; expected one of "
        "'id', 'text_frozen', 'text_finetuned', 'hybrid'"
    )
