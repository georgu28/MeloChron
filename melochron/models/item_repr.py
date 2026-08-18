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
import torch.nn.functional as F
from torch import Tensor, nn


class ItemRepresentation(nn.Module):
    """Maps item ids to vectors, and exposes the full table for weight tying.

    Subclasses set ``n_items`` and ``d_model`` and implement
    :meth:`compute_item_vectors`; lookup and the public :meth:`item_vectors` are
    both defined in terms of it so the two can never disagree, which is what
    keeps the output head genuinely tied.

    Between those two sits a cache, and it exists because of a measured
    regression rather than a guess. For a computed representation the table is a
    ``[n_items, d_text] x [d_text, d_model]`` product --- 171,904 x 384 x 128 for
    the pretrained catalog. Evaluation calls
    :meth:`~melochron.models.scorer.SASRecScorer.score` once for thousands of
    instances and hoists that product out of its batch loop, so it is amortized
    to nothing. **Serving calls it once per request**, where the same hoist
    amortizes over a single history and the projection becomes the whole cost:
    the hybrid artifact measured 538 ms p50 per recommendation against the id
    variant's 8.2 ms, a 66x gap that is entirely this matrix.

    :meth:`freeze_item_vectors` materializes it once for a model whose weights
    are done moving. The staleness risk that would otherwise make a cache like
    this a bug is closed structurally: :meth:`train` and :meth:`_apply` both
    drop it, so entering training mode or moving to another device invalidates
    it rather than serving vectors that no longer match the parameters.
    """

    n_items: int
    d_model: int
    pad_id: int

    #: Set only by :meth:`freeze_item_vectors`. ``None`` means recompute on
    #: demand, which is the default and the only state training ever sees.
    _frozen_vectors: Tensor | None = None

    @abstractmethod
    def compute_item_vectors(self) -> Tensor:
        """The whole table, ``[n_items, d_model]``, computed from parameters."""

    def item_vectors(self) -> Tensor:
        """The whole table, from cache when one has been frozen."""
        frozen = self._frozen_vectors
        return self.compute_item_vectors() if frozen is None else frozen

    def freeze_item_vectors(self) -> Tensor:
        """Materialize the table once and serve that copy until invalidated.

        For inference only. Detached, so a frozen table can never be a path
        gradient flows along, and returned for a caller that wants to hold it.
        """
        with torch.no_grad():
            self._frozen_vectors = self.compute_item_vectors().detach()
        return self._frozen_vectors

    def thaw_item_vectors(self) -> None:
        """Drop any frozen table, returning to computing on demand."""
        self._frozen_vectors = None

    def train(self, mode: bool = True) -> ItemRepresentation:
        # A frozen table asserts the parameters behind it are done moving, which
        # training mode contradicts. Dropping it here is what makes the
        # fine-tuning path (load a checkpoint, freeze, then keep training) safe
        # without the caller having to remember.
        if mode:
            self.thaw_item_vectors()
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        # .to(), .cuda() and .float() all land here. A table frozen before the
        # move would be left on the old device or in the old dtype.
        self.thaw_item_vectors()
        return super()._apply(*args, **kwargs)

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

    def compute_item_vectors(self) -> Tensor:
        return self.embedding.weight

    def freeze_item_vectors(self) -> Tensor:
        """No-op: the table is already a parameter, so there is nothing to
        precompute and a cache would only be a second copy of it.

        This is why the id variant's 8.2 ms was never the number in danger.
        """
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

    def compute_item_vectors(self) -> Tensor:
        """``[n_items, d_model]``.

        Recomputed on every call unless the base class has a frozen table. That is a real cost at full-catalog eval
        (``n_items x d_text x d_model`` per batch), which is why
        :meth:`melochron.models.heads.TiedItemScorer.full_logits` accepts a
        precomputed matrix and the scorer hoists this out of its batch loop.
        """
        vectors = self.projection(self.text_vectors)
        # Keep the reserved rows at exactly zero: the projection is unbiased, so
        # zero in stays zero out, but this is asserted rather than assumed
        # because a later bias=True would break it silently.
        return vectors

    def forward(self, item_ids: Tensor) -> Tensor:
        """Look up ``item_ids``, projecting only the rows asked for.

        The base class would build the whole table and index into it. That is
        the same arithmetic in the wrong order, and the order is worth 860x
        here: projecting a 200-play history is ``200 x 384 x 128``, while
        projecting the catalog first is ``171,904 x 384 x 128`` to then discard
        99.9% of it. Because the projection is an unbiased ``nn.Linear``,
        gathering before it and gathering after it are equal --- indexing rows
        commutes with a right multiplication --- so this is free to do.

        It was costing 166 ms of every hybrid recommendation, and it is paid on
        every training step of the text variants too, where the sequence and its
        sampled negatives are both looked up this way.
        """
        return self.projection(self.text_vectors[item_ids])

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
        normalize: bool = True,
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

        self.normalize = normalize
        # A single learned scalar, applied to every item alike. It sets the
        # softmax temperature without ever changing the *relative* order of two
        # items, which is what keeps normalization from smuggling in a new
        # per-item advantage.
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def _combine(self, text: Tensor, residual: Tensor) -> Tensor:
        vectors = text + self.residual_scale * residual
        if not self.normalize:
            return vectors
        # L2-normalize so items compete on direction alone.
        #
        # Without this the first hybrid run scored a flat 0.0000 on every cold
        # item despite the residual guarantee holding exactly: 7,306 rows were
        # still zero, as designed. The residual simply grew, so trained items
        # ended at norm 1.52 against 0.56 for pure-text items, a 2.7x gap. Since
        # scoring is a dot product, that magnitude difference alone kept cold
        # items out of the top 10 against 164k trained competitors. Their
        # direction was fine the whole time; only their length was wrong.
        return F.normalize(vectors, dim=-1) * self.log_scale.exp()

    def compute_item_vectors(self) -> Tensor:
        # self.text.item_vectors(), not compute_: if the nested text module has
        # its own frozen table this reuses it.
        return self._combine(self.text.item_vectors(), self.residual.weight)

    def forward(self, item_ids: Tensor) -> Tensor:
        return self._combine(self.text(item_ids), self.residual(item_ids))

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
