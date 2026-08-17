"""Scoring heads over the item vocabulary.

The output head shares weights with the input item-embedding table, as in the
original SASRec. Tying is not just a parameter saving: it means an item's
representation as *context* and its representation as a *target* are the same
vector, so anything the encoder learns about an item from seeing it in a
history immediately improves how well it can be predicted. With a vocabulary in
the tens of thousands and a 6GB card, the parameter saving is welcome too.

Two scoring paths, and the split between them is a correctness matter, not an
optimization:

- :meth:`TiedItemScorer.sampled_logits` scores a positive against a handful of
  sampled negatives. Training only.
- :meth:`TiedItemScorer.full_logits` scores every item in the vocabulary.
  Evaluation only.

Krichene and Rendle (KDD 2020) showed that metrics computed against sampled
negatives are inconsistent with full ranking and can invert the ordering
between two models. Reporting a sampled metric would therefore be reporting a
number that does not mean what it appears to mean, so the eval path never
touches the sampled head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from melochron.models.item_repr import ItemRepresentation
from melochron.models.sasrec import DEFAULT_PAD_ID

#: Must match ``melochron.data.vocab.OOV_ID``. See the note on
#: ``sasrec.DEFAULT_PAD_ID`` for why these are duplicated rather than imported.
DEFAULT_OOV_ID = 1


class TiedItemScorer(nn.Module):
    """Scores hidden states against the item embedding table.

    Holds a reference to the encoder's ``nn.Embedding`` rather than a copy, so
    the tie survives ``state_dict`` round-trips and optimizer construction.
    """

    def __init__(
        self,
        items: ItemRepresentation,
        use_bias: bool = True,
        pad_id: int = DEFAULT_PAD_ID,
        oov_id: int = DEFAULT_OOV_ID,
    ):
        super().__init__()
        self.items = items
        self.pad_id = pad_id
        self.oov_id = oov_id
        self.bias = nn.Parameter(torch.zeros(items.n_items)) if use_bias else None

    @property
    def n_items(self) -> int:
        return self.items.n_items

    def full_logits(
        self, hidden: Tensor, mask_reserved: bool = True, item_vectors: Tensor | None = None
    ) -> Tensor:
        """Score ``hidden`` ``[..., D]`` against the whole vocabulary.

        Returns ``[..., n_items]``. With ``mask_reserved``, the PAD and OOV
        slots are driven to ``-inf`` so they can never be ranked as a
        recommendation --- recommending "unknown track" would otherwise be a
        legal and occasionally high-scoring output, which quietly inflates
        every metric it displaces a real item from.

        ``item_vectors`` lets a caller hoist the ``[n_items, d_model]`` matrix
        out of a batch loop. That is free for an ID table, which just returns
        its weight, but a projected text representation recomputes a
        ``n_items x d_text x d_model`` product on every call and a full-catalog
        evaluation would otherwise pay it once per batch.
        """
        if item_vectors is None:
            item_vectors = self.items.item_vectors()

        logits = hidden @ item_vectors.t()
        if self.bias is not None:
            logits = logits + self.bias

        if mask_reserved:
            # Out of place, not in place: an in-place write on a tensor that is
            # still part of the autograd graph would break the backward pass,
            # and this path is shared with the fine-tuning ablation.
            reserved = torch.zeros_like(logits)
            reserved[..., self.pad_id] = float("-inf")
            reserved[..., self.oov_id] = float("-inf")
            logits = logits + reserved
        return logits

    def score_candidates(self, hidden: Tensor, candidate_ids: Tensor) -> Tensor:
        """Score a per-row candidate set.

        ``hidden`` is ``[B, D]``, ``candidate_ids`` is ``[B, C]``; returns
        ``[B, C]``. This is the serving path, where the candidate set is a
        retrieved shortlist rather than the full catalog.
        """
        emb = self.items(candidate_ids)
        logits = torch.einsum("bd,bcd->bc", hidden, emb)
        if self.bias is not None:
            logits = logits + self.bias[candidate_ids]
        return logits

    def sampled_logits(
        self, hidden: Tensor, positive_ids: Tensor, negative_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Score one positive and ``N`` negatives per row. Training only.

        ``hidden`` is ``[B, D]``, ``positive_ids`` ``[B]``, ``negative_ids``
        ``[B, N]``. Returns ``(positive [B, 1], negative [B, N])`` so the caller
        can concatenate them into a softmax with the positive at index 0, or
        use them in a pairwise loss, without this module deciding which.

        **Memory warning.** Per-row negatives materialize a ``[B, N, D]``
        embedding tensor. Training scores every position of every sequence, so
        ``B`` is ``batch x seq_len``: at 128 x 200 with 512 negatives and
        ``d_model`` 128 that is ~5.2 GB for one intermediate, which OOMs a 6 GB
        card. Prefer :meth:`shared_negative_logits` for training and keep this
        for cases that genuinely need a different candidate set per row.
        """
        pos = self.score_candidates(hidden, positive_ids.unsqueeze(1))
        neg = self.score_candidates(hidden, negative_ids)
        return pos, neg

    def shared_negative_logits(
        self, hidden: Tensor, positive_ids: Tensor, negative_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Score positives against **one** negative set shared by the batch.

        ``hidden`` is ``[B, D]``, ``positive_ids`` ``[B]``, ``negative_ids``
        ``[K]``. Returns ``(positive [B, 1], negative [B, K])``.

        The memory difference is the reason this exists. Per-row negatives need
        a ``[B, K, D]`` gather; sharing needs only ``[K, D]`` for the embeddings
        and ``[B, K]`` for the scores. At ``B=20000, K=512, D=128`` that is
        ~41 MB instead of ~5.2 GB.

        Sharing correlates the negatives across positions within a batch, which
        is a real statistical difference, not just an optimization. It is also
        standard practice in this literature, and the gradient is unbiased in
        the same sense: every position still sees ``K`` draws from the same
        sampling distribution. Cheap enough that ``K`` can be raised well past
        what per-row sampling could afford, which more than repays the
        correlation.
        """
        pos_emb = self.items(positive_ids)  # [B, D]
        neg_emb = self.items(negative_ids)  # [K, D]

        pos = (hidden * pos_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        neg = hidden @ neg_emb.t()  # [B, K]

        if self.bias is not None:
            pos = pos + self.bias[positive_ids].unsqueeze(1)
            neg = neg + self.bias[negative_ids].unsqueeze(0)
        return pos, neg


def sample_negatives(
    n_items: int,
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    counts: Tensor | None = None,
    first_item_id: int = 2,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw negative item ids, uniformly or by popularity.

    ``counts`` gives play counts aligned to item id (``Vocab.counts``); when
    supplied, negatives are drawn proportional to ``count ** 0.75``, the
    word2vec smoothing exponent. Popularity sampling matters here because
    uniform negatives from a long-tailed catalog are almost always obscure, and
    a model only ever asked to beat obscure items never learns to discriminate
    among the popular ones it will actually be ranking at eval time.

    Never draws PAD or OOV. Does not exclude the positive: at these vocabulary
    sizes the collision rate is negligible and rejection sampling costs more
    than it saves.
    """
    if counts is None:
        return torch.randint(first_item_id, n_items, shape, device=device, generator=generator)

    weights = counts.to(device=device, dtype=torch.float).clamp(min=0) ** 0.75
    weights[:first_item_id] = 0.0
    flat = torch.multinomial(
        weights, num_samples=int(torch.tensor(shape).prod()), replacement=True, generator=generator
    )
    return flat.view(shape)
