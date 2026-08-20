"""The adoption model: a reused sequence encoder plus a binary head.

The encoder is `melochron.models.sasrec.SASRec`, unchanged. `encode_last` gives
the `[B, d_model]` representation of the user's history up to the encounter; the
candidate track is embedded by the *same* item representation, so history and
candidate live in one space. A small MLP over `[h, c, h⊙c]` emits one logit.

The head comes in two shapes, and the difference is the experiment:

* **pure-sequence** — the MLP sees only the history and the candidate. It has to
  recover the per-user adoption rate from the sequence itself.
* **sequence + priors** — the MLP is also handed the two scalar baseline rates
  (`user-prior`, `item-rate`), so it starts from the baseline and learns what the
  sequence adds on top.

Reporting both is the point: their gap is "what does listening order and content
add over two numbers", which is a result, not a hyperparameter.

This is binary classification, so the loss is plain BCE and there is no negative
sampling — every first encounter is already a labelled example. That is the
structural reason none of the ranking machinery in `train/` or `models/heads.py`
is reused.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from melochron.models.item_repr import build_item_representation
from melochron.models.sasrec import SASRec

#: The two extra scalar features the `sequence + priors` head consumes.
N_PRIOR_FEATURES = 2


class AdoptionHead(nn.Module):
    """MLP mapping ``[h, c, h⊙c]`` (+ optional priors) to one adoption logit."""

    def __init__(
        self, d_model: int, hidden: int = 128, use_priors: bool = False, dropout: float = 0.2
    ):
        super().__init__()
        self.use_priors = use_priors
        in_features = 3 * d_model + (N_PRIOR_FEATURES if use_priors else 0)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, history: Tensor, candidate: Tensor, priors: Tensor | None = None) -> Tensor:
        features = [history, candidate, history * candidate]
        if self.use_priors:
            if priors is None:
                raise ValueError("this head was built with use_priors=True but got no priors")
            # Logit-space, matching how `user × item` combines them, and clamped
            # so a prior of exactly 0 or 1 does not become an infinite feature.
            features.append(torch.logit(priors.clamp(1e-4, 1 - 1e-4)))
        elif priors is not None:
            raise ValueError("this head has use_priors=False but was given priors")
        return self.net(torch.cat(features, dim=-1)).squeeze(-1)


class AdoptionModel(nn.Module):
    """Sequence encoder plus binary head, scoring one candidate per encounter.

    ``item_variant`` selects the representation shared by history and candidate:
    ``"id"`` is the Phase 3 headline, ``"hybrid"`` etc. are the Phase 4
    ablations, built behind the same seam so the model code does not branch on
    which one is in use.
    """

    def __init__(
        self,
        n_items: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
        use_time: bool = True,
        use_priors: bool = False,
        item_variant: str = "id",
        text_vectors: Tensor | None = None,
        head_hidden: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        self.config = {
            "n_items": n_items,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_blocks": n_blocks,
            "max_len": max_len,
            "dropout": dropout,
            "use_time": use_time,
            "use_priors": use_priors,
            "item_variant": item_variant,
            "head_hidden": head_hidden,
            "pad_id": pad_id,
        }
        item_repr = build_item_representation(item_variant, n_items, d_model, text_vectors, pad_id)
        self.encoder = SASRec(
            n_items=n_items,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            max_len=max_len,
            dropout=dropout,
            use_time=use_time,
            pad_id=pad_id,
            item_repr=item_repr,
        )
        self.head = AdoptionHead(d_model, head_hidden, use_priors=use_priors, dropout=dropout)
        self.use_priors = use_priors
        self.pad_id = pad_id

    def candidate_vectors(self, candidate_ids: Tensor) -> Tensor:
        """Embed candidate tracks with the encoder's own item representation."""
        return self.encoder.items(candidate_ids)

    def forward(
        self,
        item_ids: Tensor,
        time_deltas: Tensor | None,
        candidate_ids: Tensor,
        priors: Tensor | None = None,
    ) -> Tensor:
        """One adoption logit per encounter.

        ``item_ids``/``time_deltas`` are the ``[B, L]`` history windows,
        ``candidate_ids`` the ``[B]`` encountered tracks, ``priors`` the ``[B, 2]``
        baseline rates when the head consumes them.
        """
        history = self.encoder.encode_last(item_ids, time_deltas if self.encoder.use_time else None)
        candidate = self.candidate_vectors(candidate_ids)
        return self.head(history, candidate, priors)
