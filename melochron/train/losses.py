"""Training objectives over sampled negatives.

``TiedItemScorer.sampled_logits`` returns the positive and negative scores
unreduced, deliberately declining to choose a loss. This module chooses.

Both objectives take the same ``(positive, negative, mask)`` shape and both
reduce over unmasked positions only. The mask is not optional bookkeeping: a
left-padded batch is mostly padding for short histories, and averaging over
padded positions would scale the loss by an arbitrary factor that varies with
batch composition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` over ``True`` entries of ``mask``.

    Returns a real zero (still attached to the graph) when nothing is unmasked,
    rather than a NaN from dividing by zero.
    """
    mask = mask.to(values.dtype)
    total = mask.sum()
    if total == 0:
        return (values * mask).sum()
    return (values * mask).sum() / total


def sampled_softmax_loss(positive: Tensor, negative: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy over ``[positive, negatives]`` with the positive at index 0.

    ``positive`` is ``[N, 1]``, ``negative`` is ``[N, K]``, ``mask`` is ``[N]``.

    This is the sampled approximation to a full softmax over the catalog. The
    full version at ``B=128, L=200, V=50k`` would need ~5.12 GB for the logits
    alone; with 512 negatives it is ~52 MB. Evaluation still ranks against the
    entire catalog, so the approximation buys training throughput without
    touching the reported numbers.
    """
    logits = torch.cat([positive, negative], dim=-1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    per_row = F.cross_entropy(logits, target, reduction="none")
    return _masked_mean(per_row, mask)


def bpr_loss(positive: Tensor, negative: Tensor, mask: Tensor) -> Tensor:
    """Bayesian Personalized Ranking: ``-log sigmoid(pos - neg)``.

    Pairwise rather than listwise. Included as an alternative because BPR
    optimizes the ranking of a positive above each negative independently,
    which is closer to what the HR/NDCG metrics measure than a softmax over a
    sampled set is. Whether that translates into better numbers here is an
    empirical question, so it is a config switch and not a decision baked in.

    ``logsigmoid`` rather than ``log(sigmoid(...))``: the latter underflows to
    ``-inf`` once the margin is strongly negative, which is exactly the regime
    early training sits in.
    """
    per_pair = -F.logsigmoid(positive - negative)
    per_row = per_pair.mean(dim=-1)
    return _masked_mean(per_row, mask)


LOSSES = {
    "sampled_softmax": sampled_softmax_loss,
    "bpr": bpr_loss,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; expected one of {sorted(LOSSES)}")
    return LOSSES[name]
