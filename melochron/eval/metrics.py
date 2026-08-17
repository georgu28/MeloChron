"""Ranking metrics over a full candidate catalog.

Two decisions here are load-bearing and both are deliberately conservative.

**Full-catalog ranking, not sampled negatives.** Krichene and Rendle (KDD 2020)
showed that sampling a small negative set produces metrics that are not merely
noisier but *inconsistent*: they can reverse the relative ordering of two
models. Sampled softmax is used for training throughput; evaluation always
ranks against the whole capped vocabulary.

**Pessimistic tie-breaking.** The popularity baseline assigns an identical
score to every item in its tail, and the repeat baseline assigns an identical
score to everything the user has never played. Counting only strictly-greater
scores would hand those baselines a near-perfect rank on a coin flip. Ties are
therefore resolved against the model, which understates a tied model rather
than flattering it.

Seen items are **not** masked out of the candidate set. Replaying a track is a
real and dominant outcome in music listening; removing it would define away the
exact behaviour the repeat/novel split exists to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_KS = (5, 10, 20)


def ranks_from_scores(
    scores: np.ndarray, targets: np.ndarray, tie_policy: str = "pessimistic"
) -> np.ndarray:
    """0-indexed rank of each target within its row of ``scores``.

    ``scores`` is ``(B, V)``, ``targets`` is ``(B,)``. A rank of 0 means the
    target was ranked first.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (B, V), got shape {scores.shape}")
    if len(targets) != scores.shape[0]:
        raise ValueError(f"got {scores.shape[0]} score rows but {len(targets)} targets")

    target_scores = scores[np.arange(len(targets)), targets][:, None]
    greater = (scores > target_scores).sum(axis=1)

    if tie_policy == "optimistic":
        return greater
    if tie_policy == "pessimistic":
        # Every other item sharing the target's score is placed ahead of it.
        ties = (scores == target_scores).sum(axis=1) - 1
        return greater + ties
    raise ValueError(f"unknown tie_policy {tie_policy!r}")


def hr_at_k(ranks: np.ndarray, k: int) -> float:
    return float((ranks < k).mean()) if len(ranks) else float("nan")


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """NDCG with a single relevant item, so IDCG is 1 and DCG is the gain."""
    if not len(ranks):
        return float("nan")
    gains = np.where(ranks < k, 1.0 / np.log2(ranks + 2.0), 0.0)
    return float(gains.mean())


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    if not len(ranks):
        return float("nan")
    rr = np.where(ranks < k, 1.0 / (ranks + 1.0), 0.0)
    return float(rr.mean())


def compute(ranks: np.ndarray, ks: tuple[int, ...] = DEFAULT_KS) -> dict[str, float]:
    """All metrics at all cutoffs for one slice of evaluation instances."""
    out: dict[str, float] = {"n": float(len(ranks))}
    for k in ks:
        out[f"HR@{k}"] = hr_at_k(ranks, k)
        out[f"NDCG@{k}"] = ndcg_at_k(ranks, k)
        out[f"MRR@{k}"] = mrr_at_k(ranks, k)
    return out


@dataclass
class SlicedResult:
    """Metrics for one model, decomposed by the slices that matter.

    The aggregate row alone is not a result. In a corpus where most targets are
    repeats, a model that only memorizes recent plays posts a strong aggregate
    number while generalizing nothing. ``novel`` is where the recommender claim
    is actually tested, and ``cold_user`` is where the new-uploader claim is.
    """

    name: str
    overall: dict[str, float] = field(default_factory=dict)
    repeat: dict[str, float] = field(default_factory=dict)
    novel: dict[str, float] = field(default_factory=dict)
    cold_user: dict[str, float] = field(default_factory=dict)
    cold_item: dict[str, float] = field(default_factory=dict)

    def as_rows(self, ks: tuple[int, ...] = DEFAULT_KS) -> list[dict]:
        rows = []
        for slice_name in ("overall", "repeat", "novel", "cold_user", "cold_item"):
            metrics = getattr(self, slice_name)
            if metrics and metrics.get("n", 0) > 0:
                rows.append({"model": self.name, "slice": slice_name, **metrics})
        return rows


def evaluate_slices(
    ranks: np.ndarray,
    is_repeat: np.ndarray,
    is_cold_user: np.ndarray | None = None,
    is_cold_item: np.ndarray | None = None,
    name: str = "model",
    ks: tuple[int, ...] = DEFAULT_KS,
) -> SlicedResult:
    """Decompose one array of ranks into the reported slices.

    ``is_repeat`` marks instances whose target the user had already played.

    ``is_cold_item`` marks targets that are **in the vocabulary but absent from
    the training set**. Genuinely out-of-vocabulary targets never reach here:
    ``build_instances`` drops them, because an OOV target has no column to be
    ranked in. This slice is what the Phase 2 transfer ablation turns on, so the
    distinction matters. It also only stays populated if the vocabulary is built
    over the full frame rather than over training events alone; built from
    train, every in-vocab target is train-seen by construction and this slice
    silently empties.
    """
    result = SlicedResult(name=name)
    result.overall = compute(ranks, ks)
    result.repeat = compute(ranks[is_repeat], ks)
    result.novel = compute(ranks[~is_repeat], ks)
    if is_cold_user is not None:
        result.cold_user = compute(ranks[is_cold_user], ks)
    if is_cold_item is not None:
        result.cold_item = compute(ranks[is_cold_item], ks)
    return result
