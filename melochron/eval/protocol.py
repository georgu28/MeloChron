"""Evaluation instance construction and the batched scoring loop.

Every model and every baseline is scored through this one path, so the
comparison in the README is apples to apples by construction rather than by
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from melochron.data.sessions import Sequences
from melochron.data.vocab import OOV_ID
from melochron.eval import metrics as metrics_mod


class Scorer(Protocol):
    """Anything that can rank the catalog given a user's history so far."""

    name: str

    def score(self, histories: list[np.ndarray], times: list[np.ndarray]) -> np.ndarray:
        """Return a ``(len(histories), vocab_size)`` array of scores."""
        ...


@dataclass
class EvalInstances:
    """One prediction problem per row: history in, next item out."""

    histories: list[np.ndarray]
    history_times: list[np.ndarray]
    targets: np.ndarray
    target_times: np.ndarray
    is_repeat: np.ndarray
    is_cold_user: np.ndarray
    is_cold_item: np.ndarray
    user_ids: list[str]

    def __len__(self) -> int:
        return len(self.targets)

    def summary(self) -> dict[str, float]:
        return {
            "instances": float(len(self)),
            "repeat_frac": float(self.is_repeat.mean()) if len(self) else 0.0,
            "cold_user_frac": float(self.is_cold_user.mean()) if len(self) else 0.0,
            "cold_item_frac": float(self.is_cold_item.mean()) if len(self) else 0.0,
        }


def build_instances(
    seqs: Sequences,
    cutoff_ts: int,
    train_items: set[int],
    holdout_users: frozenset[str],
    max_len: int = 200,
    max_per_user: int = 50,
    seed: int = 0,
) -> EvalInstances:
    """Build evaluation instances from full user sequences.

    ``seqs`` must contain each user's *complete* history, train period and test
    period together. The context for a target is every event strictly before
    it, which includes earlier test-period events. That is not leakage: at
    serving time a user's recent plays are genuinely known. What must never
    leak is *training signal*, and that is enforced separately by fitting only
    on events before ``cutoff_ts``.

    ``max_per_user`` subsamples users with long test periods so that a handful
    of heavy listeners cannot dominate the averages.
    """
    rng = np.random.default_rng(seed)
    histories, history_times, user_ids = [], [], []
    targets, target_times = [], []
    is_repeat, is_cold_user, is_cold_item = [], [], []

    for u, user in enumerate(seqs.user_ids):
        items, times = seqs.items[u], seqs.times[u]
        eligible = np.nonzero(times >= cutoff_ts)[0]
        eligible = eligible[eligible > 0]  # need at least one prior event as context
        if not len(eligible):
            continue

        if len(eligible) > max_per_user:
            eligible = np.sort(rng.choice(eligible, size=max_per_user, replace=False))

        cold = user in holdout_users
        for pos in eligible:
            target = int(items[pos])
            # An OOV target is unrankable: it has no column in the score matrix
            # and every model would score it identically. Excluding it keeps the
            # metric meaningful; the count is reported in the run summary.
            if target == OOV_ID:
                continue

            start = max(0, pos - max_len)
            hist = items[start:pos]
            histories.append(hist)
            history_times.append(times[start:pos])
            user_ids.append(user)
            targets.append(target)
            target_times.append(int(times[pos]))
            is_repeat.append(bool(np.any(items[:pos] == target)))
            is_cold_user.append(cold)
            is_cold_item.append(target not in train_items)

    return EvalInstances(
        histories=histories,
        history_times=history_times,
        targets=np.asarray(targets, dtype=np.int64),
        target_times=np.asarray(target_times, dtype=np.int64),
        is_repeat=np.asarray(is_repeat, dtype=bool),
        is_cold_user=np.asarray(is_cold_user, dtype=bool),
        is_cold_item=np.asarray(is_cold_item, dtype=bool),
        user_ids=user_ids,
    )


def rank_all(
    scorer: Scorer,
    instances: EvalInstances,
    batch_size: int = 128,
    tie_policy: str = "pessimistic",
) -> np.ndarray:
    """Rank every instance, in batches.

    Ranks are reduced per batch and the score matrix is discarded immediately.
    Materializing scores for every instance at once would be tens of gigabytes
    at a realistic vocabulary size.
    """
    ranks = np.empty(len(instances), dtype=np.int64)
    for start in range(0, len(instances), batch_size):
        stop = min(start + batch_size, len(instances))
        scores = scorer.score(instances.histories[start:stop], instances.history_times[start:stop])
        ranks[start:stop] = metrics_mod.ranks_from_scores(
            scores, instances.targets[start:stop], tie_policy=tie_policy
        )
    return ranks


def evaluate(
    scorer: Scorer,
    instances: EvalInstances,
    batch_size: int = 128,
    ks: tuple[int, ...] = metrics_mod.DEFAULT_KS,
) -> metrics_mod.SlicedResult:
    """Score a model or baseline and decompose the result into slices."""
    ranks = rank_all(scorer, instances, batch_size=batch_size)
    return metrics_mod.evaluate_slices(
        ranks,
        is_repeat=instances.is_repeat,
        is_cold_user=instances.is_cold_user,
        is_cold_item=instances.is_cold_item,
        name=scorer.name,
        ks=ks,
    )
