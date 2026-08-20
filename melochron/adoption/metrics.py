"""Scoring an adoption predictor.

PR-AUC is the headline, always printed beside its base rate, because a
precision-recall curve is only interpretable against the rate a coin would get.

**The brief bans headlining AUROC and this module reports it anyway.** That ban
rests on the label being rare -- "low single-digit-to-~15%" -- which is where
AUROC's true-negative-heavy denominator flatters a model. Phase 1 measured the
real base rate at 0.3592, so the premise does not hold: the brief's own worked
example ("PR-AUC of 0.30 at a 0.12 base rate is ~2.5x chance") inverts here,
where 0.30 would be slightly *worse* than chance. AUROC is reported as a
secondary column, and the reason it was let back in is recorded rather than
assumed.

Bootstrapping resamples **users, not rows**. Encounters within one user share a
taste, a listening rate and a catalogue, so treating 500,000 rows from 50,000
users as 500,000 independent draws would produce confidence intervals several
times too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class Score:
    """One model on one slice."""

    slice_name: str
    n: int
    positives: int
    base_rate: float
    pr_auc: float
    lift: float
    roc_auc: float
    pr_auc_lo: float | None = None
    pr_auc_hi: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "slice": self.slice_name,
            "n": self.n,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 4),
            "pr_auc": round(self.pr_auc, 4),
            "lift": round(self.lift, 3),
            "roc_auc": round(self.roc_auc, 4),
            **self.extra,
        }
        if self.pr_auc_lo is not None:
            out["pr_auc_lo"] = round(self.pr_auc_lo, 4)
            out["pr_auc_hi"] = round(self.pr_auc_hi, 4)
        return out


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    users: np.ndarray | None = None,
    slice_name: str = "all",
    bootstrap: int = 0,
    seed: int = 0,
) -> Score:
    """PR-AUC, base rate, lift and AUROC for one set of rows.

    ``lift`` is PR-AUC divided by the base rate: how many times better than
    chance, which is the number that survives being quoted without context.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n = int(labels.shape[0])
    positives = int(labels.sum())

    if n == 0 or positives == 0 or positives == n:
        # A slice with one class has no PR curve. Returning zeros would read as
        # a real measurement, so the degenerate case is marked instead.
        rate = positives / n if n else 0.0
        return Score(slice_name, n, positives, rate, float("nan"), float("nan"), float("nan"))

    base_rate = positives / n
    pr_auc = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))

    score = Score(
        slice_name=slice_name,
        n=n,
        positives=positives,
        base_rate=base_rate,
        pr_auc=pr_auc,
        lift=pr_auc / base_rate,
        roc_auc=roc_auc,
    )

    if bootstrap and users is not None:
        lo, hi = bootstrap_pr_auc(labels, scores, users, rounds=bootstrap, seed=seed)
        score.pr_auc_lo, score.pr_auc_hi = lo, hi
    return score


def bootstrap_pr_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    users: np.ndarray,
    rounds: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for PR-AUC, resampling whole users with replacement.

    Rows inside a user are not independent draws, so the resampling unit is the
    user and every one of their rows travels together.
    """
    rng = np.random.default_rng(seed)

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.concatenate([[True], sorted_users[1:] != sorted_users[:-1], [True]])
    )
    groups = [order[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
    n_groups = len(groups)

    estimates = []
    for _ in range(rounds):
        picked = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([groups[i] for i in picked])
        sample_labels = labels[rows]
        if sample_labels.all() or not sample_labels.any():
            continue
        estimates.append(average_precision_score(sample_labels, scores[rows]))

    if not estimates:
        return float("nan"), float("nan")
    return (
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def evaluate_slices(
    labels: np.ndarray,
    scores: np.ndarray,
    users: np.ndarray,
    slices: dict[str, np.ndarray],
    bootstrap: int = 0,
    seed: int = 0,
    min_rows: int = 200,
) -> list[Score]:
    """Score every named slice, skipping ones too small to mean anything."""
    out = [evaluate(labels, scores, users, "all", bootstrap, seed)]
    for name, mask in slices.items():
        if int(mask.sum()) < min_rows:
            continue
        out.append(evaluate(labels[mask], scores[mask], users[mask], name, bootstrap, seed))
    return out


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Mean absolute gap between predicted probability and observed rate.

    Meaningful here in a way it would not be on a rare label: at a base rate
    near 0.36 a predicted probability is a usable quantity, and the train/test
    drift Phase 1 measured means every train-fitted prior is offset in a known
    direction. Reported rather than corrected.
    """
    labels = np.asarray(labels).astype(bool)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)

    total = 0.0
    for b in range(bins):
        mask = which == b
        count = int(mask.sum())
        if not count:
            continue
        total += count * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(total / labels.shape[0])
