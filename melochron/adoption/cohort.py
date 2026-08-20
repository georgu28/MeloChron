"""The fixed evaluation cohort every model is scored on.

The rule the previous project held itself to and this one inherits: a results
table is only readable if every column met the *same instances*. Scoring
baselines on all seven million test rows and the encoder on whatever fitted in
memory would make the columns incomparable while looking like a table.

**Sampled by user, not by row.** Taking a random 500,000 rows would slice users
in half, which breaks two things: bootstrapping resamples whole users, and the
encoder needs a user's history intact to build a representation. Whole users
also give the per-user baselines a fair shot -- a user prior estimated from a
random third of someone's encounters is not the prior a deployment would hold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Cohort:
    """Row indices into the encounter table, plus how they were chosen."""

    rows: np.ndarray  # int64 indices into the encounter table
    users: np.ndarray  # the sampled user codes
    seed: int
    target: int

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "rows.npy", self.rows)
        np.save(path / "users.npy", self.users)
        (path / "cohort.json").write_text(
            json.dumps(
                {
                    "rows": len(self),
                    "users": int(self.users.shape[0]),
                    "seed": self.seed,
                    "target": self.target,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Cohort:
        meta = json.loads((path / "cohort.json").read_text(encoding="utf-8"))
        return cls(
            rows=np.load(path / "rows.npy"),
            users=np.load(path / "users.npy"),
            seed=meta["seed"],
            target=meta["target"],
        )


def build(
    user_code: np.ndarray,
    eligible: np.ndarray,
    target: int = 500_000,
    seed: int = 0,
) -> Cohort:
    """Draw whole users at random until ``target`` eligible rows are covered.

    ``eligible`` should already be test rows that carry a trustworthy label.
    Users are shuffled and taken in order, so the cohort is a plain random
    sample of users rather than one weighted toward heavy listeners -- the
    alternative, sampling rows, would over-represent exactly the people with the
    most encounters.
    """
    rows = np.flatnonzero(eligible)
    if rows.shape[0] == 0:
        raise ValueError("no eligible rows to draw a cohort from")

    users_present = np.unique(user_code[rows])
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(users_present)

    counts = np.bincount(user_code[rows], minlength=int(user_code.max()) + 1)
    running = np.cumsum(counts[shuffled])
    take = int(np.searchsorted(running, target, side="left")) + 1
    take = min(take, shuffled.shape[0])
    chosen = np.sort(shuffled[:take])

    keep = np.isin(user_code[rows], chosen)
    return Cohort(rows=rows[keep], users=chosen, seed=seed, target=target)


def slice_report(slices: dict[str, np.ndarray], labels: np.ndarray) -> list[dict]:
    """Row and positive counts per slice, so thin slices are visible up front."""
    out = []
    for name, mask in slices.items():
        n = int(mask.sum())
        positives = int(labels[mask].sum()) if n else 0
        out.append(
            {
                "slice": name,
                "n": n,
                "positives": positives,
                "base_rate": round(positives / n, 4) if n else 0.0,
            }
        )
    return out
