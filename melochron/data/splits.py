"""Temporal splitting.

Global temporal split only. A random split, or a per-user leave-one-out split,
lets the model see plays that happened after the events it is asked to predict,
which inflates every metric in the report. The invariant this module enforces
is that every training timestamp precedes every evaluation timestamp, and
``assert_no_leakage`` is called by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from melochron import schema


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_ts: int
    #: Users deliberately withheld from training entirely. The Phase 4
    #: cold-start slice: at evaluation the model has never seen these users,
    #: which is exactly the situation a new uploader is in.
    holdout_users: frozenset[str]

    def summary(self) -> dict[str, float | int]:
        return {
            "cutoff_ts": int(self.cutoff_ts),
            "train_events": len(self.train),
            "test_events": len(self.test),
            "train_users": self.train[schema.USER].nunique(),
            "test_users": self.test[schema.USER].nunique(),
            "holdout_users": len(self.holdout_users),
        }


def temporal_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    holdout_user_frac: float = 0.10,
    seed: int = 0,
) -> TemporalSplit:
    """Split on a global timestamp quantile, and withhold some users entirely.

    Two independent held-out axes, because they answer different questions:

    * The **time** axis asks whether the model predicts this user's future,
      which is the deployed question for a returning user.
    * The **user** axis asks whether the model works for someone it has never
      trained on, which is the deployed question for a new uploader. Those
      users are removed from training across all time, not just after cutoff.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    cutoff = int(np.quantile(df[schema.TS].to_numpy(), 1.0 - test_frac))

    users = np.sort(df[schema.USER].unique())
    rng = np.random.default_rng(seed)
    n_holdout = round(len(users) * holdout_user_frac)
    if n_holdout >= len(users):
        # Rounding, not the caller, is usually what does this: on a single-user
        # corpus any fraction above 0.5 rounds to 1 and takes the only user,
        # leaving an empty training set that surfaces much later as the
        # unhelpful "split produced an empty side".
        raise ValueError(
            f"holdout_user_frac={holdout_user_frac} would hold out {n_holdout} "
            f"of {len(users)} users, leaving nothing to train on"
        )
    holdout = (
        frozenset(str(u) for u in rng.choice(users, size=n_holdout, replace=False))
        if n_holdout
        else frozenset()
    )

    is_past = df[schema.TS] < cutoff
    is_holdout = df[schema.USER].astype(str).isin(holdout)

    train = df[is_past & ~is_holdout].reset_index(drop=True)
    test = df[~is_past].reset_index(drop=True)

    return TemporalSplit(train=train, test=test, cutoff_ts=cutoff, holdout_users=holdout)


def assert_no_leakage(split: TemporalSplit) -> None:
    """Raise unless every training event strictly precedes every test event."""
    if split.train.empty or split.test.empty:
        raise ValueError("split produced an empty side")

    max_train = int(split.train[schema.TS].max())
    min_test = int(split.test[schema.TS].min())
    if max_train >= min_test:
        raise AssertionError(
            f"temporal leakage: max train ts {max_train} >= min test ts {min_test}"
        )

    overlap = split.holdout_users & set(split.train[schema.USER].astype(str))
    if overlap:
        raise AssertionError(f"{len(overlap)} holdout users leaked into train")
