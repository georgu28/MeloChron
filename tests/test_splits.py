"""Leakage guards on the temporal split.

The plan calls these non-negotiable. Both failures they cover are silent and
both are flattering, which is the worst combination: the metrics improve and
nothing errors.
"""

from __future__ import annotations

import pandas as pd
import pytest

from melochron import schema
from melochron.data import splits, synthetic


@pytest.fixture(scope="module")
def events():
    df, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=10, n_months=6, seed=3))
    return df


def test_no_temporal_leakage(events):
    split = splits.temporal_split(events, test_frac=0.15)
    splits.assert_no_leakage(split)
    assert split.train[schema.TS].max() < split.test[schema.TS].min()


def test_holdout_users_absent_from_train(events):
    """The cold-start axis is a *user* partition, not a time cut.

    A global cut at T puts every user's pre-T events into train, including the
    users meant to be held out. If that happens the cold-start slice quietly
    becomes a warm-start slice, which is the single most flattering bug
    available in this project.
    """
    split = splits.temporal_split(events, test_frac=0.15, holdout_user_frac=0.2)
    assert len(split.holdout_users) > 0

    train_users = set(split.train[schema.USER].astype(str))
    assert not (split.holdout_users & train_users)

    # Their history must still exist for evaluation context.
    test_users = set(split.test[schema.USER].astype(str))
    assert split.holdout_users & test_users


def test_assert_no_leakage_catches_temporal_overlap(events):
    """The guard must actually fire, not just pass on good input."""
    split = splits.temporal_split(events, test_frac=0.15)
    # Splice one post-cutoff row into train.
    poisoned = pd.concat([split.train, split.test.head(1)], ignore_index=True)
    bad = splits.TemporalSplit(
        train=poisoned,
        test=split.test,
        cutoff_ts=split.cutoff_ts,
        holdout_users=split.holdout_users,
    )
    with pytest.raises(AssertionError, match="temporal leakage"):
        splits.assert_no_leakage(bad)


def test_assert_no_leakage_catches_holdout_user_in_train(events):
    split = splits.temporal_split(events, test_frac=0.15, holdout_user_frac=0.2)

    # Must be a holdout user that actually has test-period rows: a holdout user
    # whose events all fall before the cutoff contributes nothing to splice in,
    # and the test would vacuously pass.
    candidates = split.holdout_users & set(split.test[schema.USER].astype(str))
    assert candidates, "fixture produced no holdout user with test-period events"
    leaked_user = min(candidates)

    row = split.test[split.test[schema.USER].astype(str) == leaked_user].head(1).copy()
    assert len(row) == 1
    row[schema.TS] = split.train[schema.TS].min()
    poisoned = pd.concat([split.train, row], ignore_index=True)

    bad = splits.TemporalSplit(
        train=poisoned,
        test=split.test,
        cutoff_ts=split.cutoff_ts,
        holdout_users=split.holdout_users,
    )
    with pytest.raises(AssertionError, match="holdout users leaked"):
        splits.assert_no_leakage(bad)


def test_rejects_invalid_test_fraction(events):
    with pytest.raises(ValueError):
        splits.temporal_split(events, test_frac=0.0)
    with pytest.raises(ValueError):
        splits.temporal_split(events, test_frac=1.0)
