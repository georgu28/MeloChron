"""Guards against the synthetic generator collapsing into a degenerate regime.

These exist because it already happened once. An earlier version summed the
recency bonus over every occurrence in the window, which fed a softmax and
became a rich-get-richer loop: each user's effective catalog collapsed from
2,400 items to 15, the repeat rate hit 99.9%, the novel slice emptied, and the
repeat baseline scored a perfect HR@10 of 1.0000.

The dangerous part is that none of that *looks* like a failure. It looks like a
great result. Every assertion here is calibrated to fail loudly in that regime
rather than to pin exact numbers, which would just be a change detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from melochron import schema
from melochron.data import sessions, splits, synthetic, vocab
from melochron.eval import protocol

CONFIG = synthetic.SyntheticConfig(n_users=8, n_months=6, seed=1)


@pytest.fixture(scope="module")
def generated():
    events, catalog = synthetic.generate(CONFIG)
    return events, catalog


@pytest.fixture(scope="module")
def pipeline(generated):
    events, _ = generated
    positives = sessions.filter_positives(events)
    split = splits.temporal_split(positives)
    vc = vocab.build_vocab(positives, min_count=5)
    all_seqs = sessions.build_sequences(positives, vc)
    train_seqs = sessions.build_sequences(split.train, vc)
    return positives, split, vc, all_seqs, train_seqs


def test_events_conform_to_schema(generated):
    events, _ = generated
    schema.validate(events)
    assert len(events) > 1000


def test_catalog_is_not_collapsed(pipeline):
    """The users must collectively exercise a broad slice of the catalog."""
    _, _, vc, _, _ = pipeline
    assert vc.n_items > 100, (
        f"vocabulary collapsed to {vc.n_items} items; the generator is in the "
        "rich-get-richer regime that made the repeat baseline unbeatable"
    )


def test_repeat_rate_is_high_but_not_degenerate(pipeline):
    """Music is replay-heavy, but not to the point of a single-item loop.

    The upper bound is the real guard. A repeat rate approaching 100% means
    users are cycling a handful of tracks, and every metric downstream becomes
    a measurement of that artifact rather than of any model.
    """
    _, _, _, all_seqs, _ = pipeline
    rate = sessions.repeat_rate(all_seqs)
    assert 0.40 < rate < 0.97, f"repeat rate {rate:.1%} is outside the plausible band"


def test_positive_filter_removes_skips(generated):
    """The >30s threshold must actually remove something, or it is a no-op."""
    events, _ = generated
    kept = sessions.filter_positives(events)
    assert 0.4 < len(kept) / len(events) < 0.95


def test_sessions_are_bursty(generated):
    """Plays must cluster into sittings, not spread uniformly."""
    events, _ = generated
    sessionized = sessions.sessionize(events)
    per_session = sessionized.groupby("session_id", observed=True).size()
    assert per_session.mean() > 2.0
    assert sessionized["session_id"].nunique() > 10


def test_catalog_grows_over_time(generated):
    """Some items must be released mid-timeline, or cold-item is impossible.

    With a catalog fixed for all time, every test target was necessarily
    available during training, and no split can produce a cold item.
    """
    _, catalog = generated
    release = catalog["release_ts"].to_numpy()
    assert release.max() > release.min(), "catalog is static; cold-item slice cannot populate"
    assert (release > release.min()).mean() > 0.1


def test_eval_slices_are_populated(pipeline):
    """The reported slices must be non-empty, or the table silently loses rows.

    ``SlicedResult.as_rows`` filters out slices with ``n == 0``, so an empty
    slice does not fail: it just vanishes from the results table. These are the
    two guards requested on the coordination board.
    """
    _, split, _vc, all_seqs, train_seqs = pipeline
    train_items = {int(i) for arr in train_seqs.items for i in arr.tolist()}

    inst = protocol.build_instances(
        all_seqs,
        cutoff_ts=split.cutoff_ts,
        train_items=train_items,
        holdout_users=split.holdout_users,
        max_per_user=50,
    )

    assert len(inst) > 0
    assert inst.is_cold_user.sum() > 0, "no held-out users reached evaluation"
    assert inst.is_repeat.sum() < len(inst), "novel slice is empty; every target is a repeat"


def test_generation_is_deterministic():
    """Same seed, same data. Otherwise no result here is reproducible."""
    a, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=3, n_months=2, seed=7))
    b, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=3, n_months=2, seed=7))
    assert len(a) == len(b)
    assert np.array_equal(a[schema.TS].to_numpy(), b[schema.TS].to_numpy())
    assert a[schema.TRACK].tolist() == b[schema.TRACK].tolist()
