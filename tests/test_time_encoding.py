"""Bucketing invariants for the time-delta encoding."""

from __future__ import annotations

import torch

from melochron.models.time_encoding import (
    BOUNDARIES,
    N_BUCKETS,
    NO_PREDECESSOR_BUCKET,
    PAD_BUCKET,
    TimeDeltaEncoding,
    bucketize,
    deltas_from_timestamps,
)

MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def test_bucketing_is_monotonic() -> None:
    """Longer gaps must never land in a lower bucket."""
    deltas = torch.tensor([[0, 1, 30, 5 * MINUTE, 2 * HOUR, 3 * DAY, 400 * DAY]])
    buckets = bucketize(deltas)
    assert torch.all(buckets[:, 1:] >= buckets[:, :-1])


def test_buckets_stay_in_range() -> None:
    deltas = torch.tensor([[0, 10_000 * DAY, -5]])
    buckets = bucketize(deltas)
    assert buckets.min() >= 0 and buckets.max() < N_BUCKETS


def test_session_scale_gaps_are_separated() -> None:
    """The gaps that matter most must not collapse into one bucket.

    Back-to-back plays, a short break, and a next-day return are three
    different listening situations; if the encoding cannot tell them apart it
    is not earning its parameters.
    """
    buckets = bucketize(torch.tensor([[3, 3 * MINUTE, 45 * MINUTE, DAY]]))
    assert len(set(buckets.flatten().tolist())) == 4


def test_negative_gaps_are_clamped_not_raised() -> None:
    """Out-of-order scrobbles happen; they must not kill a run.

    Positions 1 and 2, not 0 --- position 0 is the no-predecessor slot and is
    overridden before clamping ever applies.
    """
    buckets = bucketize(torch.tensor([[500, -100, 0]]))
    assert buckets[0, 1] == buckets[0, 2], "a negative gap did not clamp onto zero"


def test_first_real_event_gets_the_no_predecessor_bucket() -> None:
    """Start-of-history must not look like a genuine multi-year gap.

    This is the trap the helper is defensive about: left-padded timestamps
    padded with zeros give the first real event a delta of ~1.7e9 seconds.
    """
    ts = torch.tensor([[0, 0, 1_700_000_000, 1_700_000_060]])
    mask = torch.tensor([[False, False, True, True]])
    buckets = bucketize(deltas_from_timestamps(ts), mask)

    assert buckets[0, 2] == NO_PREDECESSOR_BUCKET, "start of history was read as a 55-year gap"
    assert buckets[0, 3] not in (PAD_BUCKET, NO_PREDECESSOR_BUCKET)


def test_no_predecessor_is_distinct_from_a_zero_second_gap() -> None:
    """'History starts here' and 'played back to back' are opposite signals."""
    buckets = bucketize(torch.tensor([[0, 0]]))
    assert buckets[0, 0] == NO_PREDECESSOR_BUCKET
    assert buckets[0, 1] != NO_PREDECESSOR_BUCKET


def test_no_predecessor_survives_however_the_caller_padded() -> None:
    """Padding timestamps with 0 or with the first real value must agree."""
    mask = torch.tensor([[False, True, True]])
    zero_padded = torch.tensor([[0, 1_700_000_000, 1_700_000_060]])
    self_padded = torch.tensor([[1_700_000_000, 1_700_000_000, 1_700_000_060]])

    a = bucketize(deltas_from_timestamps(zero_padded), mask)
    b = bucketize(deltas_from_timestamps(self_padded), mask)
    assert torch.equal(a, b), "bucketing depends on how the caller padded timestamps"


def test_masked_positions_get_the_pad_bucket() -> None:
    deltas = torch.tensor([[5 * DAY, 5 * DAY, 5 * DAY]])
    mask = torch.tensor([[False, False, True]])
    buckets = bucketize(deltas, mask)

    assert (buckets[0, :2] == PAD_BUCKET).all()
    assert buckets[0, 2] != PAD_BUCKET


def test_real_gaps_never_collide_with_the_pad_bucket() -> None:
    """A zero-second gap is real and must be distinguishable from padding."""
    assert bucketize(torch.tensor([[0]]))[0, 0] != PAD_BUCKET


def test_deltas_from_timestamps() -> None:
    ts = torch.tensor([[1000, 1060, 1160, 1160]])
    deltas = deltas_from_timestamps(ts)

    assert deltas[0, 0] == 0, "first event has no predecessor, so no gap"
    assert deltas[0, 1] == 60
    assert deltas[0, 2] == 100
    assert deltas[0, 3] == 0


def test_pad_embedding_is_zero_and_stays_zero() -> None:
    enc = TimeDeltaEncoding(d_model=8)
    assert torch.count_nonzero(enc.embedding.weight[PAD_BUCKET]) == 0

    mask = torch.tensor([[False, True]])
    out = enc(torch.tensor([[DAY, DAY]]), mask)
    assert torch.count_nonzero(out[0, 0]) == 0


def test_boundaries_are_strictly_increasing() -> None:
    """torch.bucketize silently returns nonsense on unsorted boundaries."""
    assert list(BOUNDARIES) == sorted(BOUNDARIES)
    assert len(set(BOUNDARIES)) == len(BOUNDARIES)
