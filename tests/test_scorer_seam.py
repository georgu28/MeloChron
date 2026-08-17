"""Contract test for the seam between the model and the evaluation harness.

This is the one test that spans both lanes: it drives `SASRecScorer` through
`eval/protocol.py` exactly as `scripts/evaluate.py` will, so a change to the
`Scorer` protocol on one side or the adapter on the other fails here rather
than halfway through a training run.

It deliberately asserts on *shape and contract*, not on model quality --- the
model is untrained, and the only quality claim that holds for random weights is
that it performs at chance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melochron.data.sessions import Sequences
from melochron.eval import protocol
from melochron.models.scorer import build_scorer, left_pad

N_ITEMS = 60


@pytest.fixture
def seqs() -> Sequences:
    rng = np.random.default_rng(0)
    t0 = 1_600_000_000
    users, items, times, sessions = [], [], [], []
    for u in range(12):
        n = int(rng.integers(20, 60))
        items.append(rng.integers(2, N_ITEMS, size=n).astype(np.int64))
        # Bursts inside a session, long gaps between them, so the time channel
        # sees a realistic spread of buckets rather than one constant delta.
        gaps = np.where(
            rng.random(n) < 0.7, rng.integers(30, 300, n), rng.integers(3600, 400_000, n)
        )
        times.append((t0 + np.cumsum(gaps)).astype(np.int64))
        sessions.append(np.zeros(n, dtype=np.int64))
        users.append(f"u{u}")
    return Sequences(user_ids=users, items=items, times=times, sessions=sessions)


@pytest.fixture
def instances(seqs: Sequences) -> protocol.EvalInstances:
    cutoff = int(np.median(np.concatenate(seqs.times)))
    return protocol.build_instances(
        seqs,
        cutoff_ts=cutoff,
        train_items=set(range(2, N_ITEMS - 8)),
        holdout_users=frozenset({"u0", "u1"}),
        max_len=200,
    )


@pytest.fixture
def scorer(instances):
    torch.manual_seed(0)
    _, _, s = build_scorer(N_ITEMS, d_model=32, n_heads=4, n_blocks=2, max_len=200)
    return s


def test_left_pad_puts_recent_events_last() -> None:
    padded, length = left_pad([np.array([7, 8, 9]), np.array([4])], max_len=10)
    assert length == 3
    assert padded[0].tolist() == [7, 8, 9]
    assert padded[1].tolist() == [0, 0, 4], "short sequence was not left-padded"


def test_left_pad_keeps_the_most_recent_events_when_truncating() -> None:
    padded, length = left_pad([np.arange(1, 11)], max_len=4)
    assert length == 4
    assert padded[0].tolist() == [7, 8, 9, 10], "truncation dropped the recent end"


def test_score_matrix_has_full_vocab_width(scorer, instances) -> None:
    """`ranks_from_scores` indexes by raw vocab id, so width must be n_items."""
    scores = scorer.score(instances.histories[:5], instances.history_times[:5])
    assert scores.shape == (5, N_ITEMS)


def test_pad_and_oov_are_never_rankable(scorer, instances) -> None:
    scores = scorer.score(instances.histories[:8], instances.history_times[:8])
    assert np.isneginf(scores[:, 0]).all(), "PAD could be recommended"
    assert np.isneginf(scores[:, 1]).all(), "OOV could be recommended"
    assert np.isfinite(scores[:, 2:]).all(), "a real item scored non-finite"


def test_scoring_is_deterministic_in_eval_mode(scorer, instances) -> None:
    a = scorer.score(instances.histories[:5], instances.history_times[:5])
    b = scorer.score(instances.histories[:5], instances.history_times[:5])
    assert np.array_equal(a, b), "dropout is still active, or state leaks between calls"


def test_batching_does_not_change_scores(instances) -> None:
    """A batch boundary must not alter a row --- padding is per batch."""
    torch.manual_seed(0)
    _, _, big = build_scorer(N_ITEMS, d_model=32, n_heads=4, n_blocks=2, max_len=200)
    big.batch_size = 64
    torch.manual_seed(0)
    _, _, small = build_scorer(N_ITEMS, d_model=32, n_heads=4, n_blocks=2, max_len=200)
    small.batch_size = 3

    h, t = instances.histories[:10], instances.history_times[:10]
    np.testing.assert_allclose(big.score(h, t), small.score(h, t), rtol=1e-5, atol=1e-5)


def test_time_channel_changes_the_prediction(scorer, instances) -> None:
    """If flattening every gap changes nothing, the time encoding is dead code."""
    h, t = instances.histories[:5], instances.history_times[:5]
    real = scorer.score(h, t)
    flat = scorer.score(h, [np.full_like(x, x[0]) for x in t])
    assert not np.array_equal(real, flat)


def test_full_evaluation_runs_and_populates_every_slice(scorer, instances) -> None:
    result = protocol.evaluate(scorer, instances, batch_size=64)
    rows = {r["slice"]: r for r in result.as_rows()}

    assert {"overall", "repeat", "novel"} <= set(rows), f"missing slices: {sorted(rows)}"
    for name, row in rows.items():
        assert 0.0 <= row["HR@10"] <= 1.0, f"{name} HR@10 out of range"
        assert 0.0 <= row["NDCG@10"] <= 1.0, f"{name} NDCG@10 out of range"
    assert rows["repeat"]["n"] + rows["novel"]["n"] == rows["overall"]["n"]


def test_text_variant_scores_through_the_same_seam(instances) -> None:
    """The transfer variants must evaluate through the identical path.

    If the text variants needed a different harness the ablation table would be
    comparing evaluation code as much as representations.
    """
    torch.manual_seed(0)
    text = torch.randn(N_ITEMS, 16)
    text[0].zero_()
    text[1].zero_()
    _, _, text_scorer = build_scorer(
        N_ITEMS,
        variant="text_frozen",
        text_vectors=text,
        d_model=32,
        n_heads=4,
        n_blocks=2,
        max_len=200,
    )

    scores = text_scorer.score(instances.histories[:5], instances.history_times[:5])
    assert scores.shape == (5, N_ITEMS)
    assert np.isfinite(scores[:, 2:]).all()

    result = protocol.evaluate(text_scorer, instances, batch_size=64)
    assert 0.0 <= result.overall["HR@10"] <= 1.0
    assert result.cold_item["n"] > 0, "the unseen-item slice is empty, so transfer is untested"


def test_untrained_model_scores_near_chance(scorer, instances) -> None:
    """A sanity floor. Far above chance from random weights means leakage."""
    result = protocol.evaluate(scorer, instances, batch_size=64)
    chance = 10.0 / (N_ITEMS - 2)
    assert result.overall["HR@10"] < 4 * chance, (
        f"untrained HR@10={result.overall['HR@10']:.3f} is implausibly far above "
        f"chance={chance:.3f}; suspect target leakage into the history"
    )
