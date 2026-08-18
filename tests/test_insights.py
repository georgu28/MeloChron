"""Invariants for the Phase 6 insight modules.

Insights are the easiest place in the project to ship something that looks
convincing and means nothing. A drift curve that is really tracking how many
tracks a user played, clusters named after whatever is globally popular, an
attention chart dominated by padding --- none of those crash, none of them look
obviously wrong, and all three would survive a review that only checked shapes.

So the tests here are mostly *negative controls*. The drift tests pin the metric
against the synthetic generator's known taste trajectory, which is the only
ground truth in the project, and separately confirm the metric reads zero when
nothing moved. The archetype test confirms a globally dominant item cannot name
every cluster. The attention tests confirm padding contributes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melochron.data import sessions, synthetic, vocab
from melochron.insights import archetypes, attention, drift
from melochron.models.sasrec import SASRec

WINDOW_DAYS = 60
D_GENRE = 16
N_ITEMS = 40
MAX_LEN = 16
HOUR = 3600
DAY = 86_400


def _genre_vectors(catalog, vc, seed: int = 0) -> np.ndarray:
    """Item vectors that actually carry meaning, derived from the catalog.

    Drift measures movement in learned item-vector space, so testing it against
    an untrained model would only measure noise. Genre vectors stand in for a
    representation that has learned something: items in a genre share a
    direction, which is the structure the generator's taste walk moves through.
    """
    rng = np.random.default_rng(seed)
    per_genre = {}
    for genre in sorted(catalog["genre"].unique()):
        v = rng.normal(size=D_GENRE)
        per_genre[genre] = v / np.linalg.norm(v)

    key_to_genre = {
        vocab.canonical_key(artist, track): genre
        for artist, track, genre in zip(catalog["artist"], catalog["track"], catalog["genre"])
    }

    matrix = np.zeros((len(vc), D_GENRE), dtype=np.float32)
    for item_id, key in enumerate(vc.id_to_key):
        genre = key_to_genre.get(key)
        if genre is not None:
            matrix[item_id] = per_genre[genre]
    return matrix


def _corpus(drift_scale: float, seed: int = 5, n_users: int = 10, n_months: int = 14):
    cfg = synthetic.SyntheticConfig(
        n_users=n_users, n_months=n_months, seed=seed, drift_scale=drift_scale
    )
    taste: list[synthetic.TasteSnapshot] = []
    events, catalog = synthetic.generate(cfg, taste_out=taste)
    positives = sessions.filter_positives(events)
    vc = vocab.build_vocab(positives, min_count=3)
    seqs = sessions.build_sequences(positives, vc)
    return seqs, vc, _genre_vectors(catalog, vc), taste


@pytest.fixture(scope="module")
def drifting():
    return _corpus(drift_scale=0.18)


def _timeline(seqs, vc, matrix):
    origin = min(int(t[0]) for t in seqs.times if len(t))
    return origin, drift.compute(
        seqs, matrix, vocab=vc, window_days=WINDOW_DAYS, min_events=10, origin=origin
    )


def test_drift_recovers_the_generators_true_taste_trajectory(drifting) -> None:
    seqs, vc, matrix, taste = drifting
    origin, timeline = _timeline(seqs, vc, matrix)

    # True displacement: cosine distance from each user's first recorded taste
    # vector, bucketed into the same windows the metric used.
    truth: dict[tuple[str, int], float] = {}
    first: dict[str, np.ndarray] = {}
    for snap in taste:
        first.setdefault(snap.user_id, snap.taste)
        window = int(drift.window_index(np.array([snap.ts]), origin, WINDOW_DAYS)[0])
        truth.setdefault(
            (snap.user_id, window), float(1.0 - np.dot(snap.taste, first[snap.user_id]))
        )

    measured, actual = [], []
    for row in timeline.windows:
        if row.sparse or row.displacement is None:
            continue
        if (row.user_id, row.window) in truth:
            measured.append(row.displacement)
            actual.append(truth[(row.user_id, row.window)])

    assert len(measured) > 20, "too few placed windows to correlate against"
    r = float(np.corrcoef(measured, actual)[0, 1])
    # Measured drift saturates where true drift does not --- items only span a
    # handful of genre directions, so the centroid cannot keep walking forever.
    # The relationship is real but not linear, hence a moderate floor.
    assert r > 0.4, f"drift does not track the generator's taste walk (pearson r={r:.3f})"


def test_measured_drift_increases_with_the_generators_drift_scale() -> None:
    calm = _corpus(drift_scale=0.02, n_users=8, n_months=12)
    restless = _corpus(drift_scale=0.45, n_users=8, n_months=12)

    calm_step = _timeline(*calm[:3])[1].summary()["mean_step"]
    restless_step = _timeline(*restless[:3])[1].summary()["mean_step"]

    assert restless_step > calm_step * 2, (
        f"drift metric barely responds to actual drift ({calm_step} vs {restless_step})"
    )


def test_a_constant_history_shows_no_drift() -> None:
    # The failure this guards against: a metric that is really measuring how
    # much a user listened would report drift here, because there is plenty of
    # listening and no change whatsoever.
    n = 600
    seqs = sessions.Sequences(
        user_ids=["steady"],
        items=[np.full(n, 7, dtype=np.int64)],
        times=[np.arange(n, dtype=np.int64) * HOUR],
        sessions=[np.arange(n, dtype=np.int64) // 10],
    )
    matrix = np.zeros((16, 4), dtype=np.float32)
    matrix[7] = [1.0, 0.0, 0.0, 0.0]

    timeline = drift.compute(seqs, matrix, window_days=1, min_events=5)
    steps = [w.step for w in timeline.windows if w.step is not None]

    assert steps, "no windows were placed, so the test proved nothing"
    assert max(steps) < 1e-6, f"a constant history registered drift of {max(steps)}"


def test_reserved_ids_are_excluded_from_centroids() -> None:
    # Window 0 is clean; window 1 is the same music plus a pile of OOV. If OOV
    # rows were averaged in, the centroid would move and this would read as
    # taste change when it is really vocabulary coverage.
    clean = np.array([5, 6, 7] * 10, dtype=np.int64)
    polluted = np.concatenate([clean, np.full(60, vocab.OOV_ID, dtype=np.int64)])
    times = np.concatenate(
        [
            np.full(len(clean), 0, dtype=np.int64),
            np.full(len(polluted), DAY, dtype=np.int64),
        ]
    )

    matrix = np.zeros((16, 4), dtype=np.float32)
    matrix[5] = [1.0, 0.0, 0.0, 0.0]
    matrix[6] = [0.0, 1.0, 0.0, 0.0]
    matrix[7] = [0.0, 0.0, 1.0, 0.0]
    matrix[vocab.OOV_ID] = [0.0, 0.0, 0.0, 1.0]

    windows = drift.user_timeline(
        "u",
        np.concatenate([clean, polluted]),
        times,
        drift.normalize_rows(matrix),
        origin=0,
        window_days=1,
        min_events=5,
    )

    placed = [w for w in windows if not w.sparse]
    assert len(placed) == 2, "expected both windows to be placed"
    assert placed[1].step is not None
    assert placed[1].step < 1e-6, (
        f"unknown tracks moved the centroid by {placed[1].step}, so OOV is leaking in"
    )


def test_thin_windows_are_reported_as_gaps_not_points() -> None:
    items = np.array([5] * 20 + [6] * 2 + [7] * 20, dtype=np.int64)
    times = np.concatenate(
        [
            np.zeros(20, dtype=np.int64),
            np.full(2, DAY, dtype=np.int64),
            np.full(20, 2 * DAY, dtype=np.int64),
        ]
    )
    matrix = np.zeros((16, 4), dtype=np.float32)
    matrix[5] = [1.0, 0.0, 0.0, 0.0]
    matrix[6] = [0.0, 1.0, 0.0, 0.0]
    matrix[7] = [0.0, 0.0, 1.0, 0.0]

    windows = drift.user_timeline(
        "u", items, times, drift.normalize_rows(matrix), origin=0, window_days=1, min_events=10
    )

    assert windows[1].sparse, "a 2-play window was given a centroid"
    assert windows[1].step is None and windows[1].displacement is None
    # The step across the gap must say it jumped two window slots, not one.
    assert windows[2].since_previous == 2, "a step across a dormancy is being reported as adjacent"


def _hand_built_table() -> archetypes.SessionTable:
    """Two obvious groups, plus one item that dominates every session."""
    rng = np.random.default_rng(0)
    rows, items = [], []
    dominant = 99
    for group, members in ((0, [2, 3, 4]), (1, [5, 6, 7])):
        for _ in range(30):
            # Jittered rather than identical: a table of duplicate points makes
            # any k above 2 degenerate, which is a property of the fixture and
            # not of the code under test.
            vector = rng.normal(scale=0.05, size=4).astype(np.float32)
            vector[group] += 1.0
            rows.append(vector / np.linalg.norm(vector))
            items.append(np.array(members + [dominant] * 3, dtype=np.int64))

    n = len(rows)
    return archetypes.SessionTable(
        vectors=np.stack(rows),
        user_ids=[f"u{i}" for i in range(n)],
        session_ids=np.arange(n, dtype=np.int64),
        start_ts=np.arange(n, dtype=np.int64) * HOUR,
        length=np.full(n, 6, dtype=np.int64),
        n_unique=np.full(n, 4, dtype=np.int64),
        repeat_frac=np.zeros(n, dtype=np.float32),
        median_gap=np.full(n, 60.0, dtype=np.float32),
        hour=np.zeros(n, dtype=np.int64),
        items=items,
        n_total=n,
    )


def test_archetype_labels_are_not_the_globally_most_common_item() -> None:
    result = archetypes.compute(_hand_built_table(), k_range=(2, 2), seed=0, min_support=3)

    assert result.k == 2
    tops = [[row["label"] for row in a.top_items] for a in result.archetypes]
    assert len(tops) == 2

    dominant = "99"
    assert not all(dominant in labels[:1] for labels in tops), (
        "the globally most-played item named every cluster, so labelling is by count not lift"
    )
    assert set(tops[0]) != set(tops[1]), "both clusters got the same labels"


def test_archetypes_are_deterministic_under_a_fixed_seed() -> None:
    table = _hand_built_table()
    a = archetypes.compute(table, k_range=(2, 3), seed=0)
    b = archetypes.compute(table, k_range=(2, 3), seed=0)

    assert a.k == b.k
    assert np.array_equal(a.labels, b.labels), "same seed produced a different clustering"


def test_sessions_shorter_than_the_floor_are_dropped(drifting) -> None:
    seqs, _, matrix, _ = drifting
    table = archetypes.build_sessions(seqs, matrix, min_session_len=5, max_sessions=None)

    assert len(table) > 0
    assert table.length.min() >= 5, "a session below the length floor survived"
    assert not table.sampled, "nothing was subsampled, but the table claims it was"


@pytest.fixture
def encoder() -> SASRec:
    torch.manual_seed(0)
    return SASRec(n_items=N_ITEMS, d_model=32, n_heads=2, n_blocks=2, max_len=MAX_LEN).eval()


def test_attention_trace_ignores_padded_history(encoder: SASRec) -> None:
    short = np.array([5, 6, 7], dtype=np.int64)
    long = np.arange(10, 22, dtype=np.int64)
    times = [np.arange(len(short)) * HOUR, np.arange(len(long)) * HOUR]

    traces = attention.trace(
        encoder, [short, long], times, user_ids=["short", "long"], top_k=MAX_LEN
    )

    by_user = {}
    for t in traces:
        by_user.setdefault(t.user_id, []).append(t)

    assert by_user["short"][0].history_len == 3, "padding was counted as history"
    assert by_user["long"][0].history_len == 12

    for t in traces:
        recencies = [item.recency for item in t.top]
        assert max(recencies) < t.history_len, "attention was attributed to a padded position"
        assert 0 in recencies, "the most recent play received no attention at all"


def test_attention_weights_sum_to_one_across_the_history(encoder: SASRec) -> None:
    history = np.arange(10, 22, dtype=np.int64)
    times = [np.arange(len(history)) * HOUR]

    traces = attention.trace(encoder, [history], times, top_k=MAX_LEN)

    for t in traces:
        total = sum(item.weight for item in t.top)
        assert abs(total - 1.0) < 1e-4, f"block {t.block} attention sums to {total}, not 1"


def test_an_untrained_model_shows_no_attention_concentration(encoder: SASRec) -> None:
    # The counterpart to test_untrained_model_scores_near_chance. Before any
    # training the attention row is flat, so concentration must sit at 1.0. A
    # metric that reports structure here is reporting its own arithmetic.
    history = np.arange(10, 10 + MAX_LEN, dtype=np.int64)
    traces = attention.trace(encoder, [history], [np.arange(MAX_LEN) * HOUR], top_k=4)

    for t in traces:
        assert abs(t.concentration - 1.0) < 0.35, (
            f"untrained attention looks concentrated ({t.concentration:.2f}x uniform)"
        )


def test_attention_refuses_a_model_left_in_training_mode(encoder: SASRec) -> None:
    encoder.train()
    with pytest.raises(ValueError, match="training mode"):
        attention.trace(encoder, [np.array([5, 6, 7])], [np.array([0, HOUR, 2 * HOUR])])


def test_signals_align_by_session_id_not_by_position() -> None:
    # The failure this guards against is silent and total: the table is
    # length-filtered and subsampled, so zipping a corpus-wide statistic onto it
    # pairs every session with somebody else's dwell time and still produces a
    # plausible cluster description.
    table = _hand_built_table()
    n = len(table.user_ids)
    table.session_ids = np.array([50, 10, 30] + list(range(100, 100 + n - 3)), dtype=np.int64)

    ids = np.array([10, 30, 50], dtype=np.int64)
    values = np.array([1.0, 3.0, 5.0])
    aligned = archetypes.align_signal(table, ids, values)

    assert aligned[0] == 5.0, "session 50 did not receive its own value"
    assert aligned[1] == 1.0, "session 10 did not receive its own value"
    assert aligned[2] == 3.0, "session 30 did not receive its own value"
    assert np.isnan(aligned[3]), "a session with no statistic was given someone else's"


def test_clusters_report_attached_signals() -> None:
    table = _hand_built_table()
    table.signals["skip_rate"] = np.linspace(0.0, 1.0, len(table.user_ids))
    result = archetypes.compute(table, k_range=(2, 2), seed=0, min_support=3)

    for archetype in result.archetypes:
        assert "skip_rate" in archetype.signals, "an attached signal was not summarized"
        assert 0.0 <= archetype.signals["skip_rate"] <= 1.0


def test_a_corpus_without_playback_signals_reports_none() -> None:
    # lastfm-1K has no ms_played at all. Nothing may require these fields.
    result = archetypes.compute(_hand_built_table(), k_range=(2, 2), seed=0, min_support=3)
    assert all(not a.signals for a in result.archetypes), "signals invented from nowhere"
