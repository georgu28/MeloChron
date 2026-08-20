"""Guards for the Phase 2 features, cohort, baselines and metrics.

The centre of gravity is `prefix_similarity`. It computes a per-encounter
history centroid with a bincount and a cumulative sum instead of the obvious
loop, which is fast and easy to get subtly wrong -- an off-by-one in the slot
assignment leaks the encounter itself into its own history and inflates every
similarity. So it is checked against a naive implementation rather than against
properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from melochron.adoption import baselines, features, metrics
from melochron.adoption import cohort as cohorts
from melochron.adoption.corpus import CompactCorpus

BASE_TS = 1_370_044_800


def make_corpus(rows) -> CompactCorpus:
    users = sorted({r[0] for r in rows})
    tracks = sorted({r[1] for r in rows})
    uidx = {u: i for i, u in enumerate(users)}
    tidx = {t: i for i, t in enumerate(tracks)}
    ordered = sorted(rows, key=lambda r: (uidx[r[0]], r[2]))
    user_code = np.array([uidx[r[0]] for r in ordered], dtype=np.int32)
    return CompactCorpus(
        user_code=user_code,
        track_code=np.array([tidx[r[1]] for r in ordered], dtype=np.int32),
        ts=np.array([r[2] for r in ordered], dtype=np.int32),
        user_offsets=np.searchsorted(user_code, np.arange(len(users) + 1)).astype(np.int64),
        users=np.array(users),
        tracks=np.array(tracks),
    )


def naive_similarity(corpus, matrix, user, positions, candidates):
    """The obvious loop, kept as the reference the fast path must match."""
    start, end = int(corpus.user_offsets[user]), int(corpus.user_offsets[user + 1])
    history = np.asarray(corpus.track_code[start:end])
    out = []
    for position, candidate in zip(positions, candidates, strict=True):
        centroid = matrix[history[:position]].sum(axis=0)
        vector = matrix[candidate]
        denominator = np.linalg.norm(centroid) * np.linalg.norm(vector)
        out.append(float(vector @ centroid) / denominator if denominator > 0 else 0.0)
    return np.array(out, dtype=np.float32)


def call_prefix(corpus, matrix, user, positions, candidates):
    starts, cols, values = features.sparse_triples(matrix)
    return features.prefix_similarity(
        corpus,
        starts,
        cols,
        values,
        features.row_norms(matrix),
        matrix.shape[1],
        user,
        np.asarray(positions, dtype=np.int64),
        np.asarray(candidates, dtype=np.int64),
    )


class TestPrefixSimilarity:
    def _setup(self, seed=0, n_tracks=12, dims=6, n_events=40):
        rng = np.random.default_rng(seed)
        rows = [
            ("u1", f"t{int(rng.integers(0, n_tracks)):02d}", BASE_TS + i * 60)
            for i in range(n_events)
        ]
        corpus = make_corpus(rows)
        matrix = np.zeros((corpus.n_tracks, dims), dtype=np.float32)
        for t in range(corpus.n_tracks):
            for d in rng.choice(dims, size=2, replace=False):
                matrix[t, d] = float(rng.random()) + 0.1
        return corpus, matrix

    def test_matches_the_naive_implementation(self):
        corpus, matrix = self._setup()
        positions = np.array([1, 5, 9, 17, 33])
        candidates = np.asarray(corpus.track_code)[positions]

        fast = call_prefix(corpus, matrix, 0, positions, candidates)
        slow = naive_similarity(corpus, matrix, 0, positions, candidates)

        assert fast == pytest.approx(slow, abs=1e-5)

    def test_matches_when_positions_arrive_out_of_order(self):
        """Cohort rows are not sorted by position, and the fast path sorts
        internally; the result must come back in the caller's order."""
        corpus, matrix = self._setup(seed=3)
        positions = np.array([33, 5, 17, 1, 9])
        candidates = np.asarray(corpus.track_code)[positions]

        fast = call_prefix(corpus, matrix, 0, positions, candidates)
        slow = naive_similarity(corpus, matrix, 0, positions, candidates)

        assert fast == pytest.approx(slow, abs=1e-5)

    def test_the_encounter_itself_is_excluded_from_its_own_history(self):
        """The off-by-one that would matter. A user whose only prior play is an
        unrelated genre must score 0, even though the encountered track shares a
        genre with itself."""
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + 60), ("u1", "c", BASE_TS + 120)]
        corpus = make_corpus(rows)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        # order: a(0) b(1) c(2); 'c' shares a genre with 'a' but not with 'b'.
        at_b = call_prefix(corpus, matrix, 0, [1], [1])  # history = {a}, candidate b

        assert at_b[0] == pytest.approx(0.0)

    def test_first_encounter_has_no_history_and_scores_zero(self):
        corpus, matrix = self._setup()
        result = call_prefix(corpus, matrix, 0, [0], [int(corpus.track_code[0])])

        assert result[0] == pytest.approx(0.0)

    def test_similarity_is_one_when_history_is_the_same_genre_vector(self):
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + 60)]
        corpus = make_corpus(rows)
        matrix = np.array([[0.0, 3.0], [0.0, 5.0]], dtype=np.float32)

        result = call_prefix(corpus, matrix, 0, [1], [1])

        assert result[0] == pytest.approx(1.0, abs=1e-5)


class TestShrinkage:
    def test_a_key_with_no_history_falls_back_to_the_prior(self):
        keys = np.array([0, 0, 1], dtype=np.int64)
        labels = np.array([True, True, False])

        rate, seen = baselines.shrunk_rate(keys, labels, size=4, prior=0.3, pseudocount=10.0)

        assert rate[3] == pytest.approx(0.3)
        assert seen[3] == 0

    def test_thin_evidence_is_pulled_toward_the_prior(self):
        """Two-for-two is not evidence of a 100% adopter."""
        keys = np.array([0, 0], dtype=np.int64)
        labels = np.array([True, True])

        rate, _ = baselines.shrunk_rate(keys, labels, size=1, prior=0.3, pseudocount=10.0)

        assert 0.3 < rate[0] < 0.6

    def test_plentiful_evidence_overrides_the_prior(self):
        keys = np.zeros(400, dtype=np.int64)
        labels = np.ones(400, dtype=bool)

        rate, _ = baselines.shrunk_rate(keys, labels, size=1, prior=0.3, pseudocount=10.0)

        assert rate[0] > 0.95


class TestCohort:
    def _users(self, n_users=50, per_user=20):
        return np.repeat(np.arange(n_users, dtype=np.int32), per_user)

    def test_takes_whole_users_never_partial_ones(self):
        user_code = self._users()
        eligible = np.ones(user_code.shape[0], dtype=bool)

        cohort = cohorts.build(user_code, eligible, target=200, seed=0)

        for user in cohort.users:
            in_cohort = int((user_code[cohort.rows] == user).sum())
            available = int((user_code == user).sum())
            assert in_cohort == available

    def test_is_deterministic_for_a_seed(self):
        user_code = self._users()
        eligible = np.ones(user_code.shape[0], dtype=bool)

        a = cohorts.build(user_code, eligible, target=200, seed=7)
        b = cohorts.build(user_code, eligible, target=200, seed=7)

        assert np.array_equal(a.rows, b.rows)

    def test_only_draws_from_eligible_rows(self):
        user_code = self._users()
        eligible = np.zeros(user_code.shape[0], dtype=bool)
        eligible[::3] = True

        cohort = cohorts.build(user_code, eligible, target=100, seed=0)

        assert eligible[cohort.rows].all()

    def test_round_trips_through_disk(self, tmp_path):
        user_code = self._users()
        eligible = np.ones(user_code.shape[0], dtype=bool)
        cohort = cohorts.build(user_code, eligible, target=200, seed=1)
        cohort.save(tmp_path / "c")

        loaded = cohorts.Cohort.load(tmp_path / "c")

        assert np.array_equal(cohort.rows, loaded.rows)
        assert loaded.seed == 1


class TestMetrics:
    def test_a_constant_scorer_earns_exactly_the_base_rate(self):
        """The self-check the real run asserts. If this drifts, PR-AUC is being
        computed wrong and every other number in the table is unreadable."""
        rng = np.random.default_rng(0)
        labels = rng.random(20_000) < 0.36
        scores = np.full(labels.shape[0], 0.5)

        result = metrics.evaluate(labels, scores)

        assert result.pr_auc == pytest.approx(labels.mean(), abs=1e-3)
        assert result.lift == pytest.approx(1.0, abs=1e-2)

    def test_a_perfect_scorer_reaches_one(self):
        labels = np.array([True, False, True, False, True])

        result = metrics.evaluate(labels, labels.astype(float))

        assert result.pr_auc == pytest.approx(1.0)
        assert result.roc_auc == pytest.approx(1.0)

    def test_a_single_class_slice_is_marked_rather_than_scored(self):
        labels = np.zeros(50, dtype=bool)

        result = metrics.evaluate(labels, np.random.default_rng(0).random(50))

        assert np.isnan(result.pr_auc)
        assert result.positives == 0

    def test_bootstrap_over_users_is_wider_than_over_rows(self):
        """Rows inside a user are correlated. Resampling users must produce a
        wider interval than pretending every row is independent, or the
        resampling unit is not doing anything."""
        rng = np.random.default_rng(0)
        users = np.repeat(np.arange(60), 50)
        # Label is constant within a user, which is the correlation that makes
        # the resampling unit matter at all.
        per_user = rng.random(60) < 0.5
        labels = np.repeat(per_user, 50)
        # Deliberately overlapping, so PR-AUC lands below 1 and the interval has
        # somewhere to be wide. Perfectly separable scores pin both intervals to
        # a degenerate [1.0, 1.0] and the comparison tests nothing.
        scores = rng.random(users.shape[0]) + labels * 0.3

        lo_u, hi_u = metrics.bootstrap_pr_auc(labels, scores, users, rounds=80, seed=0)
        lo_r, hi_r = metrics.bootstrap_pr_auc(
            labels, scores, np.arange(users.shape[0]), rounds=80, seed=0
        )

        assert (hi_u - lo_u) > (hi_r - lo_r)

    def test_calibration_error_is_zero_for_a_perfectly_calibrated_scorer(self):
        rng = np.random.default_rng(0)
        probabilities = rng.random(50_000)
        labels = rng.random(50_000) < probabilities

        assert metrics.expected_calibration_error(labels, probabilities) < 0.02

    def test_calibration_error_catches_a_constant_offset(self):
        """The drift Phase 1 measured shows up here: a prior fitted at 0.37 and
        applied to a 0.31 reality is miscalibrated by construction."""
        rng = np.random.default_rng(0)
        labels = rng.random(50_000) < 0.31
        probabilities = np.full(50_000, 0.37)

        assert metrics.expected_calibration_error(labels, probabilities) == pytest.approx(
            0.06, abs=0.01
        )


class TestFeatureMatrix:
    def test_sparse_triples_round_trip_the_matrix(self):
        matrix = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        starts, cols, values = features.sparse_triples(matrix)

        rebuilt = np.zeros_like(matrix)
        for row in range(matrix.shape[0]):
            span = slice(int(starts[row]), int(starts[row + 1]))
            rebuilt[row, cols[span]] = values[span]

        assert np.array_equal(rebuilt, matrix)

    def test_an_all_zero_row_has_zero_norm_not_nan(self):
        matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)

        norms = features.row_norms(matrix)

        assert norms[0] == 0.0
        assert norms[1] == pytest.approx(5.0)
