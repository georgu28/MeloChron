"""Guards for the adoption label builder.

Every failure mode here is silent on the real corpus: each one still produces a
plausible base rate over 50M rows, and none of them raises. So the fixtures are
tiny and the assertions are exact counts rather than properties.

The one that matters most is `test_encounter_is_the_first_play`. The raw archive
is ordered most-recent-first, so a builder that trusted file order would record
every track's *last* play as the encounter and invert the entire task.
"""

from __future__ import annotations

import numpy as np
import pytest

from melochron.adoption import slices as slicing
from melochron.adoption.corpus import CompactCorpus
from melochron.adoption.labels import (
    DAY,
    build_encounters,
    event_horizon,
    temporal_split,
    time_horizon,
    train_horizon_fits,
    usable,
)

BASE_TS = 1_370_044_800  # 2013-06-01T00:00:00Z


def make_corpus(rows) -> CompactCorpus:
    """Build a corpus from (user, track, ts) triples, sorted the way the store is."""
    users = sorted({r[0] for r in rows})
    tracks = sorted({r[1] for r in rows})
    uidx = {u: i for i, u in enumerate(users)}
    tidx = {t: i for i, t in enumerate(tracks)}

    ordered = sorted(rows, key=lambda r: (uidx[r[0]], r[2]))
    user_code = np.array([uidx[r[0]] for r in ordered], dtype=np.int32)
    corpus = CompactCorpus(
        user_code=user_code,
        track_code=np.array([tidx[r[1]] for r in ordered], dtype=np.int32),
        ts=np.array([r[2] for r in ordered], dtype=np.int32),
        user_offsets=np.searchsorted(user_code, np.arange(len(users) + 1)).astype(np.int64),
        users=np.array(users),
        tracks=np.array(tracks),
    )
    corpus.validate()
    return corpus


def by_pair(table, corpus):
    """Index the encounter table by (user id, track id) for readable assertions."""
    return {
        (str(corpus.users[u]), str(corpus.tracks[t])): i
        for i, (u, t) in enumerate(zip(table.user_code, table.track_code, strict=True))
    }


class TestEncounters:
    def test_one_encounter_per_distinct_pair(self):
        rows = [("u1", "a", BASE_TS + i * 60) for i in range(5)]
        rows += [("u1", "b", BASE_TS + 500), ("u2", "a", BASE_TS + 600)]
        table = build_encounters(make_corpus(rows))

        assert len(table) == 3  # (u1,a), (u1,b), (u2,a)

    def test_encounter_is_the_first_play(self):
        """The archive is stored most-recent-first. If that order were inherited,
        the last play would be recorded as the encounter and every label would
        invert."""
        rows = [("u1", "a", BASE_TS + t) for t in (0, 60, 120, 180)]
        corpus = make_corpus(rows)
        table = build_encounters(corpus)

        assert table.encounter_ts.tolist() == [BASE_TS]
        assert table.encounter_pos.tolist() == [0]

    def test_recurrence_is_the_earliest_one_not_the_last(self):
        rows = [("u1", "a", BASE_TS + t) for t in (0, 60, 6000)]
        table = build_encounters(make_corpus(rows))

        assert table.recur_ts.tolist() == [BASE_TS + 60]
        assert table.recur_pos.tolist() == [1]

    def test_a_pair_played_once_never_recurs(self):
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + 60)]
        table = build_encounters(make_corpus(rows))
        pairs = by_pair(table, make_corpus(rows))

        assert table.recur_pos[pairs[("u1", "a")]] == -1
        assert table.recur_ts[pairs[("u1", "a")]] == -1
        assert not table.recurs.any()

    def test_positions_are_per_user_not_global(self):
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + 60)]
        rows += [("u2", "a", BASE_TS + 120), ("u2", "b", BASE_TS + 180)]
        corpus = make_corpus(rows)
        table = build_encounters(corpus)
        pairs = by_pair(table, corpus)

        # u2's first encounter is at position 0 of *their* history, not 2.
        assert table.encounter_pos[pairs[("u2", "a")]] == 0
        assert table.encounter_pos[pairs[("u2", "b")]] == 1

    def test_validate_rejects_a_recurrence_before_its_encounter(self):
        rows = [("u1", "a", BASE_TS + t) for t in (0, 60)]
        corpus = make_corpus(rows)
        table = build_encounters(corpus)
        table.recur_pos = np.array([0], dtype=np.int32)

        with pytest.raises(ValueError, match="not strictly after"):
            table.validate(corpus)


class TestEventHorizon:
    def _corpus(self, recur_at: int, total: int = 500):
        """u1 plays 'a', then filler, with 'a' recurring at position ``recur_at``."""
        rows = [("u1", "a", BASE_TS)]
        for i in range(1, total):
            track = "a" if i == recur_at else f"f{i}"
            rows.append(("u1", track, BASE_TS + i * 60))
        return make_corpus(rows)

    def test_recurrence_exactly_at_n_is_positive(self):
        corpus = self._corpus(recur_at=200)
        table = build_encounters(corpus)
        horizon = event_horizon(corpus, table, n_events=200)
        i = by_pair(table, corpus)[("u1", "a")]

        assert bool(horizon.label[i]) is True

    def test_recurrence_one_past_n_is_negative(self):
        corpus = self._corpus(recur_at=201)
        table = build_encounters(corpus)
        horizon = event_horizon(corpus, table, n_events=200)
        i = by_pair(table, corpus)[("u1", "a")]

        assert bool(horizon.label[i]) is False

    def test_encounter_without_a_full_window_is_unobservable(self):
        """A user with 100 events after an encounter has not been watched long
        enough to call it a non-adoption."""
        corpus = self._corpus(recur_at=-1, total=150)
        table = build_encounters(corpus)
        horizon = event_horizon(corpus, table, n_events=200)

        assert not horizon.observable.any()
        assert usable(table, horizon).sum() == 0

    def test_a_recurrence_outside_the_window_is_a_real_negative(self):
        """Observed and negative is not the same as unobserved. With 260 events
        the window closes at 200, so a recurrence at 210 is a label, not a gap."""
        corpus = self._corpus(recur_at=210, total=260)
        table = build_encounters(corpus)
        horizon = event_horizon(corpus, table, n_events=200)
        i = by_pair(table, corpus)[("u1", "a")]

        assert bool(horizon.observable[i]) is True
        assert bool(horizon.label[i]) is False

    def test_an_unobserved_window_is_dropped_even_when_the_label_is_known(self):
        """The deliberately conservative case, and the one most likely to be
        "fixed" later.

        Here 'a' recurs at position 140 with only 149 events following the
        encounter, so the label *is* determined -- a positive is knowable before
        the window closes. It is still dropped.

        Keeping it would look like free data and would bias the base rate
        upward. Early positives near the end of a history are observable;
        the negatives beside them are not, because a non-recurrence cannot be
        confirmed until the whole window has been watched. Admitting one
        without the other keeps the positives and discards the negatives.
        Requiring full observation costs rows but leaves the surviving base rate
        unbiased for the population it describes.
        """
        corpus = self._corpus(recur_at=140, total=150)
        table = build_encounters(corpus)
        horizon = event_horizon(corpus, table, n_events=200)
        i = by_pair(table, corpus)[("u1", "a")]

        assert bool(horizon.label[i]) is True
        assert bool(horizon.observable[i]) is False
        assert usable(table, horizon).sum() == 0

    def test_window_end_timestamp_is_the_nth_event_after(self):
        corpus = self._corpus(recur_at=-1, total=500)
        table = build_encounters(corpus)
        horizon = event_horizon(corpus, table, n_events=200)
        i = by_pair(table, corpus)[("u1", "a")]

        assert int(horizon.ends_ts[i]) == BASE_TS + 200 * 60


class TestTimeHorizon:
    def _rows(self, gap_seconds: int, trailing_days: int = 90):
        rows = [("u1", "a", BASE_TS), ("u1", "a", BASE_TS + gap_seconds)]
        # Keep the user (and the corpus) alive well past the horizon.
        rows.append(("u1", "z", BASE_TS + trailing_days * DAY))
        return rows

    def test_recurrence_exactly_at_the_boundary_is_positive(self):
        corpus = make_corpus(self._rows(30 * DAY))
        table = build_encounters(corpus)
        horizon = time_horizon(corpus, table, days=30)
        i = by_pair(table, corpus)[("u1", "a")]

        assert bool(horizon.label[i]) is True

    def test_recurrence_one_second_late_is_negative(self):
        corpus = make_corpus(self._rows(30 * DAY + 1))
        table = build_encounters(corpus)
        horizon = time_horizon(corpus, table, days=30)
        i = by_pair(table, corpus)[("u1", "a")]

        assert bool(horizon.label[i]) is False

    def test_corpus_ending_inside_the_horizon_censors_the_row(self):
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + 10 * DAY)]
        corpus = make_corpus(rows)
        table = build_encounters(corpus)
        horizon = time_horizon(corpus, table, days=30)

        assert not horizon.observable.any()

    def test_user_churn_is_separable_from_corpus_end(self):
        """A user who stops listening is a different claim from a dataset that
        stops recording, and the two are reported separately."""
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + DAY)]
        # u2 keeps the corpus alive far past u1's horizon.
        rows += [("u2", "c", BASE_TS + 200 * DAY)]
        corpus = make_corpus(rows)
        table = build_encounters(corpus)
        pairs = by_pair(table, corpus)
        i = pairs[("u1", "a")]

        strict = time_horizon(corpus, table, days=30, require_user_runway=True)
        lenient = time_horizon(corpus, table, days=30, require_user_runway=False)

        assert bool(strict.observable[i]) is False  # u1 left after one day
        assert bool(lenient.observable[i]) is True  # the corpus ran on


class TestSplitAndLeakage:
    def _table(self, n_users=20, per_user=30):
        rows = []
        for u in range(n_users):
            for i in range(per_user):
                rows.append((f"u{u:02d}", f"t{i:02d}", BASE_TS + i * DAY))
        corpus = make_corpus(rows)
        return corpus, build_encounters(corpus)

    def test_no_training_encounter_reaches_the_test_period(self):
        corpus, table = self._table()
        split = temporal_split(table, corpus.n_users, test_frac=0.2, holdout_user_frac=0.1)

        assert (table.encounter_ts[split.is_train] < split.cutoff_ts).all()
        assert (table.encounter_ts[split.is_test] >= split.cutoff_ts).all()

    def test_holdout_users_never_appear_in_train(self):
        corpus, table = self._table()
        split = temporal_split(table, corpus.n_users, holdout_user_frac=0.2, seed=3)

        assert not (split.is_train & split.is_holdout_user).any()
        assert split.is_holdout_user.sum() > 0

    def test_train_rows_whose_horizon_crosses_the_cutoff_are_excluded(self):
        """Labelling a training row from post-cutoff events gives the model a
        signal no deployment could have had."""
        corpus, table = self._table()
        split = temporal_split(table, corpus.n_users, test_frac=0.2, holdout_user_frac=0.0)
        horizon = time_horizon(corpus, table, days=10, require_user_runway=False)

        fits = train_horizon_fits(split, horizon)
        observable_train = split.is_train & horizon.observable

        assert fits.sum() < observable_train.sum()
        assert (horizon.ends_ts[fits] < split.cutoff_ts).all()

    def test_holdout_fraction_that_would_empty_train_is_refused(self):
        corpus, table = self._table(n_users=2)

        with pytest.raises(ValueError, match="leaving nothing to train on"):
            temporal_split(table, corpus.n_users, holdout_user_frac=1.0)


class TestCeiling:
    def test_no_horizon_labels_more_positives_than_the_unbounded_rate(self):
        """The invariant the real run is checked against: applying a horizon can
        only turn positives into negatives, never the reverse."""
        rng = np.random.default_rng(0)
        rows = []
        for u in range(30):
            for i in range(120):
                track = f"t{rng.integers(0, 25)}"
                rows.append((f"u{u:02d}", track, BASE_TS + i * 3600))
        corpus = make_corpus(rows)
        table = build_encounters(corpus)

        unbounded = table.recurs.mean()
        for horizon in (
            event_horizon(corpus, table, n_events=50),
            time_horizon(corpus, table, days=2, require_user_runway=False),
        ):
            keep = usable(table, horizon)
            if keep.sum():
                assert horizon.label[keep].mean() <= unbounded + 1e-9


class TestSlices:
    def test_popularity_decile_orders_by_global_plays(self):
        rows = [("u1", "rare", BASE_TS)]
        rows += [(f"u{i}", "common", BASE_TS + i * 60) for i in range(1, 40)]
        corpus = make_corpus(rows)
        table = build_encounters(corpus)
        decile = slicing.popularity_decile(corpus, table)
        pairs = by_pair(table, corpus)

        assert decile[pairs[("u1", "rare")]] < decile[pairs[("u1", "common")]]

    def test_encounter_ordinal_counts_distinct_tracks_not_plays(self):
        # u1 plays 'a' ten times, then meets 'b'. 'b' is their second encounter.
        rows = [("u1", "a", BASE_TS + i * 60) for i in range(10)]
        rows.append(("u1", "b", BASE_TS + 1000))
        corpus = make_corpus(rows)
        table = build_encounters(corpus)
        ordinal = slicing.encounter_ordinal(table)
        pairs = by_pair(table, corpus)

        assert ordinal[pairs[("u1", "a")]] == 0
        assert ordinal[pairs[("u1", "b")]] == 1

    def test_year_comes_out_as_a_calendar_year(self):
        rows = [("u1", "a", BASE_TS), ("u1", "b", BASE_TS + 60)]
        table = build_encounters(make_corpus(rows))

        assert set(slicing.encounter_year(table).tolist()) == {2013}
