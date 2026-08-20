"""Encounters, horizons, censoring and splits.

This is where correctness lives. Every number the project reports is downstream
of the arrays built here, and the failure modes are all silent: a horizon that
counts the wrong events still produces a plausible base rate, and an encounter
labelled negative because nobody watched long enough is indistinguishable from
one the user genuinely ignored.

The two rules worth stating in full:

**Censoring.** An encounter is labelable only if its whole horizon was observed.
Rows that fail are *dropped and counted*, never relabelled negative -- doing that
converts real positives into fake negatives and inflates every metric.

**Train horizons stay inside the training period.** A training encounter whose
horizon reaches past the split boundary would be labelled using events from the
test period. Nothing errors; the model just gets a training signal no deployment
could have had.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from melochron.adoption.corpus import PLAUSIBLE_FLOOR, CompactCorpus

DAY = 86_400

#: The brief's primary horizon: recurrence within the user's next N events.
DEFAULT_EVENT_N = 200

#: The brief's secondary horizon, in days.
DEFAULT_TIME_DAYS = 30


@dataclass
class EncounterTable:
    """One row per (user, track): the first time that user met that track.

    ``recur_pos`` / ``recur_ts`` describe the *earliest* recurrence, which is
    all either horizon needs -- if the second play falls outside the horizon,
    every later one does too. Both are -1 when the pair never recurs.

    Positions are indices into the user's own history, not the global arrays, so
    ``encounter_pos`` doubles as "how many tracks this user had already played".
    """

    user_code: np.ndarray  # int32
    track_code: np.ndarray  # int32
    encounter_ts: np.ndarray  # int32, unix seconds
    encounter_pos: np.ndarray  # int32, index within the user's history
    recur_pos: np.ndarray  # int32, -1 when the pair never recurs
    recur_ts: np.ndarray  # int32, -1 when the pair never recurs

    def __len__(self) -> int:
        return int(self.user_code.shape[0])

    @property
    def recurs(self) -> np.ndarray:
        """Does this pair ever recur, at any distance? The unbounded label."""
        return self.recur_pos >= 0

    def validate(self, corpus: CompactCorpus) -> None:
        """Raise if an invariant the horizons depend on is broken."""
        n = len(self)
        for name in ("track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} has a different length to user_code")

        if (self.encounter_pos < 0).any():
            raise ValueError("negative encounter position")

        recurs = self.recurs
        if (self.recur_pos[recurs] <= self.encounter_pos[recurs]).any():
            # The recurrence is by construction a *later* play of the same pair.
            # If this fires, the second-occurrence search found the wrong row.
            raise ValueError("a recurrence is not strictly after its encounter")
        if (self.recur_ts[recurs] < self.encounter_ts[recurs]).any():
            raise ValueError("a recurrence predates its encounter")

        counts = np.diff(corpus.user_offsets)
        if (self.encounter_pos >= counts[self.user_code]).any():
            raise ValueError("encounter position runs past the end of a user's history")


def build_encounters(corpus: CompactCorpus) -> EncounterTable:
    """Find every (user, track) first meeting and its earliest recurrence.

    Done with one global stable sort rather than a per-user Python loop. The
    corpus is already ordered by (user, ts), so a *stable* sort on the pair key
    leaves each group's members in ascending time order -- which makes the first
    two entries of every group exactly the encounter and its earliest
    recurrence. 119,140 iterations of numpy would work too and take an order of
    magnitude longer.
    """
    user_code = np.asarray(corpus.user_code)
    track_code = np.asarray(corpus.track_code)
    ts = np.asarray(corpus.ts)

    # Pair key. Both components are non-negative and the product fits int64:
    # 119,139 * 56,512 is ~6.7e9, far below the 9.2e18 ceiling.
    key = user_code.astype(np.int64) * corpus.n_tracks + track_code
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    del key

    # Group boundaries in the sorted order: a new pair starts wherever the key
    # changes, plus position 0.
    starts = np.empty(sorted_key.shape[0], dtype=bool)
    starts[0] = True
    np.not_equal(sorted_key[1:], sorted_key[:-1], out=starts[1:])
    group_start = np.flatnonzero(starts)
    del sorted_key, starts

    first_idx = order[group_start]

    # A group has a recurrence iff it holds more than one row, which is true
    # unless the next group starts immediately after this one.
    group_size = np.diff(np.append(group_start, order.shape[0]))
    has_recurrence = group_size > 1
    second_idx = np.full(group_start.shape[0], -1, dtype=np.int64)
    second_idx[has_recurrence] = order[group_start[has_recurrence] + 1]
    del order, group_start, group_size

    users = user_code[first_idx]
    starts_of_user = np.asarray(corpus.user_offsets)[users]

    recur_pos = np.full(first_idx.shape[0], -1, dtype=np.int32)
    recur_ts = np.full(first_idx.shape[0], -1, dtype=np.int32)
    recur_pos[has_recurrence] = (
        second_idx[has_recurrence] - starts_of_user[has_recurrence]
    ).astype(np.int32)
    recur_ts[has_recurrence] = ts[second_idx[has_recurrence]]

    return EncounterTable(
        user_code=users,
        track_code=track_code[first_idx],
        encounter_ts=ts[first_idx],
        encounter_pos=(first_idx - starts_of_user).astype(np.int32),
        recur_pos=recur_pos,
        recur_ts=recur_ts,
    )


@dataclass
class Horizon:
    """Labels and observability for one horizon definition."""

    name: str
    label: np.ndarray  # bool: did it recur inside the horizon
    observable: np.ndarray  # bool: was the whole horizon watched
    #: When the horizon closes, in unix seconds. Needed to keep a training
    #: encounter's horizon inside the training period. -1 where unobservable.
    ends_ts: np.ndarray

    def summary(self, extra: dict | None = None) -> dict:
        n = int(self.label.shape[0])
        obs = int(self.observable.sum())
        pos = int((self.label & self.observable).sum())
        return {
            "horizon": self.name,
            "encounters": n,
            "labelable": obs,
            "censored_out": n - obs,
            "censored_frac": round((n - obs) / n, 4) if n else 0.0,
            "positives": pos,
            "base_rate": round(pos / obs, 4) if obs else 0.0,
            **(extra or {}),
        }


def event_horizon(
    corpus: CompactCorpus, table: EncounterTable, n_events: int = DEFAULT_EVENT_N
) -> Horizon:
    """Recurrence within the user's next ``n_events`` listening events.

    Fair across light and heavy listeners, which is why the brief makes it
    primary: a month means something different to someone playing 40 tracks a
    day than to someone playing four a week.

    Unobservable when fewer than ``n_events`` of the user's events follow the
    encounter. Those encounters are dropped, not called negative.
    """
    counts = np.diff(corpus.user_offsets)[table.user_code]
    remaining = counts - 1 - table.encounter_pos
    observable = remaining >= n_events

    label = table.recurs & ((table.recur_pos - table.encounter_pos) <= n_events)

    # The timestamp at which the window closes: the n_events-th event after the
    # encounter. Only defined where the window actually fits.
    ends_ts = np.full(len(table), -1, dtype=np.int32)
    idx = (
        np.asarray(corpus.user_offsets)[table.user_code[observable]]
        + table.encounter_pos[observable]
        + n_events
    )
    ends_ts[observable] = np.asarray(corpus.ts)[idx]

    return Horizon(f"event_n{n_events}", label, observable, ends_ts)


def time_horizon(
    corpus: CompactCorpus,
    table: EncounterTable,
    days: int = DEFAULT_TIME_DAYS,
    require_user_runway: bool = True,
) -> Horizon:
    """Recurrence within ``days`` of the encounter.

    Two different things can make this unobservable and they are not the same
    claim:

    * **corpus runway** -- the encounter sits inside the trailing window, so the
      data simply stops before the horizon closes. Purely an artefact of when
      the dataset was cut.
    * **user runway** -- the user's own last event falls inside the horizon.
      Here the data continues; the *user* stopped. Calling that a non-adoption
      asserts they chose not to return, when what was observed is that they
      stopped listening at all.

    ``require_user_runway`` controls the second. Both counts are reported, so
    the choice is visible rather than buried.
    """
    span = days * DAY
    ends = table.encounter_ts.astype(np.int64) + span

    corpus_end = int(np.asarray(corpus.ts).max())
    observable = ends <= corpus_end

    if require_user_runway:
        last_idx = np.asarray(corpus.user_offsets)[table.user_code + 1] - 1
        user_end = np.asarray(corpus.ts)[last_idx]
        observable = observable & (ends <= user_end)

    label = table.recurs & ((table.recur_ts.astype(np.int64) - table.encounter_ts) <= span)

    ends_ts = np.where(observable, ends, -1).astype(np.int32)
    return Horizon(f"time_{days}d", label, observable, ends_ts)


@dataclass
class EncounterSplit:
    """Which encounters train, which test, and which users were held out."""

    is_train: np.ndarray  # bool
    is_test: np.ndarray  # bool
    is_holdout_user: np.ndarray  # bool, the cold-user slice
    cutoff_ts: int

    def summary(self) -> dict:
        return {
            "cutoff_ts": self.cutoff_ts,
            "train": int(self.is_train.sum()),
            "test": int(self.is_test.sum()),
            "cold_user_test": int((self.is_test & self.is_holdout_user).sum()),
        }


def temporal_split(
    table: EncounterTable,
    n_users: int,
    test_frac: float = 0.15,
    holdout_user_frac: float = 0.10,
    seed: int = 0,
) -> EncounterSplit:
    """Global temporal cut plus a whole-user holdout.

    Two orthogonal axes, as in ``melochron/data/splits.py``: time asks whether
    the model predicts the future, users ask whether it works for someone it has
    never seen. Held-out users are removed from training across all time, not
    merely after the cutoff.

    Reimplemented here rather than reused because that module takes a pandas
    frame and this table has 50M rows.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    cutoff = int(np.quantile(table.encounter_ts, 1.0 - test_frac))

    rng = np.random.default_rng(seed)
    n_holdout = round(n_users * holdout_user_frac)
    if n_holdout >= n_users:
        raise ValueError(
            f"holdout_user_frac={holdout_user_frac} would hold out {n_holdout} of "
            f"{n_users} users, leaving nothing to train on"
        )
    holdout = np.zeros(n_users, dtype=bool)
    if n_holdout:
        holdout[rng.choice(n_users, size=n_holdout, replace=False)] = True

    is_holdout_user = holdout[table.user_code]
    is_test = table.encounter_ts >= cutoff
    is_train = ~is_test & ~is_holdout_user

    return EncounterSplit(is_train, is_test, is_holdout_user, cutoff)


def train_horizon_fits(split: EncounterSplit, horizon: Horizon) -> np.ndarray:
    """Training rows whose horizon closes before the split boundary.

    Without this a training encounter near the cutoff is labelled from events in
    the test period. It is not leakage into the *metric*, but it is a training
    signal that no deployed system could have had, and the fix costs only a thin
    band of rows at the boundary.
    """
    return split.is_train & horizon.observable & (horizon.ends_ts < split.cutoff_ts)


def usable(
    table: EncounterTable, horizon: Horizon, split: EncounterSplit | None = None
) -> np.ndarray:
    """Rows that carry a trustworthy label: observable, and not corrupt.

    The corpus holds exactly one event dated before Last.fm existed. It is a
    single row in 253M, but it would hand its user a fifty-year history, so any
    encounter carrying it is excluded rather than explained later.
    """
    keep = horizon.observable & (table.encounter_ts >= PLAUSIBLE_FLOOR)
    if split is not None:
        keep = keep & (split.is_train | split.is_test)
    return keep
