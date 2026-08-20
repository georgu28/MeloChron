"""Slice keys: the cuts every result table is broken down by.

A single overall number hides the only question that matters. The brief's
pass/fail line is a *slice* -- if the model cannot beat a dumb baseline on
unfamiliar material, it has learned loyalty rather than discovery.

The brief defines that slice by artist, which this dataset does not carry. Two
substitutes stand in, deliberately measuring different kinds of unfamiliar:

* ``new_neighborhood`` (added in Phase 2, where tag vectors are built) --
  unlike what *this user* plays.
* ``popularity_decile`` -- unlike what *most people* play.

A model that only wins on the head will be caught by both.
"""

from __future__ import annotations

import numpy as np

from melochron.adoption.corpus import CompactCorpus
from melochron.adoption.labels import EncounterSplit, EncounterTable

#: Boundaries for the "how far into this user's listening life" slice. Early
#: encounters are a different situation from late ones: a user's first hundred
#: tracks are all novel by construction.
ORDINAL_EDGES = (10, 100, 1000)


def track_play_counts(corpus: CompactCorpus) -> np.ndarray:
    """Global plays per track, over the whole corpus."""
    return np.bincount(np.asarray(corpus.track_code), minlength=corpus.n_tracks)


def popularity_decile(corpus: CompactCorpus, table: EncounterTable) -> np.ndarray:
    """Decile of the encountered track by global play count, 0 = least played.

    Deciles are over *tracks*, not over plays, so each band holds roughly the
    same number of distinct tracks. Ranking by play mass instead would put half
    the catalogue in one bucket and defeat the purpose.

    This is the substitute for a long-tail cutoff. Phase 0 found the brief's
    >=20-plays filter removes nothing -- the least-played track already has 52 --
    so obscurity has to be measured relatively rather than thresholded.
    """
    plays = track_play_counts(corpus)
    # Rank with ties broken by track id so the assignment is deterministic.
    rank = np.empty(corpus.n_tracks, dtype=np.int64)
    rank[np.argsort(plays, kind="stable")] = np.arange(corpus.n_tracks)
    decile = (rank * 10 // corpus.n_tracks).astype(np.int8)
    return decile[table.track_code]


def encounter_year(table: EncounterTable) -> np.ndarray:
    """Calendar year of the encounter, for the drift table.

    The corpus spans fifteen years. Whether adoption behaviour is stable across
    them is a question this project should answer rather than assume, because a
    global temporal split trains on an era that the test period may not resemble.
    """
    as_dates = table.encounter_ts.astype("datetime64[s]").astype("datetime64[Y]")
    return as_dates.astype(int) + 1970


def encounter_ordinal(table: EncounterTable) -> np.ndarray:
    """How many distinct tracks this user had already met, before this one.

    Not the same as ``encounter_pos``, which counts *plays*. A user 500 plays in
    may have met only 40 tracks.
    """
    order = np.lexsort((table.encounter_pos, table.user_code))
    users_sorted = table.user_code[order]

    # Position within each user's run of encounters.
    is_new_user = np.empty(users_sorted.shape[0], dtype=bool)
    is_new_user[0] = True
    np.not_equal(users_sorted[1:], users_sorted[:-1], out=is_new_user[1:])
    run_start = np.maximum.accumulate(np.where(is_new_user, np.arange(users_sorted.shape[0]), 0))

    ordinal = np.empty(users_sorted.shape[0], dtype=np.int32)
    ordinal[order] = (np.arange(users_sorted.shape[0]) - run_start).astype(np.int32)
    return ordinal


def ordinal_band(ordinal: np.ndarray, edges: tuple[int, ...] = ORDINAL_EDGES) -> np.ndarray:
    """Bucket the encounter ordinal into coarse bands."""
    return np.searchsorted(np.asarray(edges), ordinal, side="right").astype(np.int8)


def build(
    corpus: CompactCorpus, table: EncounterTable, split: EncounterSplit
) -> dict[str, np.ndarray]:
    """Every slice key derivable without files beyond the listening events."""
    ordinal = encounter_ordinal(table)
    return {
        "popularity_decile": popularity_decile(corpus, table),
        "year": encounter_year(table),
        "encounter_ordinal": ordinal,
        "ordinal_band": ordinal_band(ordinal),
        "cold_user": split.is_holdout_user,
    }


def named_slices(keys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The named boolean slices the headline table reports.

    ``head`` and ``tail`` are the top and bottom popularity deciles: the
    clearest read on whether a model works anywhere but the obvious places.
    """
    decile = keys["popularity_decile"]
    return {
        "tail (decile 0)": decile == 0,
        "tail (deciles 0-2)": decile <= 2,
        "head (decile 9)": decile == 9,
        "cold_user": keys["cold_user"],
        "first 10 encounters": keys["ordinal_band"] == 0,
    }
