"""The compact columnar corpus that every adoption label is derived from.

253M events do not fit in the pandas frame the next-track work uses, and they
do not need to. This corpus is three parallel arrays -- ``user_code``,
``track_code``, ``ts`` -- sorted by (user, ts), plus the per-user boundaries
into them. About 3 GB, memory-mappable, and every label operation becomes a
slice over contiguous memory.

**Why this does not route through ``melochron/schema.py``.** That schema lists
``artist`` as required and ``conform()`` drops rows missing it, raising past
50%; Music4All-Onion has no artist column at all. It also sorts a pandas frame
of Python strings, which at this row count is not a performance question but a
memory one. The one thing carried over is the lesson of ``to_unix_seconds``:
never divide a datetime by an assumed resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from melochron.adoption.onion import PUBLISHED, OnionSchema, read_batches

#: Files written by ``save``. Names are load-bearing: ``load`` reads them back.
ARRAYS = ("user_code", "track_code", "ts", "user_offsets", "users", "tracks")

_INT32_MAX = np.iinfo(np.int32).max

#: 2002-01-01. Last.fm was founded in 2002, so a listening event dated earlier
#: is a corrupt row, not an early adopter.
PLAUSIBLE_FLOOR = 1_009_843_200


@dataclass
class CompactCorpus:
    """Listening events as parallel arrays, sorted by (user_code, ts)."""

    user_code: np.ndarray  # int32 [n_events]
    track_code: np.ndarray  # int32 [n_events]
    ts: np.ndarray  # int32 [n_events], unix seconds UTC
    user_offsets: np.ndarray  # int64 [n_users + 1], boundaries into the above
    users: np.ndarray  # [n_users] original user ids, indexed by code
    tracks: np.ndarray  # [n_tracks] original track ids, indexed by code

    @property
    def n_events(self) -> int:
        return int(self.user_code.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.users.shape[0])

    @property
    def n_tracks(self) -> int:
        return int(self.tracks.shape[0])

    def events_for(self, user: int) -> slice:
        """The slice of the arrays belonging to ``user``'s events, in time order."""
        return slice(int(self.user_offsets[user]), int(self.user_offsets[user + 1]))

    def validate(self) -> None:
        """Raise if an invariant the label builder relies on is broken."""
        n = self.n_events
        if not (self.track_code.shape[0] == self.ts.shape[0] == n):
            raise ValueError("parallel arrays have different lengths")
        if self.user_offsets.shape[0] != self.n_users + 1:
            raise ValueError("user_offsets must have n_users + 1 entries")
        if int(self.user_offsets[0]) != 0 or int(self.user_offsets[-1]) != n:
            raise ValueError("user_offsets must span exactly [0, n_events]")
        if not np.all(np.diff(self.user_offsets) >= 0):
            raise ValueError("user_offsets is not monotonic")
        if not np.all(np.diff(self.user_code) >= 0):
            raise ValueError("events are not grouped by user")
        if (self.ts <= 0).any():
            raise ValueError("ts must be a positive unix timestamp in seconds")

        starts, ends = self.user_offsets[:-1], self.user_offsets[1:]
        if (starts >= ends).any():
            raise ValueError("a user has no events; codes must be dense")

        # Time order within each user is what every horizon computation assumes.
        # Checking it globally would be wrong: ts resets at each user boundary,
        # so the gap spanning a boundary is legitimately negative and is masked.
        gaps = np.diff(self.ts.astype(np.int64))
        if len(gaps):
            interior = np.ones(len(gaps), dtype=bool)
            interior[ends[ends < n] - 1] = False
            if (gaps[interior] < 0).any():
                raise ValueError("events are not sorted by ts within each user")

    def save(self, out: Path, extra: dict | None = None) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        for name in ARRAYS:
            np.save(out / f"{name}.npy", getattr(self, name))
        manifest = {
            "n_events": self.n_events,
            "n_users": self.n_users,
            "n_tracks": self.n_tracks,
            "ts_min": int(self.ts.min()),
            "ts_max": int(self.ts.max()),
            **(extra or {}),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out / "manifest.json"

    @classmethod
    def load(cls, path: Path, mmap: bool = True) -> CompactCorpus:
        """Load from disk, memory-mapping the big arrays by default."""
        mode = "r" if mmap else None
        arrays = {}
        for name in ARRAYS:
            # The vocabularies are small and are wanted as real arrays; the
            # event columns are 1 GB each and are wanted as maps.
            big = name in ("user_code", "track_code", "ts")
            arrays[name] = np.load(path / f"{name}.npy", mmap_mode=mode if big else None)
        return cls(**arrays)


def _map_to_codes(column: pa.Array, vocab: dict[str, int], values: list[str]) -> np.ndarray:
    """Map a batch of id strings to global int32 codes.

    Dictionary-encoding first means the Python-level vocabulary lookup runs once
    per *distinct value in the batch* (at most a few tens of thousands) rather
    than once per row (a million). The remap itself is a numpy gather.
    """
    encoded = pc.dictionary_encode(column)
    batch_values = encoded.dictionary.to_pylist()

    local_to_global = np.empty(len(batch_values), dtype=np.int32)
    for i, value in enumerate(batch_values):
        code = vocab.get(value)
        if code is None:
            code = len(values)
            vocab[value] = code
            values.append(value)
        local_to_global[i] = code

    if encoded.indices.null_count:
        # A null id means a malformed row that pyarrow accepted. Downstream it
        # would become a real code for the empty string and quietly join
        # unrelated events together, so it stops here.
        raise ValueError("null id encountered; the events file should have none")

    indices = encoded.indices.to_numpy(zero_copy_only=False)
    return local_to_global[indices.astype(np.int64)]


def _to_unix_seconds(column: pa.Array, ts_kind: str) -> np.ndarray:
    """Convert a string timestamp column to int32 unix seconds.

    Never ``astype("int64") // 1_000_000_000``: that hardcodes an assumed
    resolution and silently lands everything in 1970 when the assumption is
    wrong. Each branch here converts through a unit it has actually checked.
    """
    if ts_kind == "seconds":
        seconds = column.cast(pa.int64()).to_numpy(zero_copy_only=False)
    elif ts_kind == "millis":
        seconds = column.cast(pa.int64()).to_numpy(zero_copy_only=False) // 1000
    elif ts_kind == "iso":
        seconds = column.cast(pa.timestamp("s")).cast(pa.int64()).to_numpy(zero_copy_only=False)
    else:
        raise ValueError(f"unknown ts_kind {ts_kind!r}")

    if seconds.min() <= 0:
        raise ValueError("non-positive timestamp in the events file")
    if seconds.max() > _INT32_MAX:
        raise ValueError(
            f"timestamp {seconds.max()} overflows int32; the compact store assumes "
            f"unix seconds before 2038"
        )
    return seconds.astype(np.int32)


#: Bytes per row, used only to size the initial allocation. Measured on the real
#: archive: 2,211,449,511 bytes over 252,984,396 rows is 8.74 compressed bytes
#: per row, and an uncompressed row of two 16-char ids and a timestamp is ~40.
#: Deliberate underestimates, so the first guess overshoots and the array does
#: not have to grow.
_BYTES_PER_ROW = {".bz2": 8.0, "": 36.0}


def estimate_rows(path: Path) -> int:
    """A first guess at the row count, from file size alone.

    Defaulting the allocation to the published 253M would make ``build`` claim
    3 GB to parse a ten-row test fixture. Sizing from the file keeps small
    inputs small, and the growth path below keeps a wrong guess correct.
    """
    per_row = _BYTES_PER_ROW.get(path.suffix, _BYTES_PER_ROW[""])
    return max(1 << 16, int(path.stat().st_size / per_row))


def build(
    path: Path,
    schema: OnionSchema,
    capacity: int | None = None,
    progress_every: int = 25_000_000,
) -> CompactCorpus:
    """Parse the events file into a sorted compact corpus in one decompression pass."""
    capacity = capacity or estimate_rows(path)
    user_code = np.empty(capacity, dtype=np.int32)
    track_code = np.empty(capacity, dtype=np.int32)
    ts = np.empty(capacity, dtype=np.int32)

    user_vocab: dict[str, int] = {}
    track_vocab: dict[str, int] = {}
    user_values: list[str] = []
    track_values: list[str] = []

    filled = 0
    next_report = progress_every
    for batch in read_batches(path, schema):
        rows = batch.num_rows
        if filled + rows > capacity:
            # The size estimate is a guess. Grow rather than truncate: silently
            # dropping the tail of a corpus is the kind of bug that surfaces as
            # a slightly-wrong metric rather than an error.
            capacity = max(int(capacity * 1.5), filled + rows)
            user_code = np.resize(user_code, capacity)
            track_code = np.resize(track_code, capacity)
            ts = np.resize(ts, capacity)

        end = filled + rows
        user_code[filled:end] = _map_to_codes(
            batch.column(schema.user_idx), user_vocab, user_values
        )
        track_code[filled:end] = _map_to_codes(
            batch.column(schema.track_idx), track_vocab, track_values
        )
        ts[filled:end] = _to_unix_seconds(batch.column(schema.ts_idx), schema.ts_kind)
        filled = end

        if filled >= next_report:
            print(f"    {filled:,} events", flush=True)
            next_report += progress_every

    user_code = user_code[:filled]
    track_code = track_code[:filled]
    ts = ts[:filled]

    # Sort by (user, ts) with one composite key rather than np.lexsort, which
    # at this size costs an extra pass and an extra index array for no gain.
    # Both components are non-negative and below 2**32, so the packing is exact.
    order = np.argsort(
        (user_code.astype(np.int64) << 32) | ts.astype(np.int64),
        kind="stable",
    )
    user_code = user_code[order]
    track_code = track_code[order]
    ts = ts[order]
    del order

    n_users = len(user_values)
    offsets = np.searchsorted(user_code, np.arange(n_users + 1, dtype=np.int32)).astype(np.int64)

    corpus = CompactCorpus(
        user_code=user_code,
        track_code=track_code,
        ts=ts,
        user_offsets=offsets,
        users=np.array(user_values),
        tracks=np.array(track_values),
    )
    corpus.validate()
    return corpus


def _percentiles(values: np.ndarray, points=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict[str, float]:
    pct = np.percentile(values, points)
    return {f"p{p}": float(v) for p, v in zip(points, pct)}


def corpus_stats(corpus: CompactCorpus, min_track_plays: int = 20) -> dict:
    """Everything the Phase 0 report needs, measured rather than assumed.

    The expensive part is the distinct-pair pass, which is also the one that
    matters most: plays-per-pair gives the *unbounded* recurrence rate, and that
    is the ceiling on the adoption base rate before any horizon is applied.
    """
    n = corpus.n_events
    counts_per_user = np.diff(corpus.user_offsets)
    starts, ends = corpus.user_offsets[:-1], corpus.user_offsets[1:]
    span_days = (corpus.ts[ends - 1].astype(np.int64) - corpus.ts[starts]) / 86400.0

    pair_key = corpus.user_code.astype(np.int64) * corpus.n_tracks + corpus.track_code
    _, plays_per_pair = np.unique(pair_key, return_counts=True)
    del pair_key

    n_pairs = int(plays_per_pair.shape[0])
    recurring = int((plays_per_pair >= 2).sum())

    track_plays = np.bincount(corpus.track_code, minlength=corpus.n_tracks)
    keep = track_plays >= min_track_plays
    kept_events = int(track_plays[keep].sum())

    ts_min, ts_max = int(corpus.ts.min()), int(corpus.ts.max())

    # Last.fm was founded in 2002, so anything earlier is corrupt rather than
    # early. Reported rather than silently dropped: the count is what says
    # whether this is a footnote or a data-quality problem.
    implausible = corpus.ts < PLAUSIBLE_FLOOR
    n_implausible = int(implausible.sum())
    ts_min_plausible = int(corpus.ts[~implausible].min()) if n_implausible else ts_min

    return {
        "events": {"measured": n, "published": PUBLISHED["events"]},
        "users": {"measured": corpus.n_users, "published": PUBLISHED["users"]},
        "tracks": {"measured": corpus.n_tracks, "published": PUBLISHED["tracks"]},
        "pairs": {"measured": n_pairs, "published": PUBLISHED["pairs"]},
        "span": {
            "ts_min": ts_min,
            "ts_max": ts_max,
            "start": datetime.fromtimestamp(ts_min, UTC).isoformat(),
            "end": datetime.fromtimestamp(ts_max, UTC).isoformat(),
            "days": round((ts_max - ts_min) / 86400.0, 1),
        },
        "timestamp_sanity": {
            "implausible_events": n_implausible,
            "implausible_before": datetime.fromtimestamp(PLAUSIBLE_FLOOR, UTC).date().isoformat(),
            "start_excluding_implausible": datetime.fromtimestamp(
                ts_min_plausible, UTC
            ).isoformat(),
            "span_days_excluding_implausible": round((ts_max - ts_min_plausible) / 86400.0, 1),
        },
        "events_per_user": {
            "mean": round(float(counts_per_user.mean()), 1),
            **_percentiles(counts_per_user),
            "with_at_least_200": int((counts_per_user >= 200).sum()),
            "with_at_least_400": int((counts_per_user >= 400).sum()),
            # Users too short for the event horizon are numerous but tiny. The
            # share of *events* they hold is what says whether dropping them
            # costs anything.
            "at_most_200": int((counts_per_user <= 200).sum()),
            "events_held_at_most_200": int(counts_per_user[counts_per_user <= 200].sum()),
        },
        "active_span_days_per_user": {
            "mean": round(float(span_days.mean()), 1),
            **_percentiles(span_days),
            "at_least_30d": int((span_days >= 30).sum()),
            "at_least_60d": int((span_days >= 60).sum()),
        },
        "plays_per_pair": {
            "mean": round(n / n_pairs, 3),
            **_percentiles(plays_per_pair),
            "max": int(plays_per_pair.max()),
            "recurring_pairs": recurring,
            # The headline number: the share of first encounters that ever recur
            # at all. Every horizoned base rate must come in at or below this.
            "unbounded_recurrence_rate": round(recurring / n_pairs, 4),
        },
        "catalog_cutoff": {
            "min_track_plays": min_track_plays,
            "min_track_plays_observed": int(track_plays.min()),
            "tracks_kept": int(keep.sum()),
            "tracks_dropped": int((~keep).sum()),
            "events_kept": kept_events,
            "events_kept_frac": round(kept_events / n, 4),
        },
    }
