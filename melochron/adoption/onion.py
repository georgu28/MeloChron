"""Reading the Music4All-Onion listening-events file.

The dataset README documents row counts and nothing else -- no column names, no
header convention, no timestamp unit. So the schema is *sniffed from the file*
and reported, rather than assumed from the paper. ``sniff_schema`` lives here
rather than in either script so that ``inspect_onion.py`` (which prints the
schema) and ``build_onion.py`` (which parses with it) cannot drift apart.

Reading strategy: the events file is 2.2 GB of bzip2 holding ~253M rows, which
decompresses to roughly 9 GB. libbzip2 is single-threaded and is the bottleneck,
so the file is decompressed exactly once and parsed in the same pass by
pyarrow's CSV reader. A Python-level loop over 253M lines is not viable here.
"""

from __future__ import annotations

import bz2
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv as pacsv

EVENTS_FILE = "userid_trackid_timestamp.tsv.bz2"
COUNTS_FILE = "userid_trackid_count.tsv.bz2"

#: Published in the dataset README. The build verifies against these and treats
#: a mismatch as a stop condition rather than a warning: if the row count is
#: wrong the parser is wrong, and every downstream label inherits the error.
PUBLISHED = {
    "events": 252_984_396,
    "users": 119_140,
    "tracks": 56_512,
    "pairs": 50_016_042,  # from userid_trackid_count.tsv.bz2
}

USER = "user_id"
TRACK = "track_id"
TS = "ts"

#: The filename encodes the column order: userid_trackid_timestamp. Used only
#: when the file carries no header row to name the columns for us.
POSITIONAL = [USER, TRACK, TS]

_SECONDS_RANGE = (1_000_000_000, 2_000_000_000)  # 2001-09-09 .. 2033-05-18
_MILLIS_RANGE = (1_000_000_000_000, 2_000_000_000_000)


@dataclass
class OnionSchema:
    """What the events file actually looks like, as measured."""

    delimiter: str
    has_header: bool
    columns: list[str]
    user_idx: int
    track_idx: int
    ts_idx: int
    #: "seconds", "millis" or "iso" -- decided by magnitude, then sanity-checked
    #: against a plausible calendar date.
    ts_kind: str
    #: How ``ts_idx`` was identified: "header name", "numeric content" or
    #: "position fallback". Reported, because a positional guess deserves less
    #: confidence than a named column and the report should not hide which ran.
    ts_resolved_by: str = "position fallback"
    sample_rows: list[list[str]] = field(default_factory=list)

    @property
    def ts_is_numeric(self) -> bool:
        return self.ts_kind in ("seconds", "millis")

    def summary(self) -> dict:
        return {
            "delimiter": repr(self.delimiter),
            "has_header": self.has_header,
            "columns": self.columns,
            "user_idx": self.user_idx,
            "track_idx": self.track_idx,
            "ts_idx": self.ts_idx,
            "ts_kind": self.ts_kind,
            "ts_resolved_by": self.ts_resolved_by,
        }


def open_stream(path: Path) -> BinaryIO:
    """Open a possibly-bz2 file as a binary stream.

    Prefers pyarrow's native decompressor, which does the work outside the GIL;
    falls back to the stdlib module when this pyarrow build lacks bz2 support.
    """
    if path.suffix != ".bz2":
        return open(path, "rb")
    try:
        return pa.CompressedInputStream(pa.OSFile(str(path), "rb"), "bz2")
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid, KeyError):
        return bz2.open(path, "rb")


def _classify_timestamp(value: str) -> str:
    """Decide the timestamp encoding from one sample value."""
    text = value.strip()
    if text.isdigit():
        n = int(text)
        if _SECONDS_RANGE[0] <= n <= _SECONDS_RANGE[1]:
            return "seconds"
        if _MILLIS_RANGE[0] <= n <= _MILLIS_RANGE[1]:
            return "millis"
        raise ValueError(
            f"timestamp {text!r} is numeric but is neither plausible unix seconds "
            f"nor milliseconds; refusing to guess"
        )
    # Anything non-numeric is treated as a datetime string and handed to
    # pyarrow's parser, which is stricter than a hand-rolled format guess.
    return "iso"


def _looks_like_header(fields: list[str]) -> bool:
    """A header row names columns; a data row carries an id and a timestamp."""
    lowered = [f.strip().lower() for f in fields]
    named = any(key in cell for cell in lowered for key in ("user", "track", "time", "stamp"))
    if not named:
        return False
    # A data row must not be eaten just because an opaque id contains "time".
    # Requiring that nothing in the row parses as a timestamp is the check.
    return not any(cell.isdigit() and len(cell) >= 9 for cell in lowered)


def sniff_schema(path: Path, probe_bytes: int = 1 << 16) -> OnionSchema:
    """Determine the real layout of the events file from its first few KB.

    Only ``probe_bytes`` of decompressed data are read, so this is fast on a
    2.2 GB archive.
    """
    with open_stream(path) as stream:
        head = stream.read(probe_bytes)
    if isinstance(head, pa.Buffer):
        head = head.to_pybytes()

    text = head.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: could not read enough lines to sniff a schema")
    # The final line is very likely cut mid-row by the byte budget.
    lines = lines[:-1]

    first = lines[0]
    delimiter = "\t" if first.count("\t") >= first.count(",") else ","

    rows = [line.split(delimiter) for line in lines]
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: inconsistent column counts in the first rows: {sorted(widths)}")
    width = widths.pop()

    has_header = _looks_like_header(rows[0])
    if has_header:
        columns = [c.strip() for c in rows[0]]
        data_rows = rows[1:]
    else:
        if width != len(POSITIONAL):
            raise ValueError(
                f"{path}: {width} columns and no header row, so they cannot be named. "
                f"The filename implies {POSITIONAL}."
            )
        columns = list(POSITIONAL)
        data_rows = rows

    if not data_rows:
        raise ValueError(f"{path}: header found but no data rows in the probe")

    # Locate the timestamp by name where the file gives us one, then by
    # content, and only then by position. Position is a reasonable prior; it is
    # not evidence, and on this corpus the timestamps are datetime strings
    # rather than integers, so a content scan alone cannot tell a timestamp
    # column from an opaque id column.
    ts_idx = None
    ts_kind = "iso"
    resolved_by = "position fallback"

    if has_header:
        named = [
            i
            for i, c in enumerate(columns)
            if any(key in c.lower() for key in ("time", "stamp", "date"))
        ]
        if len(named) == 1:
            ts_idx = named[0]
            ts_kind = _classify_timestamp(data_rows[0][ts_idx])
            resolved_by = "header name"

    if ts_idx is None:
        for i in range(width):
            try:
                kind = _classify_timestamp(data_rows[0][i])
            except ValueError:
                continue
            if kind in ("seconds", "millis"):
                ts_idx, ts_kind = i, kind
                resolved_by = "numeric content"
                break

    if ts_idx is None:
        # No name and no numeric column: fall back to the position the filename
        # implies and let pyarrow's datetime parser be the judge of it.
        ts_idx = width - 1
        ts_kind = _classify_timestamp(data_rows[0][ts_idx])

    others = [i for i in range(width) if i != ts_idx]
    if has_header:
        lowered = [c.lower() for c in columns]
        user_idx = next((i for i in others if "user" in lowered[i]), others[0])
        track_idx = next(i for i in others if i != user_idx)
    else:
        user_idx, track_idx = others[0], others[1]

    return OnionSchema(
        delimiter=delimiter,
        has_header=has_header,
        columns=columns,
        user_idx=user_idx,
        track_idx=track_idx,
        ts_idx=ts_idx,
        ts_kind=ts_kind,
        ts_resolved_by=resolved_by,
        sample_rows=data_rows[:5],
    )


def _has_header(path: Path, probe_bytes: int = 4096) -> bool:
    """Does the first line name columns rather than carry data?

    Judged on the last field, which is the play count: a header says "count",
    a data row says a number.
    """
    with open_stream(path) as stream:
        head = stream.read(probe_bytes)
    if isinstance(head, pa.Buffer):
        head = head.to_pybytes()

    first = head.decode("utf-8", errors="replace").splitlines()[0]
    return not first.split("\t")[-1].strip().isdigit()


def read_counts_totals(path: Path, block_size: int = 1 << 26) -> dict:
    """Row count and summed plays from ``userid_trackid_count.tsv.bz2``.

    This file carries no timestamps and is useless for modelling, which is why
    the brief says to skip it. It is read anyway because it is the only
    independent statement of two numbers the label builder must reproduce: the
    distinct (user, track) pair count, which *is* the encounter count, and the
    total plays. An off-by-one in the encounter logic raises nothing and
    produces a plausible-looking metric; this catches it.

    ``sniff_schema`` is deliberately not reused here -- it insists on finding a
    timestamp column, and a play count is not one. The header check is still
    needed: this file carries one (``user_id  track_id  count``), and without
    skipping it pyarrow tries to read the literal string "count" as an int64.
    """
    names = ["user_id", "track_id", "count"]
    read_options = pacsv.ReadOptions(
        column_names=names,
        skip_rows=1 if _has_header(path) else 0,
        block_size=block_size,
    )
    parse_options = pacsv.ParseOptions(delimiter="\t", quote_char=False)
    convert_options = pacsv.ConvertOptions(
        column_types={"user_id": pa.string(), "track_id": pa.string(), "count": pa.int64()}
    )

    rows = 0
    total = 0
    with open_stream(path) as stream:
        reader = pacsv.open_csv(
            stream,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        for batch in reader:
            rows += batch.num_rows
            total += pc.sum(batch.column("count")).as_py() or 0

    return {"pairs": rows, "plays": int(total)}


def read_batches(
    path: Path,
    schema: OnionSchema,
    block_size: int = 1 << 26,
) -> Iterator[pa.RecordBatch]:
    """Stream the events file as pyarrow record batches.

    Every column is read as string and converted by the caller. Letting pyarrow
    infer types across a 253M-row file risks it settling on one type from an
    early block and failing on a later one, and the ids are opaque tokens that
    must not be silently coerced to numbers.
    """
    names = [f"c{i}" for i in range(len(schema.columns))]
    read_options = pacsv.ReadOptions(
        column_names=names,
        skip_rows=1 if schema.has_header else 0,
        block_size=block_size,
    )
    parse_options = pacsv.ParseOptions(delimiter=schema.delimiter, quote_char=False)
    convert_options = pacsv.ConvertOptions(column_types={n: pa.string() for n in names})

    with open_stream(path) as stream:
        reader = pacsv.open_csv(
            stream,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        yield from reader
