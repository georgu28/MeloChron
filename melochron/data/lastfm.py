"""Parser for the lastfm-dataset-1K corpus (Celma, 2010).

992 users, roughly 19M timestamped listening events. This is the pretraining
corpus: its schema is almost exactly the Spotify export's, which is what makes
pretrain-then-fine-tune a clean transfer rather than a schema translation.

The raw file is a headerless TSV:

    userid \\t timestamp \\t mbid-artist \\t artist-name \\t mbid-track \\t track-name

Three things about it break a naive ``read_csv`` and all three are handled here:

* **Bare double quotes inside names.** Track and artist titles contain ``"`` as
  a literal character. With pandas' default quoting, one such row swallows
  everything up to the next quote and silently merges thousands of records.
  ``quoting=csv.QUOTE_NONE`` is mandatory, not defensive styling.
* **Empty MBID fields**, which are common and are not an error.
* **Duplicate timestamps within a user**, from scrobble backfills. These are
  kept: a stable sort preserves file order, and dropping them would delete
  genuine consecutive plays.

At 19M rows the file is read in chunks so a full parse does not need the whole
frame resident at once.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from melochron import schema

SOURCE = "lastfm-1k"

COLUMNS = [
    "user_id",
    "timestamp",
    "mbid_artist",
    "artist",
    "mbid_track",
    "track",
]

#: lastfm-1K stamps are ISO 8601 with a literal trailing Z. Giving pandas the
#: exact format avoids per-row inference, which dominates runtime at this size.
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

DEFAULT_FILENAME = "userid-timestamp-artid-artname-traid-traname.tsv"


def _parse_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns={"timestamp": schema.TS})
    chunk[schema.TS] = schema.to_unix_seconds(chunk[schema.TS], format=TS_FORMAT)
    chunk = chunk.dropna(subset=[schema.TS, "artist", "track"])
    return chunk[["user_id", schema.TS, "artist", "track"]].rename(
        columns={"user_id": schema.USER, "artist": schema.ARTIST, "track": schema.TRACK}
    )


def read_lastfm1k(
    path: str | Path,
    limit: int | None = None,
    chunksize: int = 2_000_000,
    users: int | None = None,
) -> pd.DataFrame:
    """Read the lastfm-1K TSV into the canonical event schema.

    ``limit`` caps total rows read, for fast iteration during development.
    ``users`` instead caps the number of distinct users, which is usually the
    better knob: a row cap truncates mid-history and produces users whose
    sequences stop arbitrarily, while a user cap yields complete histories for
    a subset. ``ms_played`` is left null throughout; see
    :func:`melochron.data.sessions.filter_positives` for why that is correct
    rather than a gap.
    """
    path = Path(path)
    if path.is_dir():
        path = path / DEFAULT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with scripts/download_lastfm1k.py, or point "
            f"--lastfm at the directory containing {DEFAULT_FILENAME!r}."
        )

    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
        chunksize=chunksize,
        nrows=limit,
    )

    frames: list[pd.DataFrame] = []
    seen_users: set[str] = set()
    for chunk in reader:
        parsed = _parse_chunk(chunk)

        if users is not None:
            # The file is grouped by user, so once the cap is reached and the
            # current chunk introduces no already-tracked user, we are done.
            seen_users.update(parsed[schema.USER].unique().tolist())
            if len(seen_users) > users:
                keep = set(sorted(seen_users)[:users])
                parsed = parsed[parsed[schema.USER].isin(keep)]
                if len(parsed):
                    frames.append(parsed)
                break

        frames.append(parsed)

    if not frames:
        raise ValueError(f"{path} produced no parseable rows")

    return schema.conform(pd.concat(frames, ignore_index=True), source=SOURCE)
