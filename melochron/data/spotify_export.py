"""Parser for Spotify personal data exports.

Handles both exports the plan asks for, because they arrive on very different
schedules and have different schemas:

* **Extended streaming history** (``Streaming_History_Audio_*.json``). The one
  that matters: full history, with ``ms_played``, ``skipped``, ``shuffle`` and
  ``offline``. Takes up to 30 days to arrive.
* **Account data** (``StreamingHistory*.json``). Arrives in a few days and
  covers roughly the last year. Carries only ``endTime``, ``artistName``,
  ``trackName``, ``msPlayed``, so the skip/shuffle/offline columns come through
  null. Useful as an early stand-in, not as the final corpus.

Two details worth knowing before reading the output:

* **Podcast rows.** Episodes appear with a null track name and an ``episode_name``
  instead. They are dropped. On a podcast-heavy account this can be a large
  fraction of rows, which is why :func:`melochron.schema.conform` only complains
  above 50%.
* **``endTime`` is an end timestamp, not a start.** The account-data export
  stamps when a play *finished*, at minute resolution. The canonical schema is
  start-ordered, so ``ms_played`` is subtracted back off to recover an
  approximate start. Without that, ordering within a minute is wrong and long
  plays sort after short ones that actually followed them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from melochron import schema

SOURCE_EXTENDED = "spotify-extended"
SOURCE_ACCOUNT = "spotify-account"

_EXTENDED_GLOBS = ("Streaming_History_Audio*.json", "endsong*.json")
_ACCOUNT_GLOBS = ("StreamingHistory*.json",)


def _load_json_files(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, list):
            # ValueError, not TypeError, despite the isinstance shape: this is a
            # malformed *file*, not a bad argument from a caller. noqa rather
            # than changing the exception to satisfy the lint.
            raise ValueError(  # noqa: TRY004
                f"{p} does not contain a JSON array of play records"
            )
        records.extend(payload)
    return records


def _find(root: Path, globs: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pattern in globs:
        found.extend(sorted(root.rglob(pattern)))
    return found


def _parse_extended(records: list[dict], user_id: str) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df = df.rename(
        columns={
            "master_metadata_track_name": schema.TRACK,
            "master_metadata_album_artist_name": schema.ARTIST,
            "ms_played": schema.MS_PLAYED,
            "shuffle": schema.SHUFFLE,
            "offline": schema.OFFLINE,
            "skipped": schema.SKIPPED,
        }
    )

    df[schema.TS] = schema.to_unix_seconds(df["ts"], format="ISO8601")
    df[schema.USER] = user_id

    for col in (schema.MS_PLAYED, schema.SKIPPED, schema.SHUFFLE, schema.OFFLINE):
        if col not in df.columns:
            df[col] = pd.NA

    return df


def _parse_account(records: list[dict], user_id: str) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df = df.rename(
        columns={
            "trackName": schema.TRACK,
            "artistName": schema.ARTIST,
            "msPlayed": schema.MS_PLAYED,
        }
    )

    end_s = schema.to_unix_seconds(df["endTime"], format="mixed")
    ms = pd.to_numeric(df[schema.MS_PLAYED], errors="coerce").fillna(0)
    # Recover an approximate start time; see module docstring.
    df[schema.TS] = end_s - (ms // 1000)
    df[schema.USER] = user_id

    return df


def read_export(path: str | Path, user_id: str = "me") -> pd.DataFrame:
    """Read a Spotify export directory (or single JSON file) into the schema.

    Prefers the extended history when both are present in the same directory,
    since it strictly dominates the account export in coverage and fields.
    """
    root = Path(path)
    if root.is_file():
        files, parser, source = [root], _parse_extended, SOURCE_EXTENDED
        if root.name.startswith("StreamingHistory"):
            parser, source = _parse_account, SOURCE_ACCOUNT
    else:
        extended = _find(root, _EXTENDED_GLOBS)
        account = _find(root, _ACCOUNT_GLOBS)
        if extended:
            files, parser, source = extended, _parse_extended, SOURCE_EXTENDED
        elif account:
            files, parser, source = account, _parse_account, SOURCE_ACCOUNT
        else:
            raise FileNotFoundError(
                f"no Spotify history JSON found under {root}. Expected files matching "
                f"{_EXTENDED_GLOBS} (extended history) or {_ACCOUNT_GLOBS} (account data)."
            )

    records = _load_json_files(files)
    if not records:
        raise ValueError(f"{root} contained {len(files)} history file(s) but no records")

    df = parser(records, user_id=user_id)

    # Podcast and video rows carry a null track name. Drop them before conform()
    # so its 50% guard measures parse failures rather than legitimate podcasts.
    if schema.TRACK in df.columns:
        df = df[df[schema.TRACK].notna()]

    return schema.conform(df, source=source)


def summarize(df: pd.DataFrame) -> dict[str, object]:
    """Quick shape check on a freshly parsed export."""
    ts = pd.to_datetime(df[schema.TS], unit="s", utc=True)
    return {
        "events": len(df),
        "source": df[schema.SOURCE].iloc[0] if len(df) else None,
        "first_play": str(ts.min()),
        "last_play": str(ts.max()),
        "distinct_tracks": df.groupby([schema.ARTIST, schema.TRACK], observed=True).ngroups,
        "distinct_artists": df[schema.ARTIST].nunique(),
        "has_ms_played": bool(df[schema.MS_PLAYED].notna().any()),
    }
