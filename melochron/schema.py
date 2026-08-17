"""Canonical listening-event schema.

Every data source (Spotify export, lastfm-1K, synthetic) parses into this one
frame so that downstream code never branches on provenance. Source-specific
fields are nullable: lastfm-1K has no ``ms_played``, Spotify has no MusicBrainz
ids, and both parse through the same path.
"""

from __future__ import annotations

import pandas as pd

USER = "user_id"
TS = "ts"
ARTIST = "artist"
TRACK = "track"
MS_PLAYED = "ms_played"
SKIPPED = "skipped"
SHUFFLE = "shuffle"
OFFLINE = "offline"
SOURCE = "source"

#: Present in every source. Missing values here are a parse bug, not a gap.
REQUIRED = [USER, TS, ARTIST, TRACK]

#: Source-specific. Always present as columns, but may be entirely null.
OPTIONAL = [MS_PLAYED, SKIPPED, SHUFFLE, OFFLINE]

COLUMNS = REQUIRED + OPTIONAL + [SOURCE]

DTYPES = {
    USER: "string",
    TS: "int64",  # unix seconds, UTC
    ARTIST: "string",
    TRACK: "string",
    MS_PLAYED: "Int64",  # nullable
    SKIPPED: "boolean",  # nullable
    SHUFFLE: "boolean",  # nullable
    OFFLINE: "boolean",  # nullable
    SOURCE: "string",
}


def to_unix_seconds(values, **to_datetime_kwargs) -> pd.Series:
    """Parse timestamps to nullable int64 unix seconds, UTC.

    Every parser routes through this rather than rolling its own conversion,
    because the obvious idiom is silently wrong on current pandas. Writing
    ``parsed.astype("int64") // 1_000_000_000`` assumes nanosecond resolution,
    but pandas 3.x returns microsecond-resolution datetimes for many inputs, so
    that expression divides by 1000x too much and yields timestamps in 1970.
    Nothing downstream errors: the split, the ordering and the time deltas are
    all just wrong.

    Subtracting the epoch and floor-dividing by a one-second Timedelta is
    resolution-independent, so it stays correct whatever unit pandas picks.
    Unparseable values become NA rather than raising; ``conform`` drops them.
    """
    parsed = pd.to_datetime(values, utc=True, errors="coerce", **to_datetime_kwargs)
    delta = parsed - pd.Timestamp("1970-01-01", tz="UTC")
    return (delta // pd.Timedelta(seconds=1)).astype("Int64")


def empty() -> pd.DataFrame:
    """An empty frame with the canonical columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=DTYPES[c]) for c in COLUMNS})


def conform(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Coerce a parsed frame to the canonical schema.

    Adds any missing optional columns as all-null, casts dtypes, stamps the
    source, drops rows missing a required field, and sorts by (user, ts) which
    is the order every sequence builder downstream assumes.
    """
    df = df.copy()

    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = pd.NA
    df[SOURCE] = source

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: parsed frame is missing required columns {missing}")

    df = df[COLUMNS].astype(DTYPES)

    before = len(df)
    df = df.dropna(subset=REQUIRED)
    dropped = before - len(df)
    if dropped:
        # Real exports contain podcast rows and null-track entries. Expected,
        # but a large fraction means the parser is looking at the wrong field.
        frac = dropped / before if before else 0.0
        if frac > 0.5:
            raise ValueError(
                f"{source}: dropped {dropped}/{before} rows ({frac:.0%}) for missing "
                f"required fields. That is too many to be podcasts; check the parser."
            )

    return df.sort_values([USER, TS], kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Raise if ``df`` violates an invariant the pipeline relies on."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing canonical columns {missing}")

    if df[REQUIRED].isna().any().any():
        raise ValueError("required columns contain nulls")

    if not df[TS].is_monotonic_increasing:
        # Only guaranteed within a user, so check per group.
        by_user = df.groupby(USER, observed=True)[TS]
        if not by_user.apply(lambda s: s.is_monotonic_increasing).all():
            raise ValueError("events are not sorted by ts within each user")

    if (df[TS] <= 0).any():
        raise ValueError("ts must be a positive unix timestamp in seconds")
