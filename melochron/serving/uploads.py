"""Turning an uploaded export into a scoreable listening history.

Spotify hands people a ZIP archive, so that is what the service has to accept;
asking a user to unpack it and find the right JSON files inside would push a
data-engineering chore onto them. A raw ``.json`` is accepted too, since that is
what falls out of the archive and what a returning user is likely to keep.

Everything here treats the upload as hostile input. It arrives over the network
from an unauthenticated caller, and the two classic archive attacks both apply:

* **Path traversal.** A member named ``../../.ssh/authorized_keys`` writes
  outside the extraction directory on a naive ``extractall``. Every member is
  resolved and checked to be inside the destination before anything is written.
* **Decompression bombs.** A few kilobytes of ZIP can expand to gigabytes.
  Extraction is bounded by member count and by total uncompressed bytes, and
  the declared size is checked *before* the write rather than measured after.

The retained-history cap is a memory decision, not a modelling one. Scoring
only ever reads the most recent ``max_len`` (200) plays, but the Phase 6 drift
and archetype surfaces want a longer tail, so the store keeps far more than
scoring needs and still refuses to hold an unbounded amount. At roughly 100
bytes per retained play, 20k plays is ~2 MB per user, which stays comfortable at
the stated concurrency of five.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from melochron import schema
from melochron.data import spotify_export
from melochron.data.sessions import DEFAULT_MIN_MS, filter_positives
from melochron.data.vocab import canonical_key

log = logging.getLogger(__name__)

#: Refuse archives that expand beyond this. An extended history export is
#: typically tens of MB of JSON; 512 MB is generous for a real user and still
#: bounded well below what would exhaust a small dyno.
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 2_000
#: Plays retained per upload; see module docstring for the memory arithmetic.
MAX_RETAINED_EVENTS = 20_000

_JSON_SUFFIX = ".json"


class UploadError(ValueError):
    """Upload was malformed or unusable. Carries a message meant for the user."""


@dataclass
class ParsedHistory:
    """A parsed upload, ready to score."""

    #: ``(artist, track, unix_seconds)``, ascending. Possibly truncated to the
    #: most recent ``MAX_RETAINED_EVENTS``.
    history: list[tuple[str, str, int]]
    stats: dict = field(default_factory=dict)
    #: Canonical keys of everything in ``history``. Precomputed here, on the
    #: worker thread, because the repeat/novel flag on each recommendation
    #: needs it and canonicalisation is regex-heavy enough to be worth keeping
    #: off the request path.
    keys: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.history)


def history_key_set(history: list[tuple[str, str, int]]) -> set[str]:
    """Canonical keys for a history, canonicalising distinct pairs only.

    Mirrors the optimisation in :func:`melochron.data.vocab.add_item_keys`:
    normalisation runs several regex passes per call, and a heavy listener's
    20k plays collapse to a few thousand distinct artist/track pairs.
    """
    return {canonical_key(a, t) for a, t in {(a, t) for a, t, _ in history}}


def _is_within(root: Path, target: Path) -> bool:
    """True when ``target`` resolves inside ``root``."""
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def safe_extract_zip(archive: Path, dest: Path) -> list[Path]:
    """Extract the JSON members of ``archive`` into ``dest``, bounded and checked.

    Returns the paths written. Non-JSON members are skipped rather than
    rejected: real Spotify archives carry PDFs and images alongside the history,
    and failing the whole upload over a README would be user-hostile.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = 0

    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            if len(members) > MAX_MEMBERS:
                raise UploadError(
                    f"archive contains {len(members)} entries, more than the "
                    f"{MAX_MEMBERS} this service will extract"
                )

            for info in members:
                if info.is_dir() or not info.filename.lower().endswith(_JSON_SUFFIX):
                    continue

                # Flattened deliberately: the archive's internal directory
                # layout carries no information the parser uses, and flattening
                # removes traversal as a category rather than filtering it.
                name = Path(info.filename).name
                if not name or name.startswith("."):
                    continue

                target = dest / name
                if not _is_within(dest, target):
                    raise UploadError(
                        f"archive entry {info.filename!r} escapes the upload directory"
                    )

                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise UploadError(
                        "archive expands beyond the "
                        f"{MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB extraction limit"
                    )

                with zf.open(info) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                written.append(target)
    except zipfile.BadZipFile as exc:
        raise UploadError("file is not a readable ZIP archive") from exc

    if not written:
        raise UploadError(
            "archive contained no JSON history files. Upload the ZIP Spotify sent you, "
            "or the Streaming_History_Audio_*.json files from inside it."
        )
    return written


def to_history(df: pd.DataFrame, max_events: int = MAX_RETAINED_EVENTS) -> ParsedHistory:
    """Reduce a canonical event frame to the triples the recommender takes.

    Positives are filtered through the same :func:`filter_positives` the
    training pipeline uses, so the service's notion of "a play" cannot drift
    from the one the model was trained against. That drift would be invisible
    and would show up only as unexplained serving/eval disagreement.
    """
    total_events = len(df)
    played = filter_positives(df)

    ordered = played.sort_values(schema.TS, kind="stable")
    truncated = len(ordered) > max_events
    if truncated:
        ordered = ordered.iloc[-max_events:]

    history = [
        (str(a), str(t), int(ts))
        for a, t, ts in zip(ordered[schema.ARTIST], ordered[schema.TRACK], ordered[schema.TS])
    ]

    stats: dict = {
        "events_in_file": int(total_events),
        "plays_after_filter": len(played),
        "retained": len(history),
        "truncated": truncated,
        "min_ms_threshold": DEFAULT_MIN_MS,
    }
    if history:
        ts = pd.to_datetime([h[2] for h in history], unit="s", utc=True)
        stats["first_play"] = str(ts.min())
        stats["last_play"] = str(ts.max())
        stats["distinct_tracks"] = int(
            ordered.groupby([schema.ARTIST, schema.TRACK], observed=True).ngroups
        )
        stats["distinct_artists"] = int(ordered[schema.ARTIST].nunique())
        stats["source"] = str(ordered[schema.SOURCE].iloc[0])

    return ParsedHistory(history=history, stats=stats, keys=history_key_set(history))


def parse_upload(path: Path, workdir: Path, user_id: str = "upload") -> ParsedHistory:
    """Parse a saved upload (ZIP or JSON) into a scoreable history.

    Runs on a worker thread, never on the event loop: this is seconds of
    pandas work on a large export and would stall every other request.
    """
    if zipfile.is_zipfile(path):
        extract_dir = workdir / "extracted"
        safe_extract_zip(path, extract_dir)
        source: Path = extract_dir
    elif path.suffix.lower() == _JSON_SUFFIX:
        source = path
    else:
        raise UploadError(
            "unrecognised file type. Upload the ZIP from Spotify, or a "
            "Streaming_History_Audio_*.json file from inside it."
        )

    try:
        df = spotify_export.read_export(source, user_id=user_id)
    except FileNotFoundError as exc:
        raise UploadError(str(exc)) from exc
    except ValueError as exc:
        raise UploadError(f"could not parse the export: {exc}") from exc

    parsed = to_history(df)
    if not parsed.history:
        raise UploadError(
            "no plays survived filtering. Every row was shorter than "
            f"{DEFAULT_MIN_MS // 1000}s, which usually means the export is skips only."
        )
    return parsed
