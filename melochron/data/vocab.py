"""Item canonicalization and vocabulary construction.

Canonicalization is what lets a 2005-era lastfm-1K scrobble and a 2024 Spotify
play of the same song collapse to one item. It is deliberately conservative:
it strips packaging noise (featuring credits, remaster and edit markers) but
never strips markers that denote a genuinely different recording, so "Live" and
"Acoustic" survive. Over-merging silently corrupts the labels and is much
harder to notice than under-merging.

``canonical_key`` must be idempotent. There is a test for that.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from melochron import schema

PAD_ID = 0
OOV_ID = 1
FIRST_ITEM_ID = 2

# Credits appended to a title: "(feat. X)", "[ft X]", " - featuring X".
_FEAT = re.compile(
    r"""\s*(?:
          [\(\[]\s*(?:feat|ft|featuring|with)\b[^)\]]*[\)\]]
        | -\s*(?:feat|ft|featuring)\b.*$
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Packaging/mastering markers. "live" and "acoustic" are intentionally absent:
# those are different recordings, not different packaging of one recording.
_VERSION_WORDS = (
    r"(?:remaster(?:ed)?|remastered\s*version|re-?master|"
    r"radio\s*edit|single\s*version|album\s*version|original\s*mix|"
    r"mono\s*version|stereo\s*version|deluxe(?:\s*edition)?|"
    r"bonus\s*track|explicit|clean)"
)
_VERSION = re.compile(
    rf"""\s*(?:
          [\(\[]\s*(?:\d{{4}}\s*)?{_VERSION_WORDS}(?:\s*\d{{4}})?\s*[\)\]]
        | -\s*(?:\d{{4}}\s*-?\s*)?{_VERSION_WORDS}(?:\s*\d{{4}})?\s*$
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\-–—_.,;:]+|[\s\-–—_.,;:]+$")

SEP = " :: "


def _strip_noise(text: str) -> str:
    """Repeatedly strip credit/version suffixes until the string stops changing.

    Looping is what makes this idempotent: "Song (feat. A) - 2011 Remaster"
    needs two passes, and a single pass would leave a result that a second call
    would strip further.
    """
    for _ in range(8):
        stripped = _VERSION.sub("", _FEAT.sub("", text))
        stripped = _EDGE_PUNCT.sub("", stripped)
        if stripped == text:
            break
        text = stripped
    return text


def normalize_field(text: str) -> str:
    """Normalize a single artist or track string."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _WS.sub(" ", text).strip()
    text = _strip_noise(text)
    return _WS.sub(" ", text).strip().casefold()


def canonical_key(artist: str, track: str) -> str:
    """The identity of an item across corpora. Idempotent under re-parsing."""
    return f"{normalize_field(artist)}{SEP}{normalize_field(track)}"


def add_item_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach an ``item_key`` column derived from artist and track."""
    df = df.copy()
    artist = df[schema.ARTIST].astype("string").fillna("")
    track = df[schema.TRACK].astype("string").fillna("")
    df["item_key"] = [canonical_key(a, t) for a, t in zip(artist, track)]
    return df


@dataclass
class Vocab:
    """Maps canonical item keys to contiguous integer ids.

    Ids 0 and 1 are reserved for padding and out-of-vocabulary. Real items
    start at 2, so ``len(vocab)`` is the embedding-table size including both
    reserved slots.
    """

    key_to_id: dict[str, int]
    id_to_key: list[str]
    counts: np.ndarray
    #: Display strings for text embedding, aligned to ``id_to_key``.
    display: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.id_to_key)

    @property
    def n_items(self) -> int:
        """Number of real items, excluding PAD and OOV."""
        return len(self.id_to_key) - FIRST_ITEM_ID

    def encode(self, keys) -> np.ndarray:
        return np.fromiter(
            (self.key_to_id.get(k, OOV_ID) for k in keys), dtype=np.int64, count=len(keys)
        )

    def coverage(self, keys) -> float:
        """Fraction of ``keys`` that are in vocabulary. The cold-start dial."""
        keys = list(keys)
        if not keys:
            return 0.0
        hits = sum(1 for k in keys if k in self.key_to_id)
        return hits / len(keys)


def build_vocab(df: pd.DataFrame, min_count: int = 5) -> Vocab:
    """Build a vocabulary from events, capped by minimum global play count.

    The cap is what keeps the full-catalog ranking in Phase 4 tractable. Items
    below the threshold fall into OOV rather than being dropped, so sequences
    keep their shape and the model still sees that *something* was played.
    """
    if "item_key" not in df.columns:
        df = add_item_keys(df)

    grouped = (
        df.groupby("item_key", observed=True)
        .agg(
            count=("item_key", "size"),
            artist=(schema.ARTIST, "first"),
            track=(schema.TRACK, "first"),
        )
        .reset_index()
    )
    kept = grouped[grouped["count"] >= min_count].sort_values(
        ["count", "item_key"], ascending=[False, True]
    )

    id_to_key = ["<pad>", "<oov>"] + kept["item_key"].tolist()
    key_to_id = {k: i for i, k in enumerate(id_to_key)}
    counts = np.concatenate([np.zeros(FIRST_ITEM_ID, dtype=np.int64), kept["count"].to_numpy()])
    display = [("", ""), ("", "")] + list(
        zip(kept["artist"].astype(str), kept["track"].astype(str))
    )

    return Vocab(key_to_id=key_to_id, id_to_key=id_to_key, counts=counts, display=display)
