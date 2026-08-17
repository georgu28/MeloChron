"""Sessionization, positive filtering, and sequence assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from melochron import schema
from melochron.data import vocab as vocab_mod

DEFAULT_GAP_SECONDS = 30 * 60
DEFAULT_MIN_MS = 30_000


def sessionize(df: pd.DataFrame, gap_seconds: int = DEFAULT_GAP_SECONDS) -> pd.DataFrame:
    """Assign a ``session_id`` per user, cutting on an inactivity gap."""
    df = df.sort_values([schema.USER, schema.TS], kind="stable").reset_index(drop=True)
    delta = df.groupby(schema.USER, observed=True)[schema.TS].diff()
    new_session = (delta.isna()) | (delta > gap_seconds)
    df["session_id"] = new_session.cumsum().astype("int64")
    return df


def filter_positives(df: pd.DataFrame, min_ms: int = DEFAULT_MIN_MS) -> pd.DataFrame:
    """Keep only events that count as a deliberate play.

    Sources differ here and the difference is not cosmetic. Spotify exports
    carry ``ms_played``, so skips can be filtered out of the positives. The
    lastfm-1K corpus has no play duration at all: Last.fm only scrobbles a
    track once it has been played past roughly half its length, so every row is
    already an implicit positive. Rows with a null ``ms_played`` are therefore
    kept rather than dropped.

    The consequence is that the pretraining corpus has a slightly different
    positive definition from the fine-tuning corpus. That is worth stating in
    the README, because it caps how directly the two sets of numbers compare.
    """
    ms = df[schema.MS_PLAYED]
    keep = ms.isna() | (ms >= min_ms)
    return df[keep].reset_index(drop=True)


@dataclass
class Sequences:
    """Per-user play sequences, already encoded to vocabulary ids.

    Parallel lists, all the same length. ``items[i]``, ``times[i]``, and
    ``sessions[i]`` describe user ``user_ids[i]`` in ascending time order.
    """

    user_ids: list[str]
    items: list[np.ndarray]
    times: list[np.ndarray]
    sessions: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.user_ids)

    @property
    def n_events(self) -> int:
        return sum(len(x) for x in self.items)

    def describe(self) -> dict[str, float]:
        lengths = np.array([len(x) for x in self.items]) if self.items else np.array([0])
        return {
            "users": float(len(self.user_ids)),
            "events": float(self.n_events),
            "mean_len": float(lengths.mean()),
            "median_len": float(np.median(lengths)),
        }


def build_sequences(df: pd.DataFrame, vocab: vocab_mod.Vocab, min_len: int = 5) -> Sequences:
    """Encode a canonical event frame into per-user id sequences."""
    if "item_key" not in df.columns:
        df = vocab_mod.add_item_keys(df)
    if "session_id" not in df.columns:
        df = sessionize(df)

    df = df.sort_values([schema.USER, schema.TS], kind="stable")

    user_ids, items, times, sessions = [], [], [], []
    for user, g in df.groupby(schema.USER, observed=True, sort=True):
        if len(g) < min_len:
            continue
        user_ids.append(str(user))
        items.append(vocab.encode(g["item_key"].tolist()))
        times.append(g[schema.TS].to_numpy(dtype=np.int64))
        sessions.append(g["session_id"].to_numpy(dtype=np.int64))

    return Sequences(user_ids=user_ids, items=items, times=times, sessions=sessions)


def repeat_rate(seqs: Sequences) -> float:
    """Fraction of events whose item already appeared earlier for that user.

    This is the number that frames every result in the project. If it is high,
    a cache-like baseline will post strong aggregate metrics and the aggregate
    number stops being informative on its own.
    """
    total = seen_before = 0
    for arr in seqs.items:
        seen: set[int] = set()
        for it in arr.tolist():
            if it in seen:
                seen_before += 1
            seen.add(it)
            total += 1
    return seen_before / total if total else 0.0
