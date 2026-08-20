"""Turning an encounter into the history the encoder sees.

For one first encounter — a (user, track) pair at position ``e`` in that user's
history — the model is allowed to see the user's events at positions ``[0, e)``
and nothing at or after ``e``. This module builds the left-padded ``[B, max_len]``
item and gap windows for a batch of encounters, and it is where the single
invariant the whole task rests on lives: **the encounter and everything after it
must never enter its own window.**

Built per batch from the resident corpus arrays rather than materialised for
every example up front. Ten million encounters at ``max_len=200`` would be 16 GB
of int32; a batch of a few hundred is nothing, and the gather is vectorised.

Item ids are ``track_code + 1``: the encoder reserves id 0 for the pad slot, so
every real track shifts up by one and an empty history column reads as pad.
"""

from __future__ import annotations

import numpy as np

PAD_ID = 0


def build_windows(
    track_code: np.ndarray,
    ts: np.ndarray,
    user_offsets: np.ndarray,
    users: np.ndarray,
    positions: np.ndarray,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """History windows for a batch of encounters.

    ``users`` and ``positions`` are parallel arrays: encounter ``b`` belongs to
    user ``users[b]`` at within-user position ``positions[b]``. Returns
    ``item_ids`` and ``time_deltas``, both ``[B, max_len]`` int64, left-padded.

    Column ``j`` of row ``b`` holds the event at within-user position
    ``positions[b] - max_len + j``; columns whose position is negative are pad.
    The rightmost column is therefore the event immediately before the
    encounter, and a user with fewer than ``max_len`` prior events is padded on
    the left — the padding convention ``SASRec`` requires.
    """
    batch = users.shape[0]
    user_start = user_offsets[users].astype(np.int64)  # [B]
    positions = positions.astype(np.int64)

    # Within-user position of each column: [B, max_len]. Column j maps to
    # position (e - max_len + j), so the last column is e-1.
    col = np.arange(max_len, dtype=np.int64)
    within = positions[:, None] - max_len + col[None, :]
    valid = within >= 0

    # Global index into the corpus arrays, clamped where invalid so the gather
    # never reads out of the user's slice; the value is discarded by `valid`.
    global_idx = user_start[:, None] + np.clip(within, 0, None)

    items = np.where(valid, track_code[global_idx].astype(np.int64) + 1, PAD_ID)

    window_ts = np.where(valid, ts[global_idx].astype(np.int64), 0)
    # Gap from the previous column. The first real column's predecessor is a pad
    # column holding 0, so its delta is meaningless — but `TimeDeltaEncoding`
    # overrides the earliest real position to its own bucket, so whatever lands
    # there is discarded. Interior gaps are the genuine inter-event seconds.
    deltas = np.zeros((batch, max_len), dtype=np.int64)
    deltas[:, 1:] = window_ts[:, 1:] - window_ts[:, :-1]
    deltas = np.where(valid, np.clip(deltas, 0, None), 0)

    return items, deltas
