"""Global popularity baseline.

The floor. It ignores the user entirely, so anything that cannot beat it has
learned nothing personal at all. Its scores are constant across instances,
which produces enormous ties in the tail: this is the baseline that pessimistic
tie-breaking in ``eval.metrics`` exists to keep honest.
"""

from __future__ import annotations

import numpy as np

from melochron.data.vocab import FIRST_ITEM_ID, OOV_ID, PAD_ID, Vocab


class PopularityScorer:
    name = "popularity"

    def __init__(self, vocab: Vocab, train_counts: np.ndarray | None = None):
        counts = train_counts if train_counts is not None else vocab.counts
        scores = np.log1p(counts.astype(np.float64))
        # Reserved slots must never be recommended.
        scores[PAD_ID] = -np.inf
        scores[OOV_ID] = -np.inf
        self._scores = scores.astype(np.float32)
        self.vocab_size = len(scores)

    def score(self, histories: list[np.ndarray], times: list[np.ndarray]) -> np.ndarray:
        return np.broadcast_to(self._scores, (len(histories), self.vocab_size))


def counts_from_sequences(items_list: list[np.ndarray], vocab_size: int) -> np.ndarray:
    """Play counts computed from training sequences only.

    Using ``Vocab.counts`` directly would count test-period plays, which leaks
    the future into the baseline and quietly inflates it.
    """
    counts = np.zeros(vocab_size, dtype=np.int64)
    for arr in items_list:
        if len(arr):
            counts += np.bincount(arr, minlength=vocab_size)[:vocab_size]
    counts[:FIRST_ITEM_ID] = 0
    return counts
