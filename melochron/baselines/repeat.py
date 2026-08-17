"""Recency-weighted repeat baseline.

This is the baseline that decides whether the project has a result. Music
listening is dominated by replays, so a scorer that does nothing but rank the
user's own recent plays can post a strong aggregate HR@10 while generalizing
nothing. If the transformer does not clearly beat this on the *novel* slice,
the honest conclusion is that a cache was built, not a recommender.

It is therefore implemented to be genuinely strong rather than as a strawman:

* **Time-decayed**, not position-decayed. A play from ten minutes ago and one
  from ten months ago are different evidence even at the same position offset.
* **Frequency plus recency.** Repeated plays accumulate, so a long-standing
  favourite outranks something heard once an hour ago.
* **Session boost.** Tracks from the current listening session are weighted up,
  because within-session replay is the strongest short-horizon signal there is.
* **Popularity fallback** for items the user has never played, so the baseline
  degrades into the popularity baseline on the novel slice instead of
  collapsing to an all-zero tie.
"""

from __future__ import annotations

import numpy as np

from melochron.data.vocab import OOV_ID, PAD_ID


class RepeatScorer:
    name = "repeat"

    def __init__(
        self,
        vocab_size: int,
        popularity: np.ndarray | None = None,
        halflife_days: float = 7.0,
        session_gap_s: int = 30 * 60,
        session_boost: float = 1.5,
        pop_weight: float = 0.15,
    ):
        self.vocab_size = vocab_size
        self.decay = np.log(2.0) / (halflife_days * 86_400.0)
        self.session_gap_s = session_gap_s
        self.session_boost = session_boost

        if popularity is None:
            self._pop = np.zeros(vocab_size, dtype=np.float32)
        else:
            pop = np.log1p(popularity.astype(np.float64))
            denom = pop.max() if pop.max() > 0 else 1.0
            self._pop = (pop_weight * pop / denom).astype(np.float32)
        self._pop[PAD_ID] = -np.inf
        self._pop[OOV_ID] = -np.inf

    def score(self, histories: list[np.ndarray], times: list[np.ndarray]) -> np.ndarray:
        out = np.tile(self._pop, (len(histories), 1))

        for row, (items, ts) in enumerate(zip(histories, times)):
            if not len(items):
                continue

            now = int(ts[-1])
            age = (now - ts).astype(np.float64)
            weight = np.exp(-self.decay * age)

            # Everything after the last inactivity gap is the current session.
            gaps = np.diff(ts)
            breaks = np.nonzero(gaps > self.session_gap_s)[0]
            session_start = int(breaks[-1]) + 1 if len(breaks) else 0
            weight[session_start:] *= self.session_boost

            valid = items > OOV_ID
            if valid.any():
                np.add.at(out[row], items[valid], weight[valid])

        out[:, PAD_ID] = -np.inf
        out[:, OOV_ID] = -np.inf
        return out
