"""Item-item co-occurrence kNN.

The strongest of the three non-neural baselines and the one that actually
generalizes: unlike the repeat baseline it can surface a track the user has
never played, by way of what tends to be played alongside what. If the
transformer beats popularity and repeat but not this, the sequence model is not
earning its complexity.

Co-occurrence is counted within a bounded window *and* within a session, so
"played in the same sitting" is what defines relatedness. Similarity is cosine
over co-occurrence counts, pruned to the top-K neighbours per item to keep the
matrix sparse at a realistic vocabulary size.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from melochron.data.vocab import OOV_ID, PAD_ID


class ItemKNNScorer:
    name = "item-knn"

    def __init__(
        self,
        vocab_size: int,
        window: int = 5,
        top_k: int = 200,
        session_gap_s: int = 30 * 60,
        halflife_events: float = 8.0,
        recent_n: int = 30,
    ):
        self.vocab_size = vocab_size
        self.window = window
        self.top_k = top_k
        self.session_gap_s = session_gap_s
        self.decay = np.log(2.0) / halflife_events
        self.recent_n = recent_n
        self.sim: sparse.csr_matrix | None = None

    def _fold(self, rows: list[np.ndarray], cols: list[np.ndarray]) -> sparse.csr_matrix:
        """Fold a batch of co-occurrence pairs into a symmetric CSR."""
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        # Symmetrize: co-occurrence is an undirected relation here.
        return sparse.coo_matrix(
            (
                np.ones(len(r) * 2, dtype=np.float32),
                (np.concatenate([r, c]), np.concatenate([c, r])),
            ),
            shape=(self.vocab_size, self.vocab_size),
        ).tocsr()

    def fit(self, items_list: list[np.ndarray], times_list: list[np.ndarray]) -> ItemKNNScorer:
        # Folded in chunks rather than built in one allocation. On lastfm-1K the
        # complete pair list is ~19M events x 5 lags, doubled for symmetry:
        # ~190M entries, several GB of COO before it ever reaches CSR. Folding
        # every `flush_pairs` bounds the peak and costs only a few sparse adds,
        # which deduplicate as they go.
        flush_pairs = 20_000_000
        co: sparse.csr_matrix | None = None
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        pending = 0

        for items, ts in zip(items_list, times_list):
            if len(items) < 2:
                continue
            for lag in range(1, self.window + 1):
                if len(items) <= lag:
                    break
                a, b = items[:-lag], items[lag:]
                # Same session, and neither end reserved.
                within = (ts[lag:] - ts[:-lag]) <= self.session_gap_s * self.window
                ok = within & (a > OOV_ID) & (b > OOV_ID) & (a != b)
                if ok.any():
                    rows.append(a[ok])
                    cols.append(b[ok])
                    pending += int(ok.sum())

            if pending >= flush_pairs:
                chunk = self._fold(rows, cols)
                co = chunk if co is None else co + chunk
                rows, cols, pending = [], [], 0

        if rows:
            chunk = self._fold(rows, cols)
            co = chunk if co is None else co + chunk

        if co is None:
            self.sim = sparse.csr_matrix((self.vocab_size, self.vocab_size), dtype=np.float32)
            return self

        co.sum_duplicates()

        # Cosine normalization over co-occurrence mass.
        mass = np.asarray(co.sum(axis=1)).ravel()
        inv = np.zeros_like(mass)
        np.divide(1.0, np.sqrt(mass), out=inv, where=mass > 0)
        d = sparse.diags(inv.astype(np.float32))
        self.sim = (d @ co @ d).tocsr()
        self._prune()
        return self

    def _prune(self) -> None:
        """Keep only the top-K neighbours per item."""
        sim = self.sim
        assert sim is not None
        keep_data, keep_idx, keep_ptr = [], [], [0]
        for i in range(sim.shape[0]):
            start, stop = sim.indptr[i], sim.indptr[i + 1]
            data, idx = sim.data[start:stop], sim.indices[start:stop]
            if len(data) > self.top_k:
                sel = np.argpartition(-data, self.top_k)[: self.top_k]
                data, idx = data[sel], idx[sel]
            keep_data.append(data)
            keep_idx.append(idx)
            keep_ptr.append(keep_ptr[-1] + len(data))

        self.sim = sparse.csr_matrix(
            (np.concatenate(keep_data), np.concatenate(keep_idx), np.asarray(keep_ptr)),
            shape=sim.shape,
        )

    def score(self, histories: list[np.ndarray], times: list[np.ndarray]) -> np.ndarray:
        if self.sim is None:
            raise RuntimeError("ItemKNNScorer.fit must be called before score")

        out = np.zeros((len(histories), self.vocab_size), dtype=np.float32)
        for row, items in enumerate(histories):
            recent = items[-self.recent_n :]
            valid = recent > OOV_ID
            recent = recent[valid]
            if not len(recent):
                continue
            age = np.arange(len(recent) - 1, -1, -1, dtype=np.float64)
            w = np.exp(-self.decay * age).astype(np.float32)
            out[row] = w @ self.sim[recent]

        out[:, PAD_ID] = -np.inf
        out[:, OOV_ID] = -np.inf
        return out
