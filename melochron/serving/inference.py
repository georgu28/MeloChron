"""Recommendation at request time.

The cold-start path is the reason this file is interesting, and it is worth
being precise about why it is simple. SASRec has **no per-user parameters**: it
conditions on the sequence, not on a user id. So serving a brand-new uploader
needs no per-user fitting, no user embedding, and no retraining. Map their
history to item ids, run the shared model, score.

That means "cold start" is not a special branch here. It is the ordinary path,
and the only thing that varies is how much of the uploader's history is in
vocabulary. Calling zero-shot inference "per-user adaptation" would be an
overclaim; if adaptation is ever added it has to arrive with a measured delta
against this, not as an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from melochron.data.vocab import FIRST_ITEM_ID, OOV_ID, PAD_ID, Vocab, canonical_key
from melochron.models.scorer import SASRecScorer


@dataclass
class Recommendation:
    item_id: int
    key: str
    artist: str
    track: str
    score: float


@dataclass
class RecommendationResult:
    items: list[Recommendation]
    #: Fraction of the uploaded history that was in vocabulary. The honest
    #: confidence signal to surface in the UI: at very low coverage the model
    #: is working from almost nothing and should say so rather than present
    #: confident-looking output.
    coverage: float
    history_length: int
    matched: int
    cold_start: bool

    def summary(self) -> dict:
        return {
            "returned": len(self.items),
            "coverage": round(self.coverage, 4),
            "history_length": self.history_length,
            "matched": self.matched,
            "cold_start": self.cold_start,
        }


class Recommender:
    """Wraps a loaded artifact with the request-time scoring path."""

    #: Below this in-vocabulary fraction, results are flagged rather than
    #: silently returned as if they were well-founded.
    COLD_START_COVERAGE = 0.25

    def __init__(self, scorer: SASRecScorer, vocab: Vocab, max_len: int = 200):
        self.scorer = scorer
        self.vocab = vocab
        self.max_len = max_len

    def encode_history(
        self, history: list[tuple[str, str, int]]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Map ``(artist, track, ts)`` triples to ids and timestamps.

        Out-of-vocabulary plays are kept as OOV rather than dropped. Dropping
        them would silently close the gaps in the sequence and make two plays
        months apart look consecutive, corrupting the time deltas the model
        reads.
        """
        if not history:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0.0

        history = sorted(history, key=lambda row: row[2])[-self.max_len :]
        keys = [canonical_key(artist, track) for artist, track, _ in history]
        ids = self.vocab.encode(keys)
        times = np.asarray([ts for _, _, ts in history], dtype=np.int64)

        matched = int((ids >= FIRST_ITEM_ID).sum())
        return ids, times, matched / len(ids)

    def recommend(
        self,
        history: list[tuple[str, str, int]],
        k: int = 20,
        exclude_history: bool = False,
    ) -> RecommendationResult:
        """Top-``k`` next-track predictions for one uploaded history.

        ``exclude_history`` is off by default, because replaying a known track
        is the single most common real outcome and filtering it would make the
        product disagree with the evaluation. It exists because a *discovery*
        surface wants the opposite, and that is a product choice rather than a
        modelling one.
        """
        ids, times, coverage = self.encode_history(history)
        if not len(ids):
            return RecommendationResult([], 0.0, 0, 0, cold_start=True)

        scores = self.scorer.score([ids], [times])[0]
        return self._rank(scores, ids, coverage, k=k, exclude_history=exclude_history)

    def recommend_batch(
        self,
        histories: list[list[tuple[str, str, int]]],
        k: int = 20,
        exclude_history: bool = False,
    ) -> list[RecommendationResult]:
        """Score several histories in one forward pass.

        Not a convenience loop: :meth:`SASRecScorer.score` materialises the
        whole ``[n_items, d_model]`` item table once per call, and for a
        projected text representation that projection is the dominant cost of a
        single request. Scoring n histories together pays it once instead of n
        times, so the per-request cost of a batch of 8 is far below 8x the cost
        of one.

        Empty histories keep their slot in the returned list rather than being
        dropped, so the caller can zip results back to requests positionally.
        """
        encoded = [self.encode_history(h) for h in histories]
        results = [RecommendationResult([], 0.0, 0, 0, cold_start=True) for _ in histories]

        live = [i for i, (ids, _, _) in enumerate(encoded) if len(ids)]
        if not live:
            return results

        scores = self.scorer.score([encoded[i][0] for i in live], [encoded[i][1] for i in live])
        for row, i in enumerate(live):
            ids, _, coverage = encoded[i]
            results[i] = self._rank(
                scores[row], ids, coverage, k=k, exclude_history=exclude_history
            )
        return results

    def _rank(
        self,
        scores: np.ndarray,
        ids: np.ndarray,
        coverage: float,
        k: int,
        exclude_history: bool,
    ) -> RecommendationResult:
        """Turn one row of catalog scores into a ranked result.

        Copies before masking. In the batched path ``scores`` is a view into
        the scorer's output array, and writing ``-inf`` through it would edit
        data the caller still owns --- harmless today because each row is read
        once, and exactly the kind of aliasing bug that stops being harmless
        the moment someone reuses the matrix.
        """
        scores = scores.copy()
        scores[PAD_ID] = -np.inf
        scores[OOV_ID] = -np.inf
        if exclude_history:
            seen = ids[ids >= FIRST_ITEM_ID]
            scores[seen] = -np.inf

        k = max(1, min(k, len(scores)))
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]

        items = []
        for item_id in top.tolist():
            artist, track = (
                self.vocab.display[item_id]
                if self.vocab.display and item_id < len(self.vocab.display)
                else ("", "")
            )
            items.append(
                Recommendation(
                    item_id=item_id,
                    key=self.vocab.id_to_key[item_id],
                    artist=artist,
                    track=track,
                    score=float(scores[item_id]),
                )
            )

        matched = int((ids >= FIRST_ITEM_ID).sum())
        return RecommendationResult(
            items=items,
            coverage=coverage,
            history_length=len(ids),
            matched=matched,
            cold_start=coverage < self.COLD_START_COVERAGE,
        )


@torch.no_grad()
def benchmark(
    recommender: Recommender,
    histories: list[list[tuple[str, str, int]]],
    k: int = 20,
    warmup: int = 5,
) -> dict:
    """Measure per-request latency. Reported as real numbers, not estimates."""
    import time

    for history in histories[:warmup]:
        recommender.recommend(history, k=k)

    timings = []
    for history in histories:
        started = time.perf_counter()
        recommender.recommend(history, k=k)
        timings.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(timings)
    return {
        "requests": len(values),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "p99_ms": round(float(np.percentile(values, 99)), 2),
        "mean_ms": round(float(values.mean()), 2),
        "catalog_size": int(recommender.scorer.model.n_items),
    }
