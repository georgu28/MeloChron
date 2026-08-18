"""Taste drift: how far a listener's centre of gravity moves over time.

Each time window gets one vector --- the mean of the learned item vectors for
everything played in it --- and drift is the cosine distance between those
centroids. Two numbers come out of every timeline and they answer different
questions: the **step** from the previous window ("how much did taste move this
quarter") and the **displacement** from the first ("how far from where it
started"). A listener who cycles between two moods has large steps and small
displacement; one who leaves a genre behind has the opposite.

Three things here look like details and are not:

* **Vectors are normalized before averaging, and the centroid is normalized
  again.** Item vectors carry no norm convention --- ``ProjectedTextEmbedding``
  runs the unit-norm sentence embeddings through an unbiased ``nn.Linear``,
  which destroys it, and ``IdEmbedding`` never had one. Averaging raw vectors
  measures a mix of direction and magnitude, and magnitude here is mostly an
  artifact of how often an item appeared in training.

* **Reserved ids are excluded.** PAD and OOV are exact zero rows under the text
  variants. Averaging them in pulls every centroid toward the origin by an
  amount that depends on how many unknown tracks a window happened to contain,
  which would show up as drift that is really just vocabulary coverage.

* **Thin windows are reported as gaps, not as points.** A centroid over three
  plays is noise, and a timeline that silently interpolates through it produces
  a smooth curve that means nothing. Sparse windows are emitted with null
  metrics and the next real step records how many window slots it jumped.

The metric is deliberately blind to *why* it moves. Validation against the
synthetic generator's known taste trajectory --- the only ground truth in the
project --- lives in ``tests/test_insights.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from melochron.data.sessions import Sequences
from melochron.data.vocab import FIRST_ITEM_ID, Vocab

SECONDS_PER_DAY = 86_400

#: One quarter. Long enough that a single binge does not define a window, short
#: enough to resolve a taste change inside a year of listening.
DEFAULT_WINDOW_DAYS = 90

#: Below this many plays a centroid is noise rather than a position.
DEFAULT_MIN_EVENTS = 10

DEFAULT_TOP_ITEMS = 5


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; all-zero rows stay zero rather than becoming NaN."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - cos`` for two unit vectors, clipped into ``[0, 2]``."""
    return float(np.clip(1.0 - float(np.dot(a, b)), 0.0, 2.0))


def window_index(times: np.ndarray, origin: int, window_days: int) -> np.ndarray:
    """Map unix seconds to a 0-based window number.

    Integer arithmetic on seconds rather than calendar periods: no timezone, no
    month-length special cases, and the same input always lands in the same bin.
    """
    span = window_days * SECONDS_PER_DAY
    return ((np.asarray(times, dtype=np.int64) - int(origin)) // span).astype(np.int64)


@dataclass
class DriftWindow:
    """One user's position in one time window."""

    user_id: str
    window: int
    start_ts: int
    n_events: int
    n_unique: int
    #: Null when the window is too thin to place, or is the first placed window.
    step: float | None = None
    displacement: float | None = None
    #: Window slots since the previous *placed* window. 1 is consecutive; larger
    #: means the step spans a dormancy and should not be read as a single hop.
    since_previous: int | None = None
    sparse: bool = False
    top_items: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "user_id": self.user_id,
            "window": self.window,
            "start_ts": self.start_ts,
            "n_events": self.n_events,
            "n_unique": self.n_unique,
            "step": None if self.step is None else round(self.step, 4),
            "displacement": None if self.displacement is None else round(self.displacement, 4),
            "since_previous": self.since_previous,
            "sparse": self.sparse,
            "top_items": self.top_items,
        }


@dataclass
class DriftTimeline:
    windows: list[DriftWindow]
    window_days: int = DEFAULT_WINDOW_DAYS
    min_events: int = DEFAULT_MIN_EVENTS

    def as_rows(self) -> list[dict]:
        return [w.as_row() for w in self.windows]

    def placed(self) -> list[DriftWindow]:
        """Windows that got a real centroid."""
        return [w for w in self.windows if not w.sparse]

    def summary(self) -> dict:
        steps = [w.step for w in self.windows if w.step is not None]
        displacements = [w.displacement for w in self.windows if w.displacement is not None]
        placed = self.placed()
        return {
            "users": len({w.user_id for w in self.windows}),
            "windows": len(self.windows),
            "placed_windows": len(placed),
            "sparse_windows": len(self.windows) - len(placed),
            "window_days": self.window_days,
            "mean_step": round(float(np.mean(steps)), 4) if steps else 0.0,
            "median_step": round(float(np.median(steps)), 4) if steps else 0.0,
            "mean_displacement": (
                round(float(np.mean(displacements)), 4) if displacements else 0.0
            ),
            "max_displacement": (round(float(np.max(displacements)), 4) if displacements else 0.0),
        }


def _label(vocab: Vocab | None, item_id: int) -> str:
    if vocab is None or item_id >= len(vocab.display) or not vocab.display:
        return str(item_id)
    artist, track = vocab.display[item_id]
    return f"{artist} - {track}" if artist or track else str(item_id)


def user_timeline(
    user_id: str,
    items: np.ndarray,
    times: np.ndarray,
    item_vectors: np.ndarray,
    origin: int,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_events: int = DEFAULT_MIN_EVENTS,
    vocab: Vocab | None = None,
    top_items: int = DEFAULT_TOP_ITEMS,
) -> list[DriftWindow]:
    """Drift windows for one user. ``item_vectors`` must already be normalized."""
    items = np.asarray(items, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    if len(items) != len(times):
        raise ValueError(f"got {len(items)} items but {len(times)} timestamps")

    windows = window_index(times, origin, window_days)
    out: list[DriftWindow] = []

    first_centroid: np.ndarray | None = None
    previous_centroid: np.ndarray | None = None
    previous_window: int | None = None

    for w in np.unique(windows):
        in_window = windows == w
        ids = items[in_window]
        real = ids[ids >= FIRST_ITEM_ID]

        start_ts = int(origin + int(w) * window_days * SECONDS_PER_DAY)
        record = DriftWindow(
            user_id=user_id,
            window=int(w),
            start_ts=start_ts,
            n_events=int(in_window.sum()),
            n_unique=len(np.unique(real)),
        )

        if len(real) < min_events:
            record.sparse = True
            out.append(record)
            continue

        centroid = item_vectors[real].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            # Every vector in the window cancelled out, or the representation
            # has not been trained. Either way there is no position to report.
            record.sparse = True
            out.append(record)
            continue
        centroid = centroid / norm

        if previous_centroid is not None and previous_window is not None:
            record.step = cosine_distance(centroid, previous_centroid)
            record.since_previous = int(w) - previous_window
        if first_centroid is not None:
            record.displacement = cosine_distance(centroid, first_centroid)
        else:
            first_centroid = centroid
            record.displacement = 0.0

        if top_items:
            unique, counts = np.unique(real, return_counts=True)
            ranked = unique[np.argsort(-counts)][:top_items]
            record.top_items = [_label(vocab, int(i)) for i in ranked]

        previous_centroid = centroid
        previous_window = int(w)
        out.append(record)

    return out


def compute(
    seqs: Sequences,
    item_vectors: np.ndarray,
    vocab: Vocab | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_events: int = DEFAULT_MIN_EVENTS,
    origin: int | None = None,
    top_items: int = DEFAULT_TOP_ITEMS,
    max_users: int | None = None,
) -> DriftTimeline:
    """Drift timelines for every user in ``seqs``.

    ``item_vectors`` is the ``[n_items, d]`` learned representation; it is
    normalized once here rather than per user. ``origin`` defaults to the
    earliest event in the corpus so that every user's windows line up on one
    shared calendar, which is what makes cross-user aggregation meaningful.
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days!r}")
    if len(item_vectors) == 0:
        raise ValueError("item_vectors is empty")

    unit = normalize_rows(item_vectors)

    if origin is None:
        starts = [int(t[0]) for t in seqs.times if len(t)]
        if not starts:
            return DriftTimeline([], window_days=window_days, min_events=min_events)
        origin = min(starts)

    windows: list[DriftWindow] = []
    for i, user_id in enumerate(seqs.user_ids):
        if max_users is not None and i >= max_users:
            break
        windows.extend(
            user_timeline(
                user_id,
                seqs.items[i],
                seqs.times[i],
                unit,
                origin=origin,
                window_days=window_days,
                min_events=min_events,
                vocab=vocab,
                top_items=top_items,
            )
        )

    return DriftTimeline(windows, window_days=window_days, min_events=min_events)
