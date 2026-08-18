"""Session archetypes: clustering listening sessions into recognisable kinds.

A session becomes one vector --- the mean of the learned item vectors it
contains --- and KMeans over those vectors produces groups. The clustering is
the easy half. The half that decides whether the output is worth reading is how
the clusters get *named*.

**Clusters are labelled by lift, not by count.** The most-played artist inside a
cluster is, on a corpus with this popularity skew, usually just the most-played
artist overall. Labelling by raw count gives every cluster the same three names
and a reader would reasonably conclude the clustering found nothing. Lift ---
the artist's share inside the cluster over its share across all clustered
sessions --- answers the question actually being asked, which is what makes this
group *different*, and it is subject to a support floor so that a single play of
something obscure cannot post an enormous ratio.

Alongside the labels each cluster reports behavioural statistics: session
length, inter-event gap, repeat fraction, and hour of day. Those are what turn
a cluster id into a name a person can say out loud --- a group of long sessions
with low repeat and a 02:00 peak is a different animal from short, high-repeat,
commute-shaped bursts.

Two limits worth stating plainly rather than hiding:

* Sessions are **subsampled** on large corpora. Nineteen million events is well
  over a million sessions, and both KMeans and the silhouette score are far too
  expensive at that size for an insight that a sample answers just as well. The
  sample size is recorded in the output.
* Hour of day is **UTC**, because that is all the export carries. A "late night"
  cluster is late night in UTC, not in the listener's kitchen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from melochron.data.sessions import Sequences
from melochron.data.vocab import FIRST_ITEM_ID, Vocab
from melochron.insights.drift import normalize_rows

#: A one- or two-play session has no shape to cluster on.
DEFAULT_MIN_SESSION_LEN = 3

#: Sessions sampled for clustering. KMeans is fine at this size; silhouette is
#: the real constraint, being quadratic in the number of points it scores.
DEFAULT_MAX_SESSIONS = 50_000

#: Candidate cluster counts, inclusive. Reported for every k rather than
#: silently reduced to the winner.
DEFAULT_K_RANGE = (3, 8)

#: Points scored for the silhouette. Above a few thousand the estimate is
#: stable and the cost is not.
SILHOUETTE_SAMPLE = 5_000

#: An artist must appear this many times inside a cluster before its lift is
#: allowed to name that cluster.
DEFAULT_MIN_SUPPORT = 5

DEFAULT_TOP_LABELS = 5


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop index pairs for each run of equal values in a sorted array."""
    if len(values) == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(values)]])
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


@dataclass
class SessionTable:
    """Sessions reduced to vectors plus the behaviour that describes them."""

    vectors: np.ndarray
    user_ids: list[str]
    session_ids: np.ndarray
    start_ts: np.ndarray
    length: np.ndarray
    n_unique: np.ndarray
    repeat_frac: np.ndarray
    median_gap: np.ndarray
    hour: np.ndarray
    items: list[np.ndarray]
    #: Sessions that passed the length filter before any subsampling. Larger
    #: than ``len(self)`` either because sessions were sampled away or because
    #: they lost too many events to OOV; ``sampled`` distinguishes the two.
    n_total: int = 0
    sampled: bool = False
    #: Optional per-session playback statistics --- dwell, skip rate, shuffle
    #: rate --- keyed by name and aligned to the rows above. Empty on a corpus
    #: that does not carry them, which is why nothing here may assume they
    #: exist. lastfm-1K has none of these fields; a Spotify export has all of
    #: them, and they are what separates "played through, deliberate" from
    #: "skipped, shuffled, background".
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.user_ids)


def build_sessions(
    seqs: Sequences,
    item_vectors: np.ndarray,
    min_session_len: int = DEFAULT_MIN_SESSION_LEN,
    max_sessions: int | None = DEFAULT_MAX_SESSIONS,
    seed: int = 0,
) -> SessionTable:
    """Collapse ``seqs`` into one unit vector and one behaviour row per session.

    ``item_vectors`` is normalized here, once, rather than per session.
    """
    unit = normalize_rows(item_vectors)
    rng = np.random.default_rng(seed)

    # Enumerate first, sample second: building vectors for a million sessions
    # and then discarding most of them is the expensive way round.
    spans: list[tuple[int, int, int]] = []
    for u in range(len(seqs.user_ids)):
        for start, stop in _runs(np.asarray(seqs.sessions[u], dtype=np.int64)):
            if stop - start >= min_session_len:
                spans.append((u, start, stop))

    n_total = len(spans)
    sampled = max_sessions is not None and n_total > max_sessions
    if sampled:
        chosen = rng.choice(n_total, size=max_sessions, replace=False)
        chosen.sort()
        spans = [spans[i] for i in chosen]

    vectors = np.zeros((len(spans), unit.shape[1]), dtype=np.float32)
    user_ids: list[str] = []
    session_ids = np.zeros(len(spans), dtype=np.int64)
    start_ts = np.zeros(len(spans), dtype=np.int64)
    length = np.zeros(len(spans), dtype=np.int64)
    n_unique = np.zeros(len(spans), dtype=np.int64)
    repeat_frac = np.zeros(len(spans), dtype=np.float32)
    median_gap = np.zeros(len(spans), dtype=np.float32)
    hour = np.zeros(len(spans), dtype=np.int64)
    items: list[np.ndarray] = []

    keep = np.zeros(len(spans), dtype=bool)
    for row, (u, start, stop) in enumerate(spans):
        ids = np.asarray(seqs.items[u], dtype=np.int64)[start:stop]
        times = np.asarray(seqs.times[u], dtype=np.int64)[start:stop]
        real = ids[ids >= FIRST_ITEM_ID]
        if len(real) < min_session_len:
            continue

        centroid = unit[real].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            continue
        vectors[row] = centroid / norm

        user_ids.append(seqs.user_ids[u])
        session_ids[row] = int(seqs.sessions[u][start])
        start_ts[row] = int(times[0])
        length[row] = len(ids)
        n_unique[row] = len(np.unique(real))
        repeat_frac[row] = 1.0 - (len(np.unique(real)) / len(real))
        gaps = np.diff(times)
        median_gap[row] = float(np.median(gaps)) if len(gaps) else 0.0
        hour[row] = (int(times[0]) // 3600) % 24
        items.append(real)
        keep[row] = True

    return SessionTable(
        vectors=vectors[keep],
        user_ids=user_ids,
        session_ids=session_ids[keep],
        start_ts=start_ts[keep],
        length=length[keep],
        n_unique=n_unique[keep],
        repeat_frac=repeat_frac[keep],
        median_gap=median_gap[keep],
        hour=hour[keep],
        items=items,
        n_total=n_total,
        sampled=sampled,
    )


def align_signal(
    table: SessionTable,
    session_ids: np.ndarray,
    values: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """Put a per-session statistic into ``table`` row order, by session id.

    Sessions are subsampled and length-filtered, so a statistic computed over
    the whole corpus arrives in a different order and a different length.
    Joining on the id rather than zipping is the difference between a signal and
    a silent shuffle of one. Ids absent from ``session_ids`` get ``fill``.
    """
    session_ids = np.asarray(session_ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if len(session_ids) != len(values):
        raise ValueError(f"got {len(session_ids)} session ids but {len(values)} values")

    order = np.argsort(session_ids)
    keys, vals = session_ids[order], values[order]
    where = np.searchsorted(keys, table.session_ids)
    where = np.clip(where, 0, max(len(keys) - 1, 0))

    out = np.full(len(table.session_ids), fill, dtype=np.float64)
    if len(keys):
        hit = keys[where] == table.session_ids
        out[hit] = vals[where[hit]]
    return out


def _artist_of(vocab: Vocab | None, item_id: int) -> str:
    if vocab is None or not vocab.display or item_id >= len(vocab.display):
        return str(item_id)
    return vocab.display[item_id][0] or str(item_id)


def _label_of(vocab: Vocab | None, item_id: int) -> str:
    if vocab is None or not vocab.display or item_id >= len(vocab.display):
        return str(item_id)
    artist, track = vocab.display[item_id]
    return f"{artist} - {track}" if artist or track else str(item_id)


def _lift_labels(
    counts: dict[str, int],
    global_counts: dict[str, int],
    cluster_total: int,
    global_total: int,
    min_support: int,
    top: int,
) -> list[dict]:
    """Top labels by share-over-global-share, floored at ``min_support``."""
    scored = []
    for label, count in counts.items():
        if count < min_support:
            continue
        share = count / max(cluster_total, 1)
        base = global_counts.get(label, 0) / max(global_total, 1)
        if base <= 0:
            continue
        scored.append(
            {
                "label": label,
                "lift": round(share / base, 3),
                "share": round(share, 4),
                "count": int(count),
            }
        )
    scored.sort(key=lambda row: (-row["lift"], -row["count"]))
    return scored[:top]


@dataclass
class Archetype:
    cluster: int
    n_sessions: int
    share: float
    mean_length: float
    mean_unique: float
    mean_repeat_frac: float
    median_gap_s: float
    peak_hour: int
    hour_histogram: list[int] = field(default_factory=list)
    top_artists: list[dict] = field(default_factory=list)
    top_items: list[dict] = field(default_factory=list)
    #: Mean of each attached playback signal over this cluster's sessions.
    signals: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            **{k: round(v, 4) for k, v in self.signals.items()},
            "cluster": self.cluster,
            "n_sessions": self.n_sessions,
            "share": round(self.share, 4),
            "mean_length": round(self.mean_length, 2),
            "mean_unique": round(self.mean_unique, 2),
            "mean_repeat_frac": round(self.mean_repeat_frac, 4),
            "median_gap_s": round(self.median_gap_s, 1),
            "peak_hour_utc": self.peak_hour,
            "hour_histogram": self.hour_histogram,
            "top_artists": self.top_artists,
            "top_items": self.top_items,
        }


@dataclass
class SessionArchetypes:
    archetypes: list[Archetype]
    labels: np.ndarray
    k: int
    silhouette: dict[int, float]
    n_sessions: int
    n_total_sessions: int
    sampled: bool = False

    def as_rows(self) -> list[dict]:
        return [a.as_row() for a in self.archetypes]

    def summary(self) -> dict:
        return {
            "k": self.k,
            "n_sessions_clustered": self.n_sessions,
            "n_sessions_total": self.n_total_sessions,
            "sampled": self.sampled,
            "silhouette_by_k": {str(k): round(v, 4) for k, v in self.silhouette.items()},
            "best_silhouette": (
                round(max(self.silhouette.values()), 4) if self.silhouette else 0.0
            ),
        }


def choose_k(
    vectors: np.ndarray,
    k_range: tuple[int, int] = DEFAULT_K_RANGE,
    seed: int = 0,
) -> tuple[int, dict[int, float], KMeans]:
    """Fit every k in range and pick the best silhouette, reporting all of them."""
    low, high = k_range
    if low < 2:
        raise ValueError(f"k_range must start at 2 or more, got {k_range!r}")
    if len(vectors) <= low:
        raise ValueError(f"need more than {low} sessions to cluster, got {len(vectors)}")

    high = min(high, len(vectors) - 1)
    scores: dict[int, float] = {}
    fits: dict[int, KMeans] = {}

    for k in range(low, high + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        assigned = model.fit_predict(vectors)
        fits[k] = model
        if len(np.unique(assigned)) < 2:
            scores[k] = -1.0
            continue
        scores[k] = float(
            silhouette_score(
                vectors,
                assigned,
                sample_size=min(SILHOUETTE_SAMPLE, len(vectors)),
                random_state=seed,
            )
        )

    best = max(scores, key=lambda k: scores[k])
    return best, scores, fits[best]


def compute(
    table: SessionTable,
    vocab: Vocab | None = None,
    k_range: tuple[int, int] = DEFAULT_K_RANGE,
    seed: int = 0,
    min_support: int = DEFAULT_MIN_SUPPORT,
    top_labels: int = DEFAULT_TOP_LABELS,
) -> SessionArchetypes:
    """Cluster ``table`` and describe each cluster."""
    if len(table.vectors) == 0:
        raise ValueError("no sessions survived filtering; lower min_session_len")

    k, scores, model = choose_k(table.vectors, k_range=k_range, seed=seed)
    labels = model.labels_

    global_artists: dict[str, int] = {}
    global_items: dict[str, int] = {}
    for ids in table.items:
        for item_id in ids:
            artist = _artist_of(vocab, int(item_id))
            global_artists[artist] = global_artists.get(artist, 0) + 1
            label = _label_of(vocab, int(item_id))
            global_items[label] = global_items.get(label, 0) + 1
    global_total = sum(global_artists.values())
    global_item_total = sum(global_items.values())

    archetypes: list[Archetype] = []
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        if len(members) == 0:
            continue

        artists: dict[str, int] = {}
        items: dict[str, int] = {}
        for i in members:
            for item_id in table.items[i]:
                artist = _artist_of(vocab, int(item_id))
                artists[artist] = artists.get(artist, 0) + 1
                label = _label_of(vocab, int(item_id))
                items[label] = items.get(label, 0) + 1
        cluster_total = sum(artists.values())
        cluster_item_total = sum(items.values())

        histogram = np.bincount(table.hour[members], minlength=24)
        archetypes.append(
            Archetype(
                cluster=int(cluster),
                n_sessions=len(members),
                share=float(len(members) / len(labels)),
                mean_length=float(table.length[members].mean()),
                mean_unique=float(table.n_unique[members].mean()),
                mean_repeat_frac=float(table.repeat_frac[members].mean()),
                median_gap_s=float(np.median(table.median_gap[members])),
                peak_hour=int(np.argmax(histogram)),
                hour_histogram=[int(v) for v in histogram],
                top_artists=_lift_labels(
                    artists, global_artists, cluster_total, global_total, min_support, top_labels
                ),
                top_items=_lift_labels(
                    items,
                    global_items,
                    cluster_item_total,
                    global_item_total,
                    min_support,
                    top_labels,
                ),
                signals={
                    # nanmean: a session can be missing a signal without
                    # disqualifying the cluster from reporting the rest.
                    name: float(np.nanmean(values[members]))
                    for name, values in table.signals.items()
                    if not np.all(np.isnan(values[members]))
                },
            )
        )

    return SessionArchetypes(
        archetypes=archetypes,
        labels=labels,
        k=k,
        silhouette=scores,
        n_sessions=len(table.vectors),
        n_total_sessions=table.n_total or len(table.vectors),
        sampled=table.sampled,
    )
