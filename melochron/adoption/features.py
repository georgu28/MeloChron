"""Genre vectors, and the user-history centroids built from them.

Music4All-Onion ships precomputed tf-idf matrices, so no sentence-transformer is
needed here -- the previous project's whole text pipeline collapses to reading a
TSV.

**Why genres and not tags.** Measured against the interaction catalogue:

    id_genres_tf-idf   685 dims   56,512/56,512 covered (100%)
    id_tags_tf-idf   2,275 dims   45,589/56,512 covered (80.7%)

The 80.7% is not the problem; where it is missing is. Tag coverage runs from
38.3% in the least-played popularity decile to 99.9% in the most-played, so a
tag-based discovery slice would be least trustworthy exactly where the discovery
claim lives -- the same trap the previous project hit with artist-level tags.
Genres are present in every decile, have 14 all-zero vectors out of 56,512, and
their density barely moves across popularity (2.03 tail, 2.51 head).

So genres define the slice; tags are available as an ablation and never as the
basis of a cut.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import csv as pacsv

from melochron.adoption.onion import open_stream

GENRES_FILE = "id_genres_tf-idf.tsv.bz2"
TAGS_FILE = "id_tags_tf-idf.tsv.bz2"


def read_header(path) -> list[str]:
    """Column names, read without decompressing the whole file."""
    with open_stream(path) as stream:
        head = stream.read(1 << 18)
    if hasattr(head, "to_pybytes"):
        head = head.to_pybytes()
    return head.decode("utf-8", errors="replace").splitlines()[0].split("\t")


def load_matrix(path, tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load a tf-idf TSV into a dense ``[n_tracks, dims]`` float32 matrix.

    Rows are aligned to *track code*, so row ``i`` is the vector for
    ``tracks[i]``; tracks absent from the file stay all-zero and are reported by
    the returned mask rather than silently passing as "no genres".

    Every value column is typed **float32 explicitly**. Letting pyarrow infer
    fails outright here: the first block of a sparse column is all zeros, it
    settles on int64, and a later block carrying `3.83` raises. That is a parse
    error rather than a silent corruption, but only because the columns happen
    to be sparse enough to fool the inference in a detectable way.
    """
    names = read_header(path)
    types = {name: pa.float32() for name in names[1:]}
    types[names[0]] = pa.string()

    with open_stream(path) as stream:
        table = pacsv.read_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=1 << 25),
            parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=pacsv.ConvertOptions(column_types=types),
        )

    index = {track: i for i, track in enumerate(tracks.tolist())}
    ids = table.column(0).to_pylist()
    pairs = [(local, index[track]) for local, track in enumerate(ids) if track in index]
    local_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    global_idx = np.array([p[1] for p in pairs], dtype=np.int64)

    matrix = np.zeros((tracks.shape[0], table.num_columns - 1), dtype=np.float32)
    for column in range(1, table.num_columns):
        values = table.column(column).to_numpy(zero_copy_only=False)
        matrix[global_idx, column - 1] = values[local_idx]

    present = np.zeros(tracks.shape[0], dtype=bool)
    present[global_idx] = True
    return matrix, present


def coverage_report(matrix: np.ndarray, present: np.ndarray, plays: np.ndarray) -> dict:
    """Coverage and density, broken down by popularity decile.

    The decile breakdown is the point. A vector source that is complete overall
    but thin on unpopular tracks cannot support a discovery claim, and an
    aggregate coverage number hides exactly that.
    """
    n_tracks = matrix.shape[0]
    nonzero = (matrix != 0).sum(axis=1)

    rank = np.empty(n_tracks, dtype=np.int64)
    rank[np.argsort(plays, kind="stable")] = np.arange(n_tracks)
    decile = rank * 10 // n_tracks

    by_decile = []
    for d in range(10):
        mask = decile == d
        by_decile.append(
            {
                "decile": d,
                "tracks": int(mask.sum()),
                "present_frac": round(float(present[mask].mean()), 4),
                "mean_nonzero": round(float(nonzero[mask].mean()), 2),
                "all_zero": int((mask & (nonzero == 0)).sum()),
            }
        )

    return {
        "dims": int(matrix.shape[1]),
        "tracks": n_tracks,
        "present": int(present.sum()),
        "present_frac": round(float(present.mean()), 4),
        "all_zero": int((nonzero == 0).sum()),
        "mean_nonzero": round(float(nonzero.mean()), 2),
        "by_popularity_decile": by_decile,
    }


def row_norms(matrix: np.ndarray) -> np.ndarray:
    """L2 norm of each row, with zeros left as zero rather than nan."""
    return np.sqrt((matrix.astype(np.float32) ** 2).sum(axis=1))


def sparse_triples(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nonzero structure as (row_start, col, value), CSR-style.

    Genre vectors average 2.2 nonzeros in 685 dimensions, so every centroid
    computation below works off this rather than the dense matrix. Keeping it
    explicit avoids a scipy dependency the project does not otherwise have.
    """
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols].astype(np.float32)
    counts = np.bincount(rows, minlength=matrix.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return starts, cols.astype(np.int32), values


def prefix_similarity(
    corpus,
    starts: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    norms: np.ndarray,
    dims: int,
    user: int,
    positions: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Cosine between each candidate track and the user's history *before* it.

    ``positions`` are encounter positions within this user's history and
    ``candidates`` the track encountered at each. The centroid for a position is
    the sum of genre vectors of every event strictly earlier in that user's
    history -- so the score never sees the encounter itself, nor anything after
    it.

    Computed with one scatter and one cumulative sum over the *requested*
    positions rather than every position, which keeps the working set at
    ``len(positions) x dims`` instead of the user's full history.
    """
    start, end = int(corpus.user_offsets[user]), int(corpus.user_offsets[user + 1])
    history = np.asarray(corpus.track_code[start:end])

    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    n_slots = sorted_positions.shape[0]

    # Which slot each historical event lands in: events before the first
    # requested position go to slot 0, and so on. Events at or after the last
    # requested position fall into a trailing slot that is never read.
    event_slot = np.searchsorted(sorted_positions, np.arange(history.shape[0]), side="right")

    # Expand the events' sparse genre entries, then bin them by (slot, genre) in
    # one bincount -- np.add.at on the same data is an order of magnitude slower.
    lengths = (starts[history + 1] - starts[history]).astype(np.int64)
    if lengths.sum() == 0:
        return np.zeros(n_slots, dtype=np.float32)

    event_of_entry = np.repeat(np.arange(history.shape[0]), lengths)
    entry_index = np.repeat(starts[history], lengths) + (
        np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    )
    slot_of_entry = event_slot[event_of_entry]

    flat = slot_of_entry.astype(np.int64) * dims + cols[entry_index]
    binned = np.bincount(flat, weights=values[entry_index], minlength=(n_slots + 1) * dims)
    centroids = np.cumsum(binned[: n_slots * dims].reshape(n_slots, dims), axis=0, dtype=np.float64)

    # Bucket k holds the events lying between requests k-1 and k, so the
    # centroid for request k is the running total *through* bucket k: an event
    # sits before position p_k exactly when its slot is <= k.
    sorted_candidates = candidates[order]
    cand_lengths = (starts[sorted_candidates + 1] - starts[sorted_candidates]).astype(np.int64)
    slot_of_cand = np.repeat(np.arange(n_slots), cand_lengths)
    cand_entry = np.repeat(starts[sorted_candidates], cand_lengths) + (
        np.arange(cand_lengths.sum())
        - np.repeat(np.cumsum(cand_lengths) - cand_lengths, cand_lengths)
    )

    if cand_entry.shape[0]:
        contrib = values[cand_entry] * centroids[slot_of_cand, cols[cand_entry]]
        # bincount rather than np.add.at: same result, roughly an order of
        # magnitude faster, and the slots are already grouped.
        dots = np.bincount(slot_of_cand, weights=contrib, minlength=n_slots)
    else:
        dots = np.zeros(n_slots, dtype=np.float64)

    centroid_norms = np.sqrt((centroids**2).sum(axis=1))
    denominator = centroid_norms * norms[sorted_candidates]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(denominator > 0, dots / denominator, 0.0)

    out = np.empty(n_slots, dtype=np.float32)
    out[order] = cosine.astype(np.float32)
    return out
