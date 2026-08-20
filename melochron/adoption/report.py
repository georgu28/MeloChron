"""Shared scoring and rendering for the cohort tables.

Both the baseline run and the model run score the *same* fixed cohort through
the *same* slice definitions, and the only way to guarantee that is one code
path. This module is it: genre similarity, the slice masks, per-column scoring,
and the markdown table. The Phase 3 report therefore places the model heads
beside the baselines in a single table built here, so the comparison is exact by
construction rather than by two scripts agreeing.
"""

from __future__ import annotations

import math

import numpy as np

from melochron.adoption import features, metrics
from melochron.adoption import slices as slicing


def genre_similarity(
    corpus,
    matrix: np.ndarray,
    user_code: np.ndarray,
    encounter_pos: np.ndarray,
    track_code: np.ndarray,
) -> np.ndarray:
    """Cosine to each user's pre-encounter genre centroid, for a set of rows.

    Rows are grouped by user so the sparse prefix-similarity kernel runs once
    per user over that user's requested positions, never over the whole history.
    """
    starts, cols, values = features.sparse_triples(matrix)
    norms = features.row_norms(matrix)
    dims = matrix.shape[1]

    order = np.argsort(user_code, kind="stable")
    sorted_users = user_code[order]
    edges = np.flatnonzero(np.concatenate([[True], sorted_users[1:] != sorted_users[:-1], [True]]))

    out = np.zeros(user_code.shape[0], dtype=np.float32)
    for i in range(edges.shape[0] - 1):
        block = slice(int(edges[i]), int(edges[i + 1]))
        rows_here = order[block]
        out[rows_here] = features.prefix_similarity(
            corpus,
            starts,
            cols,
            values,
            norms,
            dims,
            int(sorted_users[edges[i]]),
            encounter_pos[rows_here].astype(np.int64),
            track_code[rows_here].astype(np.int64),
        )
    return out


def cohort_slices(corpus, table, split, rows: np.ndarray, similarity: np.ndarray) -> dict:
    """The named boolean slices every column is scored on.

    Popularity deciles, cold-user and ordinal bands come from ``slices.py``; the
    two discovery slices are added here because they need the genre similarity.
    ``new_neighborhood`` is the strict "shares no genre with anything the user
    has played" cut; ``unfamiliar`` is the populated bottom-decile version that
    does not become tautological for the genre baseline.
    """
    keys = slicing.build(corpus, table, split)
    named = {k: v[rows] for k, v in slicing.named_slices(keys).items()}
    named["new_neighborhood"] = similarity == 0.0
    named["known_neighborhood"] = similarity > 0.0
    cut = float(np.quantile(similarity, 0.10))
    named["unfamiliar (bottom decile)"] = similarity <= cut
    return named


def build_table(scores: dict[str, list], field: str, better_high: bool = True) -> list[str]:
    """One markdown table: slices down the side, columns across the top."""
    columns = list(scores)
    slice_names = [s["slice"] for s in scores[columns[0]]]

    header = "| slice | base rate | " + " | ".join(columns) + " |"
    rule = "|---|---|" + "---|" * len(columns)
    rows = [header, rule]

    for i, name in enumerate(slice_names):
        base = scores[columns[0]][i]["base_rate"]
        values = [scores[c][i].get(field, float("nan")) for c in columns]
        finite = [v for v in values if not math.isnan(v)]
        best = (max if better_high else min)(finite) if finite else None
        cells = []
        for v in values:
            if math.isnan(v):
                cells.append("—")
            elif best is not None and abs(v - best) < 1e-12:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        rows.append(f"| {name} | {base:.4f} | " + " | ".join(cells) + " |")
    return rows


def score_columns(
    columns: dict[str, np.ndarray],
    labels: np.ndarray,
    users: np.ndarray,
    named_slices: dict[str, np.ndarray],
    bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Score every named column (baseline or model) on every slice."""
    out = {}
    for name, values in columns.items():
        scored = metrics.evaluate_slices(labels, values, users, named_slices, bootstrap, seed)
        out[name] = [s.as_dict() for s in scored]
    return out
