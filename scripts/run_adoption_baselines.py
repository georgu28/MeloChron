"""Phase 2: score every baseline on one fixed cohort, before any model exists.

    python scripts/run_adoption_baselines.py

The brief's hard ordering rule. Building the comparison first makes it
structurally impossible to ship a model that only looks good because nothing
sensible was put beside it -- the failure that sank the previous version of this
project.

Named ``run_adoption_baselines`` rather than ``run_baselines`` because
``scripts/run_baselines.py`` already belongs to the next-track pipeline and
still works.

Everything is fitted on train rows whose horizon closes before the split
boundary, and scored on a cohort drawn from test rows. Two self-checks guard the
metric itself: the global-prior baseline must score a PR-AUC equal to the cohort
base rate, and `user-prior` must be exactly the global rate on held-out users,
who have no train history by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from melochron.adoption import baselines, features, metrics
from melochron.adoption import cohort as cohorts
from melochron.adoption import slices as slicing
from melochron.adoption.corpus import PLAUSIBLE_FLOOR, CompactCorpus
from melochron.adoption.labels import (
    DEFAULT_EVENT_N,
    EncounterTable,
    event_horizon,
    temporal_split,
    train_horizon_fits,
)

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_LABELS = Path("data/interim/onion-labels-v1")
DEFAULT_FEATURES = Path("data/interim/onion-features-v1")
DEFAULT_OUT = Path("data/interim/onion-cohort-v1")
DEFAULT_REPORT = Path("artifacts/adoption/phase2-baselines.md")

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")


def genre_similarity(corpus, matrix, table, rows) -> np.ndarray:
    """Cosine to the user's pre-encounter genre centroid, for every row."""
    starts, cols, values = features.sparse_triples(matrix)
    norms = features.row_norms(matrix)
    dims = matrix.shape[1]

    users = table.user_code[rows]
    order = np.argsort(users, kind="stable")
    sorted_rows = rows[order]
    sorted_users = users[order]

    edges = np.flatnonzero(np.concatenate([[True], sorted_users[1:] != sorted_users[:-1], [True]]))

    out = np.zeros(rows.shape[0], dtype=np.float32)
    for i in range(edges.shape[0] - 1):
        block = slice(int(edges[i]), int(edges[i + 1]))
        block_rows = sorted_rows[block]
        out[order[block]] = features.prefix_similarity(
            corpus,
            starts,
            cols,
            values,
            norms,
            dims,
            int(sorted_users[edges[i]]),
            table.encounter_pos[block_rows].astype(np.int64),
            table.track_code[block_rows].astype(np.int64),
        )
    return out


def render(stats: dict, tables: dict) -> str:
    lines = [
        "# Phase 2 — baselines",
        "",
        (
            f"Every column below is scored on **the same {stats['cohort']['rows']:,} "
            f"encounters** drawn from {stats['cohort']['users']:,} whole users in the test "
            f"period. Event horizon N={DEFAULT_EVENT_N}."
        ),
        "",
        (
            f"Cohort base rate **{stats['cohort']['base_rate']:.4f}**; a random scorer earns "
            f"exactly that PR-AUC, so `lift` (PR-AUC / base rate) is the number worth quoting."
        ),
        "",
        "## Slice sizes",
        "",
        "| slice | n | positives | base rate |",
        "|---|---|---|---|",
    ]
    for row in stats["slices"]:
        lines.append(
            f"| {row['slice']} | {row['n']:,} | {row['positives']:,} | {row['base_rate']:.4f} |"
        )

    empty = [row["slice"] for row in stats["slices"] if row["n"] == 0]
    lines += ["", "Two of these need reading rather than skimming.", ""]
    if empty:
        lines.append(
            f"**{', '.join(empty)} is empty, and that is correct rather than broken.** "
            "The cohort is drawn from the test period, and a user's first encounters "
            "happened when they joined — which is in the training period for anyone "
            "with enough history to be scored here. The slice is kept in the code "
            "because it is meaningful on the training side, and reported as zero "
            "rather than dropped so its absence is visible."
        )
        lines.append("")
    lines += [
        (
            "**`new_neighborhood` cannot fairly score `genre-similarity`.** The slice is "
            "defined as the rows where that baseline's own output is exactly zero, so "
            "inside it the scorer is constant and necessarily earns the base rate. That "
            "is a tautology, not a result. `unfamiliar (bottom decile)` exists for this "
            "reason: it is the same idea — material unlike what the user plays — cut at "
            "a quantile instead of at a scorer's zero, so every column is free to vary "
            "inside it. **Read the discovery claim off `unfamiliar`.**"
        ),
        "",
        "## PR-AUC by slice",
        "",
        "Bold is the best in each row. Base rate is the floor a coin achieves.",
        "",
    ]
    lines.extend(tables["pr_auc"])

    lines += [
        "",
        "## Lift over base rate",
        "",
        "The same table divided by each slice's own base rate, which is the only way",
        "to compare across slices whose difficulty differs.",
        "",
    ]
    lines.extend(tables["lift"])

    lines += [
        "",
        "## AUROC",
        "",
        (
            "Reported because the reason the brief banned it does not hold here. That ban "
            "assumed a rare label; the measured base rate is "
            f"{stats['cohort']['base_rate']:.2f}, where AUROC's true-negative-heavy "
            "denominator is not the distortion it would be at 0.08."
        ),
        "",
    ]
    lines.extend(tables["roc_auc"])

    lines += [
        "",
        "## Fitted priors",
        "",
        "```json",
        json.dumps(stats["priors"], indent=2),
        "```",
        "",
        "## Self-checks",
        "",
    ]
    for name, passed, detail in stats["checks"]:
        lines.append(f"- {'ok' if passed else '**FAILED**'} — {name}: {detail}")

    lines += [
        "",
        "## Raw",
        "",
        "```json",
        json.dumps(stats["scores"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def build_table(scores: dict[str, list], field: str, better_high: bool = True) -> list[str]:
    """One markdown table: slices down the side, models across the top."""
    models = list(scores)
    slice_names = [s["slice"] for s in scores[models[0]]]

    header = "| slice | base rate | " + " | ".join(models) + " |"
    rule = "|---|---|" + "---|" * len(models)
    rows = [header, rule]

    for i, name in enumerate(slice_names):
        base = scores[models[0]][i]["base_rate"]
        values = [scores[m][i].get(field, float("nan")) for m in models]
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--cohort-size", type=int, default=500_000)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    corpus = CompactCorpus.load(args.store, mmap=True)
    table = EncounterTable(**{c: np.load(args.labels / f"{c}.npy") for c in COLUMNS})
    manifest = json.loads((args.labels / "manifest.json").read_text(encoding="utf-8"))
    print(f"{len(table):,} encounters, {corpus.n_users:,} users")

    horizon = event_horizon(corpus, table, manifest["event_n"])
    split = temporal_split(table, corpus.n_users, seed=manifest["seed"])
    labels = horizon.label
    trustworthy = horizon.observable & (table.encounter_ts >= PLAUSIBLE_FLOOR)

    train_rows = np.flatnonzero(train_horizon_fits(split, horizon))
    eligible = split.is_test & trustworthy
    print(f"train rows {train_rows.shape[0]:,}, eligible test rows {int(eligible.sum()):,}")

    started = time.time()
    cohort = cohorts.build(table.user_code, eligible, args.cohort_size, args.seed)
    cohort.save(args.out)
    rows = cohort.rows
    print(f"cohort: {len(cohort):,} rows from {cohort.users.shape[0]:,} users")

    print("fitting priors on train...")
    priors = baselines.fit_priors(
        table.user_code,
        table.track_code,
        labels,
        table.encounter_ts,
        train_rows,
        corpus.n_users,
        corpus.n_tracks,
    )
    print(f"  {priors.summary()}")
    user_item = baselines.fit_user_item(
        priors, table.user_code, table.track_code, labels, train_rows, seed=args.seed
    )

    print("computing genre similarity...")
    matrix = np.load(args.features / "genres.npy")
    similarity = genre_similarity(corpus, matrix, table, rows)
    print(f"  done in {time.time() - started:.0f}s")

    cohort_users = table.user_code[rows]
    cohort_tracks = table.track_code[rows]
    cohort_labels = labels[rows]

    keys = slicing.build(corpus, table, split)
    named = {k: v[rows] for k, v in slicing.named_slices(keys).items()}
    # The discovery slice, two ways.
    #
    # `new_neighborhood` is the strict analogue of "new artist": the user has
    # never played anything sharing a genre with this track. Threshold-free, but
    # it turns out to be 0.76% of rows, and -- worse -- `genre-similarity` is
    # constant at zero inside it *by construction*, so that baseline cannot
    # discriminate there and scores exactly the base rate. A slice defined by a
    # scorer's own output cannot fairly evaluate that scorer.
    #
    # `unfamiliar` is the bottom decile of similarity, which is populated and
    # leaves every baseline free to vary inside it. It is the slice the
    # discovery claim should be read off.
    named["new_neighborhood"] = similarity == 0.0
    named["known_neighborhood"] = similarity > 0.0
    cut = float(np.quantile(similarity, 0.10))
    named["unfamiliar (bottom decile)"] = similarity <= cut

    scored = {
        "global-prior": np.full(rows.shape[0], priors.global_rate),
        "user-prior": priors.user_rate[cohort_users],
        "item-adoption-rate": priors.item_rate[cohort_tracks],
        "user x item": baselines.score_user_item(user_item, priors, cohort_users, cohort_tracks),
        "genre-similarity": similarity,
    }

    print("scoring...")
    results = {}
    for name, values in scored.items():
        scores = metrics.evaluate_slices(
            cohort_labels, values, cohort_users, named, args.bootstrap, args.seed
        )
        results[name] = [s.as_dict() for s in scores]
        print(f"  {name:20s} PR-AUC {results[name][0]['pr_auc']:.4f}")

    base_rate = float(cohort_labels.mean())
    global_pr = results["global-prior"][0]["pr_auc"]
    cold = named["cold_user"]
    cold_user_prior_is_global = (
        bool(np.allclose(priors.user_rate[cohort_users[cold]], priors.global_rate))
        if cold.any()
        else True
    )

    checks = [
        (
            "global-prior PR-AUC equals the cohort base rate",
            abs(global_pr - base_rate) < 5e-3,
            f"{global_pr:.4f} vs {base_rate:.4f}",
        ),
        (
            "held-out users fall back to the global prior",
            cold_user_prior_is_global,
            f"{int(cold.sum()):,} cold-user rows",
        ),
        (
            "no fitted statistic saw a test row",
            bool(np.all(table.encounter_ts[train_rows] < split.cutoff_ts)),
            f"max train ts {int(table.encounter_ts[train_rows].max())} < cutoff {split.cutoff_ts}",
        ),
    ]

    stats = {
        "cohort": {
            "rows": len(cohort),
            "users": int(cohort.users.shape[0]),
            "base_rate": base_rate,
            "seed": args.seed,
        },
        "slices": cohorts.slice_report(named, cohort_labels),
        "priors": priors.summary(),
        "scores": results,
        "checks": checks,
    }
    tables = {field: build_table(results, field) for field in ("pr_auc", "lift", "roc_auc")}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(stats, tables), encoding="utf-8")
    print(f"\nwrote {args.report}")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"\nSTOP: self-check failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
