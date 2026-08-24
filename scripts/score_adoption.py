"""Score saved adoption checkpoints on the fixed cohort, beside the baselines.

This is the scoring tail of ``train_adoption.py`` with the training removed: it
loads one or more checkpoints, refits the baselines on the *same* train rows,
and scores everything on the *same* fixed cohort through the *same* slice
definitions in ``report.py`` -- so the table is identical in construction to the
one the training script emits, only without paying to retrain.

    python scripts/score_adoption.py \
        --model pure artifacts/adoption/runs/adoption-pure/best.pt \
        --model priors artifacts/adoption/runs/adoption-priors/best.pt

Why it exists separately: on a small-RAM box the per-epoch window gathers thrash
swap, but a single scoring pass over the 500k-row cohort does not. Keeping the
trained checkpoint and scoring it here recovers the full slice table cheaply,
and Phase 5's demo needs a checkpoint-loading entry point regardless.

The ``EncounterTable`` columns are memory-mapped here (unlike the training
script, which loaded them resident): scoring never iterates them per-epoch, so
paging them lazily frees ~1.2 GB of anonymous RAM for the corpus page cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from melochron.adoption import baselines, metrics, report
from melochron.adoption import cohort as cohorts
from melochron.adoption.corpus import PLAUSIBLE_FLOOR, CompactCorpus
from melochron.adoption.labels import (
    EncounterTable,
    event_horizon,
    temporal_split,
    train_horizon_fits,
)
from melochron.adoption.train import Corpus, Examples, compiled_forward, load_checkpoint, predict

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_LABELS = Path("data/interim/onion-labels-v1")
DEFAULT_FEATURES = Path("data/interim/onion-features-v1")
DEFAULT_COHORT = Path("data/interim/onion-cohort-v1")
DEFAULT_REPORT = Path("artifacts/adoption/phase3-model.md")

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")


def build_examples(table, labels, rows, priors):
    """A scoring set for one column; priors attached only when the head needs them."""
    ex = Examples(
        users=np.asarray(table.user_code[rows]),
        positions=np.asarray(table.encounter_pos[rows]),
        candidates=np.asarray(table.track_code[rows]),
        labels=np.asarray(labels[rows]),
    )
    if priors is not None:
        ex.priors = np.column_stack(
            [priors.user_rate[ex.users], priors.item_rate[ex.candidates]]
        ).astype(np.float32)
    return ex


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=[],
        help="a checkpoint to score, as a column named 'model (NAME)'. Repeatable.",
    )
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=None, help="override; default from checkpoint")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--paired-vs",
        default="user x item",
        help="reference column for the paired-difference CI (empty to skip)",
    )
    ap.add_argument("--paired-rounds", type=int, default=1000)
    ap.add_argument(
        "--event-n",
        type=int,
        default=None,
        help="relabel at this event horizon N instead of the labels' manifest N",
    )
    ap.add_argument(
        "--dump", type=Path, default=None, help="save scored columns + slices to an .npz"
    )
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    compact = CompactCorpus.load(args.store, mmap=True)
    table = EncounterTable(**{c: np.load(args.labels / f"{c}.npy", mmap_mode="r") for c in COLUMNS})
    manifest = json.loads((args.labels / "manifest.json").read_text(encoding="utf-8"))
    # `--event-n` relabels at a different horizon than the one the checkpoints were
    # trained on (default). It refits the baselines at that N too, so the table is
    # "N-trained-and-fit baselines + N=<train>-trained model, scored on N labels" —
    # a horizon-robustness check, not a per-N retrain (which is prohibitive here).
    event_n = args.event_n or manifest["event_n"]
    horizon = event_horizon(compact, table, event_n)
    split = temporal_split(table, compact.n_users, seed=manifest["seed"])
    labels = horizon.label
    train_rows = np.flatnonzero(train_horizon_fits(split, horizon))

    cohort = cohorts.Cohort.load(args.cohort)
    rows = cohort.rows
    print(f"cohort: {len(cohort):,} rows from {cohort.users.shape[0]:,} users", flush=True)

    print("refitting priors on the identical train rows...", flush=True)
    priors = baselines.fit_priors(
        table.user_code,
        table.track_code,
        labels,
        table.encounter_ts,
        train_rows,
        compact.n_users,
        compact.n_tracks,
    )
    user_item = baselines.fit_user_item(
        priors, table.user_code, table.track_code, labels, train_rows, seed=args.seed
    )

    corpus = Corpus(
        track_code=np.asarray(compact.track_code),
        ts=np.asarray(compact.ts),
        user_offsets=np.asarray(compact.user_offsets),
    )

    cohort_users = np.asarray(table.user_code[rows])
    cohort_tracks = np.asarray(table.track_code[rows])
    cohort_positions = np.asarray(table.encounter_pos[rows])
    cohort_labels = np.asarray(labels[rows])
    # Same-cohort invariant: every column is scored on the saved cohort index.
    assert np.array_equal(rows, cohort.rows), "scored rows drifted from the saved cohort"

    similarity = report.genre_similarity(
        compact,
        np.load(args.features / "genres.npy"),
        cohort_users,
        cohort_positions,
        cohort_tracks,
    )
    named = report.cohort_slices(compact, table, split, rows, similarity)

    # Cold-ITEM slice: cohort encounters whose track never appears in the training
    # rows. This is the ID-vs-hybrid differentiator — an ID embedding has no learned
    # row for such a track (it falls back to a near-random init), while genre/hybrid
    # can still represent it from metadata. Added here, not in report.cohort_slices,
    # because it needs the train rows.
    train_tracks = np.unique(np.asarray(table.track_code[train_rows]))
    cold_item = ~np.isin(cohort_tracks, train_tracks)
    named["cold_item"] = cold_item
    print(f"cold_item slice: {int(cold_item.sum()):,} of {cold_item.shape[0]:,} cohort rows")

    # incontext-user-rate: the running per-user adoption rate a user's *own*
    # history yields at inference — the baseline that adjudicates whether the
    # cold_user win is more than in-context rate recovery (fitted priors are blind
    # on held-out users; a running rate is not). No look-ahead: a prior encounter
    # counts only once its horizon has closed (resolution_pos < encounter_pos).
    resolution_pos = np.asarray(table.encounter_pos).astype(np.int64) + event_n
    resolution_pos[labels] = np.asarray(table.recur_pos)[labels]
    pool_mask = (
        split.is_test & horizon.observable & (np.asarray(table.encounter_ts) >= PLAUSIBLE_FLOOR)
    )
    incontext_rate, incontext_seen = baselines.incontext_user_rate(
        np.asarray(table.user_code),
        np.asarray(table.encounter_pos),
        resolution_pos,
        labels,
        pool_mask,
        rows,
        prior=priors.global_rate,
        pseudocount=priors.user_pseudocount,
    )

    columns = {
        "global-prior": np.full(rows.shape[0], priors.global_rate),
        "incontext-user-rate": incontext_rate,
        "user-prior": priors.user_rate[cohort_users],
        "item-rate": priors.item_rate[cohort_tracks],
        "user x item": baselines.score_user_item(user_item, priors, cohort_users, cohort_tracks),
        "genre-sim": similarity,
    }

    for name, path in args.model:
        print(f"\nscoring model ({name}) from {path}...", flush=True)
        model, payload = load_checkpoint(Path(path), device)
        max_len = args.max_len or model.config["max_len"]
        use_priors = model.config["use_priors"]
        examples = build_examples(table, labels, rows, priors if use_priors else None)
        runner = compiled_forward(model, args.compile)
        probs = predict(model, corpus, examples, max_len, device, forward=runner)
        columns[f"model ({name})"] = probs
        print(
            f"  loaded: best val PR-AUC {payload['metrics'].get('best_val_pr_auc', 'n/a')}, "
            f"use_priors={use_priors}, max_len={max_len}",
            flush=True,
        )

    print("\nscoring all columns on the cohort...", flush=True)
    scores = report.score_columns(
        columns, cohort_labels, cohort_users, named, args.bootstrap, args.seed
    )

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump,
            labels=cohort_labels,
            users=cohort_users,
            incontext_seen=incontext_seen,
            **{f"col::{k}": v for k, v in columns.items()},
            **{f"slice::{k}": v.astype(bool) for k, v in named.items()},
        )
        print(f"dumped scored columns to {args.dump}", flush=True)

    paired_lines: list[str] = []
    ref = args.paired_vs
    model_cols = [c for c in columns if c.startswith("model (")]
    if ref and ref in columns and model_cols:
        print(
            f"\npaired-difference bootstrap vs '{ref}' ({args.paired_rounds} rounds)...", flush=True
        )
        key_slices = [
            "all",
            "cold_user",
            "cold_item",
            "unfamiliar (bottom decile)",
            "new_neighborhood",
        ]
        header = "| slice | " + " | ".join(f"{c} − {ref}" for c in model_cols) + " |"
        paired_lines = [header, "|---|" + "---|" * len(model_cols)]
        for sl in key_slices:
            mask = None if sl == "all" else named.get(sl)
            if sl != "all" and mask is None:
                continue
            cells = []
            for c in model_cols:
                d, lo, hi = metrics.paired_delta_pr_auc(
                    cohort_labels,
                    columns[c],
                    columns[ref],
                    cohort_users,
                    mask,
                    rounds=args.paired_rounds,
                    seed=args.seed,
                )
                star = " *" if (lo > 0 or hi < 0) else ""
                cells.append(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
            paired_lines.append(f"| {sl} | " + " | ".join(cells) + " |")

    base_rate = float(cohort_labels.mean())
    pr_lines = report.build_table(scores, "pr_auc")
    lift_lines = report.build_table(scores, "lift")
    print(f"\n# Cohort base rate {base_rate:.4f}\n")
    print("## PR-AUC")
    print("\n".join(pr_lines))
    print("\n## Lift")
    print("\n".join(lift_lines))
    if paired_lines:
        print(f"\n## Paired Δ PR-AUC vs '{ref}'  (* = 95% CI excludes 0)")
        print("\n".join(paired_lines))

    paired_section = (
        [
            f"## Paired Δ PR-AUC vs `{ref}`",
            "",
            (
                "Difference of PR-AUC per bootstrap round, resampling whole users, so "
                "the within-user correlation cancels. `*` marks a 95% interval that "
                "excludes 0 — a real win at this cohort's user count."
            ),
            "",
            *paired_lines,
            "",
        ]
        if paired_lines
        else []
    )

    lines = [
        "# Phase 3 — the model (scored from checkpoints)",
        "",
        (
            f"Scored on the same {len(cohort):,} cohort rows every baseline used. "
            f"Cohort base rate **{base_rate:.4f}**."
        ),
        "",
        "## PR-AUC by slice",
        "",
        "Bold is the best column in each row.",
        "",
        *pr_lines,
        "",
        "## Lift over base rate",
        "",
        *lift_lines,
        "",
        *paired_section,
        "## AUROC",
        "",
        *report.build_table(scores, "roc_auc"),
        "",
        "## Raw scores",
        "",
        "```json",
        json.dumps(scores, indent=2),
        "```",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
