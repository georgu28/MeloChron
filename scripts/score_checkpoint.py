"""Score a trained seq+incontext checkpoint on the fixed cohort.

The training script (`train_seq_over_incontext.py`) trains and reports in one flow;
this is the report half on its own, for a checkpoint trained elsewhere (e.g. a
cloud GPU). It reproduces the exact cohort-scoring path so the PR-AUC is directly
comparable to the numbers in `README.md`/`process.md` (base rate 0.3079), and it
prints the paired delta over the training-free in-context rate.

    python scripts/score_checkpoint.py \
        --checkpoint artifacts/adoption/runs-full/residual/best.pt

The checkpoint is self-contained (`train.load_checkpoint` rebuilds the content
encoder with its frozen audio buffer), so no feature matrix is needed here; only
the corpus store, labels, cohort, and the genre matrix (for the slices) are read.
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
from melochron.adoption.train import Corpus, Examples, load_checkpoint, predict

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")


def resolve_slice(named: dict, want: str) -> str | None:
    """The cohort slice key, tolerant of suffixes (e.g. 'unfamiliar (bottom decile)')."""
    if want in named:
        return want
    return next((k for k in named if k.startswith(want)), None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=Path("data/interim/onion-v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/onion-labels-v1"))
    ap.add_argument("--features", type=Path, default=Path("data/interim/onion-features-v1"))
    ap.add_argument("--cohort", type=Path, default=Path("data/interim/onion-cohort-v1"))
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--paired-rounds", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    compact = CompactCorpus.load(args.store, mmap=True)
    table = EncounterTable(**{c: np.load(args.labels / f"{c}.npy", mmap_mode="r") for c in COLUMNS})
    manifest = json.loads((args.labels / "manifest.json").read_text(encoding="utf-8"))
    event_n = manifest["event_n"]
    horizon = event_horizon(compact, table, event_n)
    split = temporal_split(table, compact.n_users, seed=manifest["seed"])
    labels = horizon.label
    train_rows = np.flatnonzero(train_horizon_fits(split, horizon))

    cohort = cohorts.Cohort.load(args.cohort)
    rows = cohort.rows
    print(f"cohort: {len(rows):,} rows", flush=True)

    priors = baselines.fit_priors(
        table.user_code, table.track_code, labels, table.encounter_ts,
        train_rows, compact.n_users, compact.n_tracks,
    )

    # The audited in-context rate for the cohort (the fixed residual base).
    uc = np.asarray(table.user_code)
    ep = np.asarray(table.encounter_pos)
    resolution_pos = ep.astype(np.int64) + event_n
    resolution_pos[labels] = np.asarray(table.recur_pos)[labels]
    plausible = np.asarray(table.encounter_ts) >= PLAUSIBLE_FLOOR
    test_pool = split.is_test & horizon.observable & plausible
    ic_cohort, _ = baselines.incontext_user_rate(
        uc, ep, resolution_pos, labels, test_pool, rows,
        prior=priors.global_rate, pseudocount=priors.user_pseudocount,
    )
    del resolution_pos

    corpus = Corpus(
        track_code=np.asarray(compact.track_code),
        ts=np.asarray(compact.ts),
        user_offsets=np.asarray(compact.user_offsets),
    )
    model, _ = load_checkpoint(args.checkpoint, device)
    max_len = model.config["max_len"]
    print(f"checkpoint: item_variant={model.config['item_variant']} max_len={max_len}", flush=True)

    ex = Examples(
        users=uc[rows],
        positions=ep[rows],
        candidates=np.asarray(table.track_code[rows]),
        labels=np.asarray(labels[rows]),
    )
    ex.priors = ic_cohort[:, None].astype(np.float32)
    print("scoring the cohort ...", flush=True)
    probs = predict(model, corpus, ex, max_len, device)

    cohort_labels = np.asarray(labels[rows])
    cohort_users = uc[rows]
    columns = {"incontext-alone": ic_cohort, "content+incontext": probs}

    similarity = report.genre_similarity(
        compact,
        np.load(args.features / "genres.npy"),
        cohort_users,
        np.asarray(table.encounter_pos[rows]),
        np.asarray(table.track_code[rows]),
    )
    named = report.cohort_slices(compact, table, split, rows, similarity)
    scores = report.score_columns(columns, cohort_labels, cohort_users, named, args.bootstrap, args.seed)

    base_rate = float(cohort_labels.mean())
    print(f"\ncohort base rate {base_rate:.4f}\n")
    print(*report.build_table(scores, "pr_auc"), sep="\n")

    print("\nPaired delta: content+incontext minus incontext-alone (* = 95% CI clears 0)")
    wanted = ["all", "cold_user", "unfamiliar", "new_neighborhood"]
    for want in wanted:
        key = "all" if want == "all" else resolve_slice(named, want)
        if key is None:
            continue
        mask = None if key == "all" else named[key]
        d, lo, hi = metrics.paired_delta_pr_auc(
            cohort_labels, columns["content+incontext"], columns["incontext-alone"],
            cohort_users, mask, rounds=args.paired_rounds, seed=args.seed,
        )
        star = " *" if (lo > 0 or hi < 0) else ""
        print(f"  {key:22s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
