"""Terminal experiment: does the sequence add anything OVER the in-context rate?

The cold-start adjudication showed the training-free ``incontext-user-rate`` beats
the encoder. This hands the model that exact rate as an input and measures the
sequence's contribution on top, two ways:

* **concat** -- ``[h, c, h⊙c, logit(incontext), logit(confidence)]`` through the
  ordinary head (``confidence = seen/(seen+pseudocount)`` is the evidence signal).
* **residual** -- the clean test: the head sees only ``[h, c, h⊙c]`` and predicts a
  *correction*; the output is ``correction + logit(incontext)`` with the base fixed,
  so the head can only add, never dilute the rate.

If seq+incontext does not beat incontext-alone, the encoder adds nothing even when
handed the baseline (a clean negative); if it does, the paired delta is the
sequence's true contribution over an honest floor.

    python scripts/train_seq_over_incontext.py --compile --variant both

Validity: the feature is the exact output of ``baselines.incontext_user_rate`` (no
look-ahead, resolution-gated), computed once and used as *both* the incontext-alone
baseline column *and* the model input. Cohort rows use the test-period pool
(byte-identical to the audited baseline); train rows use the analogous train-period
pool (a test-only pool is empty before the cutoff).
"""

from __future__ import annotations

import argparse
import json
import time
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
from melochron.adoption.model import AdoptionModel
from melochron.adoption.train import (
    Corpus,
    Examples,
    TrainConfig,
    compiled_forward,
    predict,
    save_checkpoint,
    train,
)

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")
DEFAULT_SCORES = Path("artifacts/adoption/cohort-scores.npz")


def subsample_users(user_code, train_rows, n_users, seed):
    present = np.unique(user_code[train_rows])
    if n_users >= present.shape[0]:
        return train_rows
    rng = np.random.default_rng(seed)
    chosen = rng.choice(present, size=n_users, replace=False)
    return train_rows[np.isin(user_code[train_rows], chosen)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=Path("data/interim/onion-v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/onion-labels-v1"))
    ap.add_argument("--features", type=Path, default=Path("data/interim/onion-features-v1"))
    ap.add_argument("--cohort", type=Path, default=Path("data/interim/onion-cohort-v1"))
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=Path("artifacts/adoption/runs-seq-over-incontext"))
    ap.add_argument(
        "--report", type=Path, default=Path("artifacts/adoption/phase4-seq-over-incontext.md")
    )
    ap.add_argument("--variant", choices=["concat", "residual", "both"], default="both")
    ap.add_argument("--train-users", type=int, default=15_000)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--paired-rounds", type=int, default=500)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
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
    print(f"cohort: {len(cohort):,} rows from {cohort.users.shape[0]:,} users", flush=True)

    priors = baselines.fit_priors(
        table.user_code,
        table.track_code,
        labels,
        table.encounter_ts,
        train_rows,
        compact.n_users,
        compact.n_tracks,
    )
    fit_rows = subsample_users(table.user_code, train_rows, args.train_users, args.seed)
    print(f"training on {fit_rows.shape[0]:,} encounters", flush=True)

    # The in-context rate, computed once by the audited function. Cohort rows use
    # the test-period pool (identical to the incontext-user-rate baseline); train
    # rows use the train-period pool so the head has a learnable signal.
    uc = np.asarray(table.user_code)
    ep = np.asarray(table.encounter_pos)
    resolution_pos = ep.astype(np.int64) + event_n
    resolution_pos[labels] = np.asarray(table.recur_pos)[labels]
    plausible = np.asarray(table.encounter_ts) >= PLAUSIBLE_FLOOR
    pc = priors.user_pseudocount
    ic_kw = {"prior": priors.global_rate, "pseudocount": pc}
    test_pool = split.is_test & horizon.observable & plausible
    train_pool = split.is_train & horizon.observable & plausible
    ic_cohort, seen_cohort = baselines.incontext_user_rate(
        uc, ep, resolution_pos, labels, test_pool, rows, **ic_kw
    )
    ic_fit, seen_fit = baselines.incontext_user_rate(
        uc, ep, resolution_pos, labels, train_pool, fit_rows, **ic_kw
    )
    del resolution_pos

    corpus = Corpus(
        track_code=np.asarray(compact.track_code),
        ts=np.asarray(compact.ts),
        user_offsets=np.asarray(compact.user_offsets),
    )
    cohort_users = uc[rows]
    cohort_labels = np.asarray(labels[rows])
    assert np.array_equal(rows, cohort.rows), "scored rows drifted from the saved cohort"

    def examples(query_rows, prior_cols):
        ex = Examples(
            users=uc[query_rows],
            positions=np.asarray(table.encounter_pos[query_rows]),
            candidates=np.asarray(table.track_code[query_rows]),
            labels=np.asarray(labels[query_rows]),
        )
        ex.priors = prior_cols.astype(np.float32)
        return ex

    conf_fit = seen_fit / (seen_fit + pc)
    conf_cohort = seen_cohort / (seen_cohort + pc)
    feats = {
        "concat": (
            np.column_stack([ic_fit, conf_fit]),
            np.column_stack([ic_cohort, conf_cohort]),
            {"use_priors": True, "n_prior_features": 2},
        ),
        "residual": (
            ic_fit[:, None],
            ic_cohort[:, None],
            {"use_priors": False, "residual_base": True},
        ),
    }
    wanted = ["concat", "residual"] if args.variant == "both" else [args.variant]

    columns = {"incontext-alone": ic_cohort}
    # The model is literally handed the incontext-alone column as its input.
    assert np.array_equal(columns["incontext-alone"], feats["residual"][1][:, 0])

    n_items = compact.n_tracks + 1
    runs = {}
    for name in wanted:
        fit_p, cohort_p, kw = feats[name]
        model = AdoptionModel(
            n_items=n_items, d_model=128, max_len=args.max_len, item_variant="id", **kw
        ).to(device)
        config = TrainConfig(
            max_len=args.max_len, batch_size=args.batch_size, epochs=args.epochs, seed=args.seed
        )
        print(f"\n=== training seq+incontext ({name}) ===", flush=True)
        started = time.time()
        result = train(
            model,
            corpus,
            examples(fit_rows, fit_p),
            np.asarray(table.encounter_ts[fit_rows]),
            config,
            device,
            compile=args.compile,
        )
        result["runtime_s"] = round(time.time() - started, 1)
        runner = compiled_forward(model, args.compile)
        probs = predict(
            model, corpus, examples(rows, cohort_p), args.max_len, device, forward=runner
        )
        save_checkpoint(args.out / name / "best.pt", model, config, result)
        columns[f"seq+incontext ({name})"] = probs
        runs[name] = result
        print(f"  {name}: best val PR-AUC {result['best_val_pr_auc']:.4f}", flush=True)

    # Context columns from the existing dump (same cohort order), if available.
    dump = np.load(args.scores, allow_pickle=True)
    if np.array_equal(dump["labels"].astype(bool), cohort_labels.astype(bool)):
        for src, dst in (("model (pure)", "id-pure"), ("model (priors)", "id-priors")):
            if f"col::{src}" in dump.files:
                columns[dst] = dump[f"col::{src}"]

    similarity = report.genre_similarity(
        compact,
        np.load(args.features / "genres.npy"),
        cohort_users,
        np.asarray(table.encounter_pos[rows]),
        np.asarray(table.track_code[rows]),
    )
    named = report.cohort_slices(compact, table, split, rows, similarity)
    scores = report.score_columns(
        columns, cohort_labels, cohort_users, named, args.bootstrap, args.seed
    )

    # Calibration (ECE) and the decisive paired deltas vs incontext-alone.
    ece = {
        c: {
            sl: round(
                metrics.expected_calibration_error(
                    cohort_labels if sl == "all" else cohort_labels[named[sl]],
                    v if sl == "all" else v[named[sl]],
                ),
                4,
            )
            for sl in ("all", "cold_user")
        }
        for c, v in columns.items()
    }

    verdict = ["## Verdict — seq+incontext − incontext-alone (paired, * = 95% CI excludes 0)", ""]
    for name in wanted:
        col = f"seq+incontext ({name})"
        for sl in ("all", "cold_user"):
            mask = None if sl == "all" else named["cold_user"]
            d, lo, hi = metrics.paired_delta_pr_auc(
                cohort_labels,
                columns[col],
                columns["incontext-alone"],
                cohort_users,
                mask,
                rounds=args.paired_rounds,
                seed=args.seed,
            )
            verdict.append(
                f"- {col} on {sl}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if (lo > 0 or hi < 0) else ''}"
            )
    verdict.append("")

    base_rate = float(cohort_labels.mean())
    lines = [
        "# Does the sequence add anything over the in-context rate?",
        "",
        (
            f"Cohort base rate **{base_rate:.4f}**. The in-context feature is the exact "
            "`incontext-user-rate` baseline (same array as the model input)."
        ),
        "",
        "## PR-AUC by slice",
        "",
        *report.build_table(scores, "pr_auc"),
        "",
        "## Lift over base rate",
        "",
        *report.build_table(scores, "lift"),
        "",
        "## Calibration (ECE, lower is better)",
        "",
        "| column | all | cold_user |",
        "|---|---|---|",
        *[f"| {c} | {ece[c]['all']} | {ece[c]['cold_user']} |" for c in columns],
        "",
        *verdict,
        "## Raw scores",
        "",
        "```json",
        json.dumps(scores, indent=2),
        "```",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(verdict))
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
