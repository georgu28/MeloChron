"""Aggregate demo charts for the write-up: the full cohort, not a 6-row sample.

Scores a trained seq+incontext checkpoint on the fixed cohort (the same path as
`score_checkpoint.py`) and emits the data behind the write-up's Demo charts:

  * a precision-recall curve for the content+rate model, the training-free
    in-context rate, and the base-rate floor (the PR-AUC story, drawn);
  * a reliability curve: bin every encounter into deciles of predicted chance and
    compare the model's average call to the actual return rate in each bin;
  * the between-listener signal: each listener's average predicted call vs their
    real return rate (correlation), as scatter points;
  * the within-listener signal: pooled top-half vs bottom-half return rate.

All of it is computed over the whole 500k-row cohort, so nothing here rests on a
handful of sampled tracks.

    python scripts/demo_charts.py \
        --checkpoint artifacts/adoption/runs-full/residual/best.pt \
        --expected-pr-auc 0.4820 --out artifacts/adoption/demo-charts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from melochron.adoption import baselines
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


def pr_envelope(label: np.ndarray, score: np.ndarray, grid: np.ndarray) -> list[float]:
    """Interpolated precision at each recall in `grid` (max precision at recall>=r)."""
    prec, rec, _ = precision_recall_curve(label, score)
    return [float(prec[rec >= r].max()) if np.any(rec >= r) else 0.0 for r in grid]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=Path("data/interim/onion-v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/onion-labels-v1"))
    ap.add_argument("--cohort", type=Path, default=Path("data/interim/onion-cohort-v1"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/adoption/demo-charts.json"))
    ap.add_argument("--dump-scores", type=Path, default=None,
                    help="optional npz cache of prob/label/users/ic_rate")
    ap.add_argument("--expected-pr-auc", type=float, default=0.4820,
                    help="sanity-gate target for the checkpoint's overall PR-AUC (+/- 0.01)")
    ap.add_argument("--pr-points", type=int, default=80)
    ap.add_argument("--deciles", type=int, default=10)
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

    tc = np.asarray(np.load(args.labels / "track_code.npy", mmap_mode="r")[rows])
    ex = Examples(users=uc[rows], positions=ep[rows], candidates=tc, labels=np.asarray(labels[rows]))
    ex.priors = ic_cohort[:, None].astype(np.float32)
    print("scoring the cohort ...", flush=True)
    prob = predict(model, corpus, ex, max_len, device)

    label = np.asarray(labels[rows]).astype(bool)
    users = uc[rows]
    base_rate = float(label.mean())

    overall = float(average_precision_score(label, prob))
    ic_pr = float(average_precision_score(label, ic_cohort))
    print(f"overall PR-AUC (expect ~{args.expected_pr_auc}): {overall:.4f}", flush=True)
    print(f"in-context PR-AUC (expect ~0.4212): {ic_pr:.4f}", flush=True)
    print(f"base rate: {base_rate:.4f}", flush=True)
    assert abs(overall - args.expected_pr_auc) < 0.01, (
        f"PR-AUC {overall:.4f} drifted from expected {args.expected_pr_auc}"
    )

    if args.dump_scores is not None:
        args.dump_scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.dump_scores, prob=prob, label=label, users=users, ic_rate=ic_cohort)
        print(f"cached scores to {args.dump_scores}", flush=True)

    # --- PR curves (interpolated precision on a shared recall grid) ---
    grid = np.linspace(0.0, 1.0, args.pr_points)
    pr = {
        "recall": [round(float(r), 4) for r in grid],
        "model": [round(p, 4) for p in pr_envelope(label, prob, grid)],
        "incontext": [round(p, 4) for p in pr_envelope(label, ic_cohort, grid)],
        "base_rate": round(base_rate, 4),
        "model_auc": round(overall, 4),
        "incontext_auc": round(ic_pr, 4),
    }

    # --- reliability curve: deciles of predicted chance, model call vs actual ---
    order = np.argsort(prob, kind="stable")
    bins = np.array_split(order, args.deciles)
    reliability = [
        {
            "mean_pred": round(float(prob[b].mean()), 4),
            "actual": round(float(label[b].mean()), 4),
            "count": int(b.size),
        }
        for b in bins
    ]

    # --- per-listener aggregates over the whole cohort ---
    uniq, inv = np.unique(users, return_inverse=True)
    u_count = np.bincount(inv)
    u_return = np.bincount(inv, weights=label.astype(float)) / u_count
    u_meanpred = np.bincount(inv, weights=prob.astype(float)) / u_count

    stable = u_count >= 5
    between_corr = float(np.corrcoef(u_meanpred[stable], u_return[stable])[0, 1])
    scatter = [
        [round(float(mp), 4), round(float(rt), 4), int(c)]
        for mp, rt, c in zip(u_meanpred[stable], u_return[stable], u_count[stable])
    ]

    tops, bots = [], []
    for j in range(uniq.shape[0]):
        if u_count[j] < 6:
            continue
        m = users == uniq[j]
        pu, lu = prob[m], label[m]
        med = np.median(pu)
        top, bot = lu[pu >= med], lu[pu < med]
        if top.size and bot.size:
            tops.append(float(top.mean()))
            bots.append(float(bot.mean()))
    within_top, within_bot = float(np.mean(tops)), float(np.mean(bots))

    out = {
        "base_rate": round(base_rate, 4),
        "overall_pr_auc": round(overall, 4),
        "incontext_pr_auc": round(ic_pr, 4),
        "pr_curve": pr,
        "reliability": reliability,
        "between": {
            "corr_meanpred_vs_actual": round(between_corr, 3),
            "n_listeners": int(stable.sum()),
            "scatter": scatter,
        },
        "within": {
            "top_half_return": round(within_top, 3),
            "bottom_half_return": round(within_bot, 3),
            "gap": round(within_top - within_bot, 3),
            "n_listeners": len(tops),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {k: v for k, v in out.items() if k not in ("pr_curve", "reliability")}
    summary["between"] = {k: v for k, v in out["between"].items() if k != "scatter"}
    print(json.dumps(summary, indent=2))
    print("reliability (decile mean_pred -> actual):")
    for i, r in enumerate(reliability):
        print(f"  d{i}: pred {r['mean_pred']:.3f}  actual {r['actual']:.3f}  n {r['count']:,}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
