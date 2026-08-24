"""The encoder's fair last shot: does the sequence add anything over the strongest
baseline (the in-context running rate)?

The cold-start adjudication showed a training-free ``incontext-user-rate`` beats
the neural model overall and on cold_user. So give the encoder the feature it was
losing to: train a head fed the sequence **plus** three scalar priors
(user-prior, item-rate, incontext-rate), and compare it to

  * ``incontext-user-rate`` alone (the baseline it must beat), and
  * a **no-sequence** logistic on the same three priors (the control that
    isolates the sequence's marginal value).

If ``priors+ic`` does not beat the 3-feature logistic, the sequence adds nothing
over the features — the honest conclusion. Same fixed cohort, same metrics.

    python scripts/train_incontext_head.py --compile

The incontext rate here is the *deployable* variant: pooled over each user's
observable prior encounters across both periods (not test-only as in the
adjudicator), resolution-gated for no look-ahead, and restricted to the users
actually queried so the running-rate build stays cheap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

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


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def three_features(user_rate, item_rate, incontext, users, tracks):
    return np.column_stack([_logit(user_rate[users]), _logit(item_rate[tracks]), _logit(incontext)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=Path("data/interim/onion-v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/onion-labels-v1"))
    ap.add_argument("--features", type=Path, default=Path("data/interim/onion-features-v1"))
    ap.add_argument("--cohort", type=Path, default=Path("data/interim/onion-cohort-v1"))
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=Path("artifacts/adoption/runs-incontext"))
    ap.add_argument(
        "--report", type=Path, default=Path("artifacts/adoption/phase4-incontext-table.md")
    )
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
    user_item = baselines.fit_user_item(
        priors, table.user_code, table.track_code, labels, train_rows, seed=args.seed
    )

    fit_rows = subsample_users(table.user_code, train_rows, args.train_users, args.seed)
    print(f"training on {fit_rows.shape[0]:,} encounters", flush=True)

    # Deployable in-context running rate (resolution-gated, no look-ahead), pooled
    # over both periods and restricted to the users we actually query.
    uc = np.asarray(table.user_code)
    ep = np.asarray(table.encounter_pos)
    resolution_pos = ep.astype(np.int64) + event_n
    resolution_pos[labels] = np.asarray(table.recur_pos)[labels]
    relevant = np.union1d(np.unique(uc[fit_rows]), cohort.users)
    pool_mask = (
        horizon.observable
        & (np.asarray(table.encounter_ts) >= PLAUSIBLE_FLOOR)
        & np.isin(uc, relevant)
    )
    ic_kw = {"prior": priors.global_rate, "pseudocount": priors.user_pseudocount}
    ic_fit, _ = baselines.incontext_user_rate(
        uc, ep, resolution_pos, labels, pool_mask, fit_rows, **ic_kw
    )
    ic_cohort, _ = baselines.incontext_user_rate(
        uc, ep, resolution_pos, labels, pool_mask, rows, **ic_kw
    )
    del resolution_pos, pool_mask

    corpus = Corpus(
        track_code=np.asarray(compact.track_code),
        ts=np.asarray(compact.ts),
        user_offsets=np.asarray(compact.user_offsets),
    )

    fit_users = uc[fit_rows]
    fit_tracks = np.asarray(table.track_code[fit_rows])
    cohort_users = uc[rows]
    cohort_tracks = np.asarray(table.track_code[rows])
    cohort_labels = np.asarray(labels[rows])

    def stack3(users, tracks, ic):
        return np.column_stack([priors.user_rate[users], priors.item_rate[tracks], ic]).astype(
            np.float32
        )

    fit_examples = Examples(
        users=fit_users,
        positions=np.asarray(table.encounter_pos[fit_rows]),
        candidates=fit_tracks,
        labels=np.asarray(labels[fit_rows]),
    )
    fit_examples.priors = stack3(fit_users, fit_tracks, ic_fit)
    cohort_examples = Examples(
        users=cohort_users,
        positions=np.asarray(table.encounter_pos[rows]),
        candidates=cohort_tracks,
        labels=cohort_labels,
    )
    cohort_examples.priors = stack3(cohort_users, cohort_tracks, ic_cohort)

    n_items = compact.n_tracks + 1
    model = AdoptionModel(
        n_items=n_items,
        d_model=128,
        max_len=args.max_len,
        use_priors=True,
        n_prior_features=3,
        item_variant="id",
    ).to(device)
    config = TrainConfig(
        max_len=args.max_len, batch_size=args.batch_size, epochs=args.epochs, seed=args.seed
    )
    print("\n=== training model (priors+ic) ===", flush=True)
    started = time.time()
    result = train(
        model,
        corpus,
        fit_examples,
        np.asarray(table.encounter_ts[fit_rows]),
        config,
        device,
        compile=args.compile,
    )
    result["runtime_s"] = round(time.time() - started, 1)
    runner = compiled_forward(model, args.compile)
    probs_ic = predict(model, corpus, cohort_examples, args.max_len, device, forward=runner)
    save_checkpoint(args.out / "priors_ic" / "best.pt", model, config, result)
    print(f"  priors+ic: best val PR-AUC {result['best_val_pr_auc']:.4f}", flush=True)

    # No-sequence control: a logistic on the same three priors, fit on the same rows.
    logit3 = LogisticRegression(max_iter=1000)
    logit3.fit(
        three_features(priors.user_rate, priors.item_rate, ic_fit, fit_users, fit_tracks),
        np.asarray(labels[fit_rows]).astype(np.int8),
    )
    probs_logit3 = logit3.predict_proba(
        three_features(priors.user_rate, priors.item_rate, ic_cohort, cohort_users, cohort_tracks)
    )[:, 1]

    similarity = report.genre_similarity(
        compact,
        np.load(args.features / "genres.npy"),
        cohort_users,
        np.asarray(table.encounter_pos[rows]),
        cohort_tracks,
    )
    named = report.cohort_slices(compact, table, split, rows, similarity)
    train_tracks = np.unique(np.asarray(table.track_code[train_rows]))
    named["cold_item"] = ~np.isin(cohort_tracks, train_tracks)

    columns = {
        "user x item": baselines.score_user_item(user_item, priors, cohort_users, cohort_tracks),
        "incontext-user-rate": ic_cohort,
        "priors3 logistic (no seq)": probs_logit3,
        "model (priors+ic)": probs_ic,
    }
    dump = np.load(args.scores, allow_pickle=True)
    if "col::model (priors)" in dump.files and np.array_equal(
        dump["labels"].astype(bool), cohort_labels
    ):
        columns["model (id-priors)"] = dump["col::model (priors)"]

    scores = report.score_columns(
        columns, cohort_labels, cohort_users, named, args.bootstrap, args.seed
    )

    # The verdict: does the sequence add over the same features, and over the baseline?
    def paired(a, b, mask):
        return metrics.paired_delta_pr_auc(
            cohort_labels,
            columns[a],
            columns[b],
            cohort_users,
            mask,
            rounds=args.paired_rounds,
            seed=args.seed,
        )

    verdict = ["## Verdict — does the sequence add anything? (paired Δ, * = 95% CI excludes 0)", ""]
    pairs = [
        (
            "model (priors+ic) − priors3 logistic (no seq)",
            "model (priors+ic)",
            "priors3 logistic (no seq)",
        ),
        ("model (priors+ic) − incontext-user-rate", "model (priors+ic)", "incontext-user-rate"),
        (
            "priors3 logistic (no seq) − incontext-user-rate",
            "priors3 logistic (no seq)",
            "incontext-user-rate",
        ),
    ]
    for slname, mask in (("all", None), ("cold_user", named.get("cold_user"))):
        verdict.append(f"**{slname}:**")
        for title, a, b in pairs:
            dd, lo, hi = paired(a, b, mask)
            verdict.append(
                f"- {title}: {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if (lo > 0 or hi < 0) else ''}"
            )
        verdict.append("")

    base_rate = float(cohort_labels.mean())
    lines = [
        "# Phase 4 — the encoder's fair last shot (sequence + priors + incontext)",
        "",
        f"Cohort base rate **{base_rate:.4f}**. incontext is the deployable variant.",
        (
            f"priors+ic trained on {fit_rows.shape[0]:,} encounters, best val PR-AUC "
            f"{result['best_val_pr_auc']:.4f} in {result['runtime_s']:.0f}s."
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
