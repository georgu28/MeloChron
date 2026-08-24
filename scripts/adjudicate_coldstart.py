"""Adjudicate the cold_user win: model understanding, or in-context rate recovery?

On ``cold_user`` the fitted priors are structurally blind (a held-out user has no
train row, so ``user-prior`` and ``user×item`` fall back to global). The model
beats them — but it might only be recovering the user's *own* adoption rate from
their in-context test-period history, which is not the same as understanding
adoption. This pits the model against a baseline that does *exactly* that:
``incontext-user-rate`` (see ``baselines.incontext_user_rate``), computed at
inference with no look-ahead, and asks two questions with a paired user-bootstrap:

    Q1  does incontext-user-rate beat global-prior on cold_user?
        (if yes, the running rate alone is enough — the doubt is real)
    Q2  do id-pure / id-priors beat incontext-user-rate on cold_user, significantly?
        (NO  -> the win is in-context rate recovery; shrink the claim.
         YES -> the sequence adds signal beyond the rate; the residual is the win.)

CPU only: the model probabilities are read from the dump written by
``score_adoption.py --dump`` (real checkpoint outputs), so nothing is re-run.

    python scripts/adjudicate_coldstart.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from melochron.adoption import baselines, metrics
from melochron.adoption import cohort as cohorts
from melochron.adoption.corpus import PLAUSIBLE_FLOOR, CompactCorpus
from melochron.adoption.labels import (
    EncounterTable,
    event_horizon,
    temporal_split,
    train_horizon_fits,
)

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_LABELS = Path("data/interim/onion-labels-v1")
DEFAULT_COHORT = Path("data/interim/onion-cohort-v1")
DEFAULT_SCORES = Path("artifacts/adoption/cohort-scores.npz")
DEFAULT_OUT = Path("artifacts/adoption/coldstart-adjudication.md")

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")


def _fmt(s):
    ci = f" [{s.pr_auc_lo:.4f}, {s.pr_auc_hi:.4f}]" if s.pr_auc_lo is not None else ""
    return f"{s.pr_auc:.4f}{ci}  base {s.base_rate:.4f}  lift {s.lift:.3f}  auroc {s.roc_auc:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--paired-rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

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

    print("refitting priors (for global rate + user pseudocount)...", flush=True)
    priors = baselines.fit_priors(
        table.user_code,
        table.track_code,
        labels,
        table.encounter_ts,
        train_rows,
        compact.n_users,
        compact.n_tracks,
    )

    # incontext-user-rate on the cohort, no look-ahead (resolution-gated).
    resolution_pos = np.asarray(table.encounter_pos).astype(np.int64) + event_n
    resolution_pos[labels] = np.asarray(table.recur_pos)[labels]
    pool_mask = (
        split.is_test & horizon.observable & (np.asarray(table.encounter_ts) >= PLAUSIBLE_FLOOR)
    )
    incontext, seen = baselines.incontext_user_rate(
        np.asarray(table.user_code),
        np.asarray(table.encounter_pos),
        resolution_pos,
        labels,
        pool_mask,
        rows,
        prior=priors.global_rate,
        pseudocount=priors.user_pseudocount,
    )

    d = np.load(args.scores, allow_pickle=True)
    users = d["users"]
    y = d["labels"].astype(bool)
    cold = d["slice::cold_user"]
    cols = {
        "global-prior": d["col::global-prior"],
        "incontext-user-rate": incontext,
        "id-pure": d["col::model (pure)"],
        "id-priors": d["col::model (priors)"],
    }
    # Same-cohort invariant: the dump's labels/users must be the cohort rows, in
    # order, or the incontext column (computed on cohort.rows) would be misaligned.
    assert np.array_equal(y, np.asarray(labels[rows])), "dump labels != N-labels on cohort rows"
    assert np.array_equal(users, np.asarray(table.user_code[rows])), "dump users != cohort users"

    lines: list[str] = ["# Cold-start adjudication — is the cold_user win real?", ""]

    # Per-column table on cold_user (and all for context).
    for slname, mask in (("cold_user", cold), ("all", None)):
        lines += [f"## {slname}", ""]
        for name, v in cols.items():
            s = (
                metrics.evaluate(y, v, users, slname, args.bootstrap, args.seed)
                if mask is None
                else metrics.evaluate(
                    y[mask], v[mask], users[mask], slname, args.bootstrap, args.seed
                )
            )
            lines.append(f"- **{name}**: PR-AUC {_fmt(s)}")
        lines.append("")

    # Paired comparisons (the verdict).
    def paired(a, b, mask):
        return metrics.paired_delta_pr_auc(
            y, cols[a], cols[b], users, mask, rounds=args.paired_rounds, seed=args.seed
        )

    lines += ["## Paired Δ PR-AUC (resampling users; * = 95% CI excludes 0)", ""]
    comparisons = [
        ("Q1: incontext-user-rate − global-prior", "incontext-user-rate", "global-prior"),
        ("Q2: id-pure − incontext-user-rate", "id-pure", "incontext-user-rate"),
        ("Q2: id-priors − incontext-user-rate", "id-priors", "incontext-user-rate"),
    ]
    for slname, mask in (("cold_user", cold), ("all", None)):
        lines.append(f"**{slname}:**")
        for title, a, b in comparisons:
            dd, lo, hi = paired(a, b, mask)
            star = " *" if (lo > 0 or hi < 0) else ""
            lines.append(f"- {title}: {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
        lines.append("")

    # Effective-n: how much prior evidence the running rate actually had on cold_user.
    sc = seen[cold]
    q = np.quantile(sc, [0.25, 0.5, 0.75, 0.9]) if sc.size else [0, 0, 0, 0]
    lines += [
        "## Effective evidence (`seen`) on cold_user",
        "",
        (
            f"- rows: {int(cold.sum()):,}; with zero resolved priors: "
            f"{float((sc == 0).mean()) * 100:.1f}%"
        ),
        (
            f"- seen quantiles p25/p50/p75/p90: "
            f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f}; max {sc.max():.0f}"
        ),
        "",
    ]

    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
