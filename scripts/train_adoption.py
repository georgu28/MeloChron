"""Phase 3: train the adoption model and score it beside every baseline.

    python scripts/train_adoption.py

Trains two heads on the reused SASRec encoder — pure-sequence and
sequence+priors — on a subsample of users sized to the GPU, and scores both on
the *same* fixed cohort the baselines used, in one combined table. The baselines
are recomputed here from the identical train rows, so the whole table is one
code path and the comparison is exact rather than two scripts agreeing.

The question the table answers: does either head beat `user × item` on
`cold_user` and `unfamiliar` — the slices where the per-user prior is weak and a
sequence model has something to add. Winning only overall means the model
rediscovered the user prior, which two lines of numpy already provide.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from melochron.adoption import baselines, report
from melochron.adoption import cohort as cohorts
from melochron.adoption import train as train_mod
from melochron.adoption.corpus import CompactCorpus
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
    predict,
    save_checkpoint,
    train,
)

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_LABELS = Path("data/interim/onion-labels-v1")
DEFAULT_FEATURES = Path("data/interim/onion-features-v1")
DEFAULT_COHORT = Path("data/interim/onion-cohort-v1")
DEFAULT_OUT = Path("artifacts/adoption/runs")
DEFAULT_REPORT = Path("artifacts/adoption/phase3-model.md")

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")


def build_examples(
    table: EncounterTable,
    labels: np.ndarray,
    rows: np.ndarray,
    priors: baselines.Priors | None,
) -> Examples:
    ex = Examples(
        users=table.user_code[rows],
        positions=table.encounter_pos[rows],
        candidates=table.track_code[rows],
        labels=labels[rows],
    )
    if priors is not None:
        ex.priors = np.column_stack(
            [priors.user_rate[ex.users], priors.item_rate[ex.candidates]]
        ).astype(np.float32)
    return ex


def subsample_users(user_code: np.ndarray, train_rows: np.ndarray, n_users: int, seed: int):
    """Pick whole users and keep all their train encounters.

    Whole users, because the encoder needs a user's history intact and a per-user
    rate estimated from a fraction of someone's encounters is not the rate a
    deployment would hold.
    """
    users_present = np.unique(user_code[train_rows])
    if n_users >= users_present.shape[0]:
        return train_rows
    rng = np.random.default_rng(seed)
    chosen = rng.choice(users_present, size=n_users, replace=False)
    keep = np.isin(user_code[train_rows], chosen)
    return train_rows[keep]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--train-users", type=int, default=30_000)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--item-variant", default="id")
    ap.add_argument("--no-time", action="store_true", help="position-only ablation")
    ap.add_argument("--heads", nargs="+", default=["pure", "priors"], choices=["pure", "priors"])
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (needs Triton — Linux/WSL, not Windows). "
        "The real throughput lever: fuses this small model's many tiny kernels.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # TF32 for any fp32 matmul that escapes autocast; free on Ampere/Ada.
    torch.set_float32_matmul_precision("high")

    compact = CompactCorpus.load(args.store, mmap=True)
    table = EncounterTable(**{c: np.load(args.labels / f"{c}.npy") for c in COLUMNS})
    manifest = json.loads((args.labels / "manifest.json").read_text(encoding="utf-8"))
    horizon = event_horizon(compact, table, manifest["event_n"])
    split = temporal_split(table, compact.n_users, seed=manifest["seed"])
    labels = horizon.label
    train_rows = np.flatnonzero(train_horizon_fits(split, horizon))

    cohort = cohorts.Cohort.load(args.cohort)
    rows = cohort.rows
    print(f"cohort: {len(cohort):,} rows from {cohort.users.shape[0]:,} users (loaded, unchanged)")

    print("refitting priors on the identical train rows...")
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

    # Resident arrays for fast window gathers (3 GB in RAM, not memmapped).
    print("loading corpus arrays into RAM for window building...")
    corpus = Corpus(
        track_code=np.asarray(compact.track_code),
        ts=np.asarray(compact.ts),
        user_offsets=np.asarray(compact.user_offsets),
    )

    fit_rows = subsample_users(table.user_code, train_rows, args.train_users, args.seed)
    print(
        f"training on {fit_rows.shape[0]:,} encounters "
        f"from {np.unique(table.user_code[fit_rows]).shape[0]:,} users"
    )

    n_items = compact.n_tracks + 1  # +1 for the reserved pad slot at id 0
    cohort_users = table.user_code[rows]
    cohort_labels = labels[rows]

    similarity = report.genre_similarity(
        compact,
        np.load(args.features / "genres.npy"),
        cohort_users,
        table.encounter_pos[rows],
        table.track_code[rows],
    )
    named = report.cohort_slices(compact, table, split, rows, similarity)

    columns = {
        "global-prior": np.full(rows.shape[0], priors.global_rate),
        "user-prior": priors.user_rate[cohort_users],
        "item-rate": priors.item_rate[table.track_code[rows]],
        "user x item": baselines.score_user_item(
            user_item, priors, cohort_users, table.track_code[rows]
        ),
        "genre-sim": similarity,
    }

    cohort_examples_pure = build_examples(table, labels, rows, None)
    cohort_examples_priors = build_examples(table, labels, rows, priors)

    runs = {}
    for head in args.heads:
        use_priors = head == "priors"
        name = f"model ({head})"
        print(f"\n=== training {name} ===")
        model = AdoptionModel(
            n_items=n_items,
            d_model=args.d_model,
            max_len=args.max_len,
            use_time=not args.no_time,
            use_priors=use_priors,
            item_variant=args.item_variant,
        ).to(device)
        examples = build_examples(table, labels, fit_rows, priors if use_priors else None)
        config = TrainConfig(
            max_len=args.max_len, batch_size=args.batch_size, epochs=args.epochs, seed=args.seed
        )
        started = time.time()
        result = train(
            model,
            corpus,
            examples,
            table.encounter_ts[fit_rows],
            config,
            device,
            compile=args.compile,
        )
        result["runtime_s"] = round(time.time() - started, 1)
        result["compiled"] = args.compile

        scoring = cohort_examples_priors if use_priors else cohort_examples_pure
        runner = train_mod.compiled_forward(model, args.compile)
        probs = predict(model, corpus, scoring, args.max_len, device, forward=runner)
        columns[name] = probs

        run_dir = args.out / f"adoption-{head}"
        save_checkpoint(run_dir / "best.pt", model, config, result)
        runs[name] = result
        print(
            f"  {name}: best val PR-AUC {result['best_val_pr_auc']:.4f} in {result['runtime_s']:.0f}s"
        )

    print("\nscoring all columns on the cohort...")
    scores = report.score_columns(
        columns, cohort_labels, cohort_users, named, args.bootstrap, args.seed
    )

    base_rate = float(cohort_labels.mean())
    lines = [
        "# Phase 3 — the model",
        "",
        (
            f"Both heads on the reused SASRec encoder, scored on the same "
            f"{len(cohort):,} cohort rows every baseline used. Item representation: "
            f"`{args.item_variant}`. Cohort base rate **{base_rate:.4f}**."
        ),
        "",
        "## PR-AUC by slice",
        "",
        "Bold is the best column in each row.",
        "",
        *report.build_table(scores, "pr_auc"),
        "",
        "## Lift over base rate",
        "",
        *report.build_table(scores, "lift"),
        "",
        "## AUROC",
        "",
        *report.build_table(scores, "roc_auc"),
        "",
        "## Training",
        "",
        "```json",
        json.dumps(runs, indent=2),
        "```",
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
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
