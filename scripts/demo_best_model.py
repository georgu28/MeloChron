"""Phase 5 demo: the best model's predictions on real listeners, with real names.

For a few *real* users spanning the slices (a heavy user, a cold user with no
history, a new-neighbourhood listener), show a handful of the tracks each met for
the first time, the **best model's** predicted probability that they would return,
and what actually happened. The best model is the content sequence encoder with the
in-context running rate handed in as a fixed residual base (PR-AUC 0.4520 on the
fixed cohort).

    python scripts/demo_best_model.py --out artifacts/adoption/demo-data.json

Unlike ``demo_adoption.py`` (which reuses the id-priors dump and shows opaque Onion
ids), this re-runs the best checkpoint and joins the 16-char Onion ids back to real
artist/song names through Music4All's ``id_information.csv`` (the Onion ids *are*
Music4All ids, 100% overlap). The per-encounter probabilities are the real
checkpoint outputs on the exact rows every table in the report was scored on.

Validity: the in-context feature is the exact output of
``baselines.incontext_user_rate`` (no look-ahead, resolution-gated), the same array
used as the model input during training. The scored rows are asserted equal to the
saved cohort and to the aligned ``cohort-scores.npz`` dump, and the overall PR-AUC
is asserted to reproduce the reported 0.4520 (the sanity gate).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from melochron.adoption import baselines
from melochron.adoption import cohort as cohorts
from melochron.adoption import train
from melochron.adoption.corpus import PLAUSIBLE_FLOOR, CompactCorpus
from melochron.adoption.labels import (
    EncounterTable,
    event_horizon,
    temporal_split,
    train_horizon_fits,
)
from melochron.adoption.train import Corpus, Examples

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_LABELS = Path("data/interim/onion-labels-v1")
DEFAULT_COHORT = Path("data/interim/onion-cohort-v1")
DEFAULT_SCORES = Path("artifacts/adoption/cohort-scores.npz")
DEFAULT_CKPT = Path("artifacts/adoption/runs-content-incontext-residual/residual/best.pt")
DEFAULT_INFO = Path("data/raw/music4all/music4all/id_information.csv")
DEFAULT_OUT = Path("artifacts/adoption/demo-data.json")

EXPECTED_PR_AUC = 0.4520  # the reported headline; the sanity gate tolerates +/- 0.01.


def load_names(info_csv: Path) -> dict[str, tuple[str, str]]:
    """id -> (artist, song) from Music4All's tab-separated id_information.csv."""
    names: dict[str, tuple[str, str]] = {}
    with open(info_csv, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: id, artist, song, album_name
        for row in reader:
            if len(row) >= 3:
                names[row[0]] = (row[1], row[2])
    return names


def within_halves(prob_u: np.ndarray, label_u: np.ndarray):
    """Return-rate in the top vs bottom half of one listener's predicted scores.

    The honest within-listener ranking signal: do the tracks the model rated
    higher for this listener actually come back more often? None if a median split
    leaves either half empty or single-class.
    """
    if prob_u.size < 4:
        return None
    med = np.median(prob_u)
    top, bot = label_u[prob_u >= med], label_u[prob_u < med]
    if top.size == 0 or bot.size == 0:
        return None
    return float(top.mean()), float(bot.mean())


def spread_rows(order_desc: np.ndarray, k: int) -> list[int]:
    """k evenly spaced ranks across a listener's predicted-sorted rows (high->low),
    so a shown table spans the model's full range for them rather than only its
    confident tail."""
    n = len(order_desc)
    if n <= k:
        return list(order_desc)
    return [int(order_desc[round(j * (n - 1) / (k - 1))]) for j in range(k)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--info", type=Path, default=DEFAULT_INFO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--dump-scores",
        type=Path,
        default=None,
        help="optional npz of the best model's per-row prob + aligned columns (for analysis)",
    )
    ap.add_argument("--per-user", type=int, default=6, help="tracks shown per user")
    ap.add_argument(
        "--expected-pr-auc",
        type=float,
        default=EXPECTED_PR_AUC,
        help="sanity-gate target for the checkpoint's overall cohort PR-AUC (+/- 0.01)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    # --- corpus, labels, horizon, split, cohort (mirrors train_seq_over_incontext) ---
    compact = CompactCorpus.load(args.store, mmap=True)
    table = EncounterTable(
        **{c: np.load(args.labels / f"{c}.npy", mmap_mode="r") for c in COLUMNS}
    )
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
        table.user_code,
        table.track_code,
        labels,
        table.encounter_ts,
        train_rows,
        compact.n_users,
        compact.n_tracks,
    )

    # --- the audited in-context running rate for the cohort (the model input) ---
    uc = np.asarray(table.user_code)
    ep = np.asarray(table.encounter_pos)
    resolution_pos = ep.astype(np.int64) + event_n
    resolution_pos[labels] = np.asarray(table.recur_pos)[labels]
    plausible = np.asarray(table.encounter_ts) >= PLAUSIBLE_FLOOR
    test_pool = split.is_test & horizon.observable & plausible
    ic_cohort, _seen = baselines.incontext_user_rate(
        uc,
        ep,
        resolution_pos,
        labels,
        test_pool,
        rows,
        prior=priors.global_rate,
        pseudocount=priors.user_pseudocount,
    )
    del resolution_pos

    # --- best model: load the checkpoint and score the cohort ---
    corpus = Corpus(
        track_code=np.asarray(compact.track_code),
        ts=np.asarray(compact.ts),
        user_offsets=np.asarray(compact.user_offsets),
    )
    model, _payload = train.load_checkpoint(args.checkpoint, device)
    max_len = model.config["max_len"]
    tc = np.asarray(np.load(args.labels / "track_code.npy", mmap_mode="r")[rows])
    ex = Examples(
        users=uc[rows],
        positions=ep[rows],
        candidates=tc,
        labels=np.asarray(labels[rows]),
    )
    ex.priors = ic_cohort[:, None].astype(np.float32)
    print("scoring the cohort with the best checkpoint ...", flush=True)
    prob = train.predict(model, corpus, ex, max_len, device)

    label = np.asarray(labels[rows]).astype(bool)

    # --- sanity gates: same cohort, aligned dump, reproduced headline ---
    assert np.array_equal(rows, cohort.rows), "scored rows drifted from the saved cohort"
    overall = float(average_precision_score(label, prob))
    print(
        f"overall PR-AUC (sanity gate, expect ~{args.expected_pr_auc}): {overall:.4f}",
        flush=True,
    )
    assert abs(overall - args.expected_pr_auc) < 0.01, (
        f"PR-AUC {overall:.4f} drifted from the expected {args.expected_pr_auc}"
    )

    # Reuse the aligned dump for the slice masks, item base rate, and user prior.
    dump = np.load(args.scores, allow_pickle=True)
    users = dump["users"]
    assert np.array_equal(users, uc[rows]), "dump users drifted from the cohort order"
    assert np.array_equal(dump["labels"].astype(bool), label), "dump labels drifted"
    item_rate = dump["col::item-rate"]
    user_prior = dump["col::user-prior"]
    slices = {k[len("slice::") :]: dump[k] for k in dump.files if k.startswith("slice::")}

    tracks = np.asarray(compact.tracks)
    names = load_names(args.info)
    enc_pos = np.asarray(np.load(args.labels / "encounter_pos.npy", mmap_mode="r")[rows])
    cold_slice = slices["cold_user"].astype(bool)

    if args.dump_scores is not None:
        args.dump_scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dump_scores,
            prob=prob,
            label=label,
            users=users,
            track_code=tc,
            enc_pos=enc_pos,
            item_rate=item_rate,
            user_prior=user_prior,
            **{f"slice::{k}": v for k, v in slices.items()},
        )
        print(f"dumped per-row scores to {args.dump_scores}", flush=True)

    base_rate = float(label.mean())

    # --- per-listener aggregates over the whole cohort (bincount, no O(U*N) loop) ---
    uniq, inv = np.unique(users, return_inverse=True)
    u_count = np.bincount(inv)
    u_return = np.bincount(inv, weights=label.astype(float)) / u_count  # actual return rate
    u_meanpred = np.bincount(inv, weights=prob.astype(float)) / u_count  # model's average call
    is_cold = np.zeros(uniq.shape[0], dtype=bool)
    is_cold[inv[cold_slice]] = True

    # Between-listener signal: does the model's average call track each listener's
    # actual openness? (correlation across listeners with a handful of encounters.)
    stable = u_count >= 5
    between_corr = float(np.corrcoef(u_meanpred[stable], u_return[stable])[0, 1])

    # Within-listener signal: pooled top-half vs bottom-half return rate, averaged
    # over listeners with enough rows and both outcomes present.
    tops, bots = [], []
    for j in range(uniq.shape[0]):
        if u_count[j] < 6:
            continue
        m = users == uniq[j]
        halves = within_halves(prob[m], label[m])
        if halves is not None:
            tops.append(halves[0])
            bots.append(halves[1])
    within_top, within_bot = float(np.mean(tops)), float(np.mean(bots))

    # --- deterministic, archetype-driven picks: the MOST ACTIVE listener in each
    #     openness band (chosen by activity + openness, never by model accuracy) ---
    def most_active(pos_mask: np.ndarray) -> int:
        cand = np.flatnonzero(pos_mask)
        return int(cand[np.argmax(u_count[cand])])

    reg_j = most_active((u_return >= base_rate - 0.05) & (u_return <= base_rate + 0.05))
    cold_j = most_active(is_cold & (u_return >= base_rate))  # an engaged newcomer
    open_j = most_active((u_return >= 0.55) & (u_count >= 40))
    picks = [
        ("a steady regular", "regular", reg_j),
        ("an engaged newcomer, no prior history", "newcomer", cold_j),
        ("an open enthusiast", "open", open_j),
    ]

    out = {
        "base_rate": round(base_rate, 4),
        "overall_pr_auc": round(overall, 4),
        "between": {
            "corr_meanpred_vs_actual": round(between_corr, 3),
            "n_listeners": int(stable.sum()),
        },
        "within": {
            "top_half_return": round(within_top, 3),
            "bottom_half_return": round(within_bot, 3),
            "gap": round(within_top - within_bot, 3),
            "n_listeners": len(tops),
        },
        "users": [],
    }
    for label_text, role, j in picks:
        u = uniq[j]
        m = users == u
        order = np.flatnonzero(m)[np.argsort(-prob[m])]  # rows high->low predicted
        halves = within_halves(prob[m], label[m])

        rtracks = []
        for i in spread_rows(order, args.per_user):
            tid = str(tracks[tc[i]])
            artist, song = names.get(tid, ("unknown artist", tid))
            rtracks.append(
                {
                    "artist": artist,
                    "song": song,
                    "predicted": round(float(prob[i]), 3),
                    "returned": bool(label[i]),
                }
            )

        out["users"].append(
            {
                "archetype": label_text,
                "role": role,
                "user_id": int(u),
                "n_encounters": int(m.sum()),
                "actual_rate": round(float(label[m].mean()), 3),
                "mean_pred": round(float(prob[m].mean()), 3),
                "within": None if halves is None else [round(halves[0], 3), round(halves[1], 3)],
                "tracks": rtracks,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
