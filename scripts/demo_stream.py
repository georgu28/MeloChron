"""Per-listener track streams for the write-up's live Demo.

Scores the cloud checkpoint on the fixed cohort (same path as score_checkpoint.py)
and, for three real listeners spanning the openness range, emits their tracks in
listening order with real artist/song names, the model's predicted chance of return,
and whether they actually came back. The write-up's Demo steps through these live.

    python scripts/demo_stream.py \
        --checkpoint artifacts/adoption/runs-full/residual/best.pt \
        --out artifacts/adoption/demo-stream.json
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
BASE = 0.3079

# (user_code, label, role) - three listeners spanning the openness range, since how
# open a listener is drives most of the prediction. The open enthusiast is also the
# rightmost highlighted dot in the Results scatter.
TARGETS = [
    (36791, "A picky listener", "picky"),
    (5820, "A fifty-fifty listener", "fifty"),
    (50465, "An open enthusiast", "open"),
]


def load_names(info_csv: Path) -> dict[str, tuple[str, str]]:
    names: dict[str, tuple[str, str]] = {}
    with open(info_csv, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                names[row[0]] = (row[1], row[2])
    return names


def stride(idx: np.ndarray, k: int) -> np.ndarray:
    if idx.size <= k:
        return idx
    picks = np.linspace(0, idx.size - 1, k).round().astype(int)
    return idx[np.unique(picks)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=Path("data/interim/onion-v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/onion-labels-v1"))
    ap.add_argument("--cohort", type=Path, default=Path("data/interim/onion-cohort-v1"))
    ap.add_argument("--info", type=Path, default=Path("data/raw/music4all/music4all/id_information.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/adoption/demo-stream.json"))
    ap.add_argument("--per-user", type=int, default=32)
    ap.add_argument("--expected-pr-auc", type=float, default=0.4820)
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
    model, _ = train.load_checkpoint(args.checkpoint, device)
    max_len = model.config["max_len"]
    tc = np.asarray(np.load(args.labels / "track_code.npy", mmap_mode="r")[rows])
    ex = Examples(users=uc[rows], positions=ep[rows], candidates=tc, labels=np.asarray(labels[rows]))
    ex.priors = ic_cohort[:, None].astype(np.float32)
    print("scoring the cohort ...", flush=True)
    prob = train.predict(model, corpus, ex, max_len, device)

    label = np.asarray(labels[rows]).astype(bool)
    users = uc[rows]
    overall = float(average_precision_score(label, prob))
    print(f"overall PR-AUC (expect ~{args.expected_pr_auc}): {overall:.4f}", flush=True)
    assert abs(overall - args.expected_pr_auc) < 0.01

    tracks = np.asarray(compact.tracks)
    names = load_names(args.info)
    enc_pos = np.asarray(np.load(args.labels / "encounter_pos.npy", mmap_mode="r")[rows])

    out = {"base_rate": BASE, "listeners": []}
    for uid, lbl, role in TARGETS:
        m = users == uid
        idx = np.flatnonzero(m)
        chrono = idx[np.argsort(enc_pos[idx])]  # listening order
        picks = stride(chrono, args.per_user)
        stream = []
        for i in picks:
            tid = str(tracks[tc[i]])
            artist, song = names.get(tid, ("unknown artist", tid))
            stream.append({
                "artist": artist,
                "song": song,
                "p": round(float(prob[i]), 3),
                "ret": bool(label[i]),
            })
        p_all, l_all = prob[m], label[m].astype(float)
        beats = float(((p_all - l_all) ** 2 < (BASE - l_all) ** 2).mean())
        out["listeners"].append({
            "label": lbl,
            "role": role,
            "return_rate": round(float(label[m].mean()), 3),
            "n_total": int(m.sum()),
            "beats_average": round(beats, 3),
            "tracks": stream,
        })
        print(f"{lbl}: {len(stream)} tracks, rate {label[m].mean():.2f}, beats-average {beats:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
