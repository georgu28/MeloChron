"""Phase 5: the prediction demo (the brief's second deliverable).

For a handful of *real* users spanning the slices — a cold user with no history,
a heavy user, and someone meeting genuinely new-neighbourhood tracks — show a few
of their first-encountered tracks, the model's predicted adoption probability,
and what actually happened.

    python scripts/demo_adoption.py

It reads the per-row scores dumped by ``score_adoption.py --dump`` (the id-priors
model's probabilities on the fixed cohort) joined to the label table for track
ids and encounter positions. Those probabilities are the real checkpoint outputs;
reusing the dump avoids re-running the model and keeps the demo to the exact rows
every table in the report was scored on.

Music4All-Onion carries no titles or artists, so a track shows as its 16-char
Onion id — honest about what the dataset actually contains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from melochron.adoption.corpus import CompactCorpus

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_LABELS = Path("data/interim/onion-labels-v1")
DEFAULT_COHORT = Path("data/interim/onion-cohort-v1")
DEFAULT_SCORES = Path("artifacts/adoption/cohort-scores.npz")
DEFAULT_OUT = Path("artifacts/adoption/phase5-demo.md")

MODEL_COL = "col::model (priors)"


def pick_users(users, slices, per_user_rows, seed=0):
    """One user per archetype, deterministically, each with enough cohort rows."""
    rng = np.random.default_rng(seed)
    counts = per_user_rows
    chosen = {}

    def eligible(mask, min_rows):
        # users all of whose *shown* archetype holds and who have enough rows
        us = np.unique(users[mask])
        us = [u for u in us if counts.get(int(u), 0) >= min_rows]
        return us

    # Heavy user: the most first-encounters in the eval window (most active).
    heavy = max(counts, key=counts.get)
    chosen["heavy user"] = heavy

    # Cold user: no training history at all (the slice is defined that way).
    cold = [u for u in eligible(slices["cold_user"], 4) if u != heavy]
    if cold:
        chosen["cold user (no history)"] = int(rng.choice(cold))

    # New-neighbourhood: meets tracks sharing no genre with anything they've played.
    newn = [u for u in eligible(slices["new_neighborhood"], 3) if u not in chosen.values()]
    if newn:
        chosen["new-neighbourhood listener"] = int(rng.choice(newn))

    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-user", type=int, default=8, help="tracks shown per user")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    compact = CompactCorpus.load(args.store, mmap=True)
    tracks = np.asarray(compact.tracks)
    rows = np.load(args.cohort / "rows.npy")
    # Only two label columns are needed for the demo; load them directly rather
    # than constructing the full EncounterTable.
    track_code = np.asarray(np.load(args.labels / "track_code.npy", mmap_mode="r")[rows])
    enc_pos = np.asarray(np.load(args.labels / "encounter_pos.npy", mmap_mode="r")[rows])

    d = np.load(args.scores, allow_pickle=True)
    prob = d[MODEL_COL]
    label = d["labels"].astype(bool)
    users = d["users"]
    item_rate = d["col::item-rate"]
    user_prior = d["col::user-prior"]
    sl = {k[len("slice::") :]: d[k] for k in d.files if k.startswith("slice::")}

    counts = {int(u): int(c) for u, c in zip(*np.unique(users, return_counts=True))}
    picks = pick_users(users, sl, counts, seed=args.seed)

    lines = [
        "# Phase 5 — adoption demo",
        "",
        (
            "Real users from the fixed cohort, a few of the tracks each met for the "
            "first time, the **id-priors** model's predicted probability that they'd "
            "return, and what actually happened. Probabilities are the real checkpoint "
            "outputs (from `cohort-scores.npz`). Tracks show as their Onion id — the "
            "dataset has no titles or artists."
        ),
        "",
    ]

    for archetype, u in picks.items():
        m = users == u
        idx = np.flatnonzero(m)
        # Order by the model's confidence so the ranking is legible.
        idx = idx[np.argsort(-prob[idx])][: args.per_user]

        n_all = int(m.sum())
        actual_rate = float(label[m].mean())
        # Does the model separate returns from non-returns for this user?
        pos = prob[m][label[m]]
        neg = prob[m][~label[m]]
        sep = (
            f"{pos.mean():.2f} vs {neg.mean():.2f}" if pos.size and neg.size else "n/a (one class)"
        )

        lines += [
            f"## {archetype} — user #{u}",
            "",
            (
                f"- first-encounters in cohort: **{n_all}**, of which "
                f"**{actual_rate:.0%}** were actually adopted"
            ),
            (
                f"- mean predicted prob, adopted vs not: **{sep}** "
                f"(higher-left = the model separates them)"
            ),
            f"- user's historical adoption rate (prior): **{user_prior[m][0]:.2f}**",
            "",
            "| track (Onion id) | listen # | item base rate | predicted P(return) | returned? |",
            "|---|---|---|---|---|",
        ]
        for i in idx:
            tid = str(tracks[track_code[i]])
            got = "✅ yes" if label[i] else "❌ no"
            lines.append(
                f"| `{tid}` | {int(enc_pos[i]):,} | {item_rate[i]:.2f} | "
                f"**{prob[i]:.2f}** | {got} |"
            )
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
