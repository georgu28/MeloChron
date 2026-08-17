"""Phase 1 gate: score every baseline through the shared harness, one command.

    python scripts/run_baselines.py --data synthetic
    python scripts/run_baselines.py --data lastfm1k --path data/raw/lastfm-1k --users 200
    python scripts/run_baselines.py --data spotify --path data/raw/spotify-export

This is the table the transformer has to beat. Two properties of the wiring
here are load-bearing and easy to get backwards:

* **The vocabulary is built from the full frame, the counts and the fit are
  not.** The catalog is the universe of rankable items, not a learned
  parameter, so choosing it globally is not leakage. Building it from train
  instead would make every in-vocabulary target train-seen by construction and
  silently empty the cold-item slice, which is the Phase 2 transfer ablation.
* **Popularity counts come from training sequences only** via
  ``counts_from_sequences``. Using ``Vocab.counts`` would count test-period
  plays and quietly inflate the baseline it is meant to be a floor for.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from melochron import schema
from melochron.baselines.itemknn import ItemKNNScorer
from melochron.baselines.popularity import PopularityScorer, counts_from_sequences
from melochron.baselines.repeat import RepeatScorer
from melochron.data import lastfm, sessions, splits, spotify_export, synthetic, vocab
from melochron.eval import protocol, report


def load_events(args):
    if args.data == "synthetic":
        events, _ = synthetic.generate(
            synthetic.SyntheticConfig(n_users=args.users or 50, seed=args.seed)
        )
        return events
    if args.data == "parquet":
        # Preferred path: scripts/build_dataset.py already parsed, filtered and
        # sessionized. Re-parsing the 2.4 GB TSV per run costs ~70s for nothing.
        events = pd.read_parquet(args.path)
        if args.users:
            keep = sorted(events[schema.USER].unique())[: args.users]
            events = events[events[schema.USER].isin(keep)].reset_index(drop=True)
        return events
    if args.data == "lastfm1k":
        return lastfm.read_lastfm1k(args.path, users=args.users, limit=args.limit)
    if args.data == "spotify":
        return spotify_export.read_export(args.path)
    raise ValueError(f"unknown corpus {args.data!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data", choices=["synthetic", "parquet", "lastfm1k", "spotify"], default="synthetic"
    )
    ap.add_argument("--path", type=Path, help="parquet cache, or corpus dir for raw parsing")
    ap.add_argument("--users", type=int, default=None, help="cap on distinct users")
    ap.add_argument("--limit", type=int, default=None, help="cap on raw rows (lastfm1k)")
    ap.add_argument("--min-count", type=int, default=5, help="vocabulary play-count floor")
    ap.add_argument("--min-ms", type=int, default=sessions.DEFAULT_MIN_MS)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--holdout-user-frac", type=float, default=0.10)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--max-per-user", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/baselines"))
    args = ap.parse_args(argv)

    if args.data != "synthetic" and args.path is None:
        ap.error(f"--path is required for --data {args.data}")

    started = time.time()

    events = load_events(args)
    print(f"events                {len(events):>12,}")

    positives = sessions.filter_positives(events, min_ms=args.min_ms)
    print(f"positives             {len(positives):>12,}  ({len(positives) / len(events):.1%} kept)")

    split = splits.temporal_split(
        positives,
        test_frac=args.test_frac,
        holdout_user_frac=args.holdout_user_frac,
        seed=args.seed,
    )
    splits.assert_no_leakage(split)
    print(f"split                 {split.summary()}")

    # Catalog scope: full frame. See module docstring.
    vc = vocab.build_vocab(positives, min_count=args.min_count)
    print(f"vocabulary            {vc.n_items:>12,} items (+2 reserved)")

    all_seqs = sessions.build_sequences(positives, vc)
    train_seqs = sessions.build_sequences(split.train, vc)
    print(f"repeat rate           {sessions.repeat_rate(all_seqs):>12.1%}")

    train_items = {int(i) for arr in train_seqs.items for i in arr.tolist()}
    instances = protocol.build_instances(
        all_seqs,
        cutoff_ts=split.cutoff_ts,
        train_items=train_items,
        holdout_users=split.holdout_users,
        max_len=args.max_len,
        max_per_user=args.max_per_user,
        seed=args.seed,
    )
    print(f"instances             {instances.summary()}")

    if not len(instances):
        print("\nno evaluation instances; nothing to score", file=sys.stderr)
        return 1
    if instances.is_repeat.all():
        print(
            "\nWARNING: every target is a repeat, so the novel slice is empty. "
            "Aggregate numbers below measure caching, not recommendation.",
            file=sys.stderr,
        )

    counts = counts_from_sequences(train_seqs.items, len(vc))
    scorers = [
        PopularityScorer(vc, train_counts=counts),
        RepeatScorer(len(vc), popularity=counts),
        ItemKNNScorer(len(vc)).fit(train_seqs.items, train_seqs.times),
    ]

    results = []
    for scorer in scorers:
        t0 = time.time()
        results.append(protocol.evaluate(scorer, instances, batch_size=args.batch_size))
        print(f"scored {scorer.name:<12} {time.time() - t0:>8.1f}s")

    context = {
        "corpus": args.data,
        "events": len(events),
        "positives": len(positives),
        "vocab_items": vc.n_items,
        "repeat_rate": round(sessions.repeat_rate(all_seqs), 4),
        "cutoff_ts": split.cutoff_ts,
        "test_frac": args.test_frac,
        "holdout_user_frac": args.holdout_user_frac,
        "min_count": args.min_count,
        "ranking": "full catalog, pessimistic ties",
        "runtime_s": round(time.time() - started, 1),
    }
    paths = report.write(results, args.outdir, stem=f"baselines-{args.data}", context=context)

    print()
    print(report.format_markdown(results))
    print(f"written: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
