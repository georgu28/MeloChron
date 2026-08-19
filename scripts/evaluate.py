"""Score a saved checkpoint on the test period, optionally beside the baselines.

    python scripts/evaluate.py --checkpoint artifacts/runs/id-real/best.pt \
        --path data/interim/lastfm1k-v1.parquet --min-count 20

Separate from ``scripts/train.py`` for two reasons. Training runs are long and
interruptible, and losing the final evaluation because a run was killed after
the checkpoint was written is a waste of an hour. And the ablation table needs
several checkpoints scored under identical conditions, which is easier to
guarantee when scoring is one command that takes a checkpoint than when it is
a side effect of training.

The vocabulary is read **from the checkpoint**, not rebuilt. Rebuilding would
usually produce the same thing, and "usually" is the problem: a vocabulary that
differs by one item shifts every id after it, so the model would score a
catalog whose rows mean something else. Nothing would crash.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from melochron.baselines.itemknn import ItemKNNScorer
from melochron.baselines.popularity import PopularityScorer, counts_from_sequences
from melochron.baselines.repeat import RepeatScorer
from melochron.data import sessions, splits, synthetic
from melochron.eval import protocol, report
from melochron.train import checkpoint as ckpt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--path", type=Path, help="parquet from scripts/build_dataset.py")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="defaults beside the checkpoint")
    ap.add_argument("--stem", default="results")
    ap.add_argument("--name", default=None, help="model label in the table")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--holdout-user-frac", type=float, default=0.10)
    ap.add_argument("--max-per-user", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-baselines", action="store_true")
    args = ap.parse_args(argv)

    started = time.time()

    artifact = ckpt.load(args.checkpoint, device=args.device, name=args.name or "sasrec")
    cfg = artifact.config
    vc = artifact.vocab
    print(f"checkpoint    {args.checkpoint}")
    print(f"variant       {cfg.get('variant')}  use_time={cfg.get('use_time')}")
    print(f"vocabulary    {vc.n_items:,} items (from checkpoint)")
    if artifact.metrics:
        print(f"saved at      epoch {artifact.metrics.get('epoch')}")

    if args.synthetic:
        events, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=40, seed=args.seed))
        events = sessions.filter_positives(events)
    elif args.path:
        events = pd.read_parquet(args.path)
    else:
        raise SystemExit("pass --path or --synthetic")

    # Split parameters must match the training run or the test period differs
    # and the numbers are not comparable to anything.
    outer = splits.temporal_split(
        events,
        test_frac=args.test_frac,
        holdout_user_frac=args.holdout_user_frac,
        seed=args.seed,
    )
    splits.assert_no_leakage(outer)

    all_seqs = sessions.build_sequences(events, vc)
    train_period_seqs = sessions.build_sequences(outer.train, vc)
    train_items = {int(i) for arr in train_period_seqs.items for i in arr.tolist()}

    instances = protocol.build_instances(
        all_seqs,
        cutoff_ts=outer.cutoff_ts,
        train_items=train_items,
        holdout_users=outer.holdout_users,
        max_len=cfg.get("max_len", 200),
        max_per_user=args.max_per_user,
        seed=args.seed,
    )
    print(f"instances     {instances.summary()}")

    label = args.name or f"sasrec-{cfg.get('variant', 'id')}"
    artifact.scorer.name = label
    results = [protocol.evaluate(artifact.scorer, instances, batch_size=args.batch_size)]
    print(f"scored {label}")

    if args.with_baselines:
        counts = counts_from_sequences(train_period_seqs.items, len(vc))
        for baseline in (
            PopularityScorer(vc, train_counts=counts),
            RepeatScorer(len(vc), popularity=counts),
            ItemKNNScorer(len(vc)).fit(train_period_seqs.items, train_period_seqs.times),
        ):
            results.append(protocol.evaluate(baseline, instances, batch_size=256))
            print(f"scored {baseline.name}")

    outdir = args.out or args.checkpoint.parent
    context = {
        "checkpoint": str(args.checkpoint),
        "variant": cfg.get("variant"),
        "use_time": cfg.get("use_time"),
        "vocab_items": vc.n_items,
        "events": len(events),
        "instances": len(instances),
        # Two checkpoints are only comparable if they were scored over the same
        # instances, and the flags below are what decide that. Recorded next to
        # the numbers so a later reader can check rather than assume.
        "test_frac": args.test_frac,
        "holdout_user_frac": args.holdout_user_frac,
        "max_per_user": args.max_per_user,
        "seed": args.seed,
        "ranking": "full catalog, pessimistic ties",
        "runtime_s": round(time.time() - started, 1),
    }
    paths = report.write(results, outdir, stem=args.stem, context=context)

    print()
    print(report.format_markdown(results))
    print(f"written: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
