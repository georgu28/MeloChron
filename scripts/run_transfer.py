"""The transfer table: zero-shot, fine-tuned and from-scratch on one catalog.

    python scripts/run_transfer.py --path data/interim/spotify-v1.parquet \
        --text-vectors data/embeddings/text-a2153cdb8d2b123a.npy \
        --pretrained zero-shot-text=artifacts/runs/text-tagged/best.pt \
        --pretrained zero-shot-hybrid=artifacts/runs/hybrid-norm/best.pt \
        --checkpoint scratch-id=artifacts/runs/personal-id/best.pt \
        --min-count 1 --max-per-user 20000

This is the experiment the transfer design has been promising since Phase 2 and
could not run until a second, genuinely disjoint catalog existed. The personal
export overlaps the pretraining corpus by about 6% of items, so most targets
here are tracks the pretrained model has never seen --- which is the condition
under which text representations are supposed to matter and ID embeddings are
supposed to fail.

**Every row is scored on the same instances.** The instances are built once,
before any model is loaded, and handed to each scorer in turn. Building them
per model is how two rows end up answering slightly different questions while
looking like a comparison.

The zero-shot row is not a checkpoint. It is the pretrained encoder re-pointed
at this catalog by ``train/transfer.py``, with nothing fitted on this user at
all --- no gradient step, no per-user parameter. That is the honest meaning of
"cold start" for a model with no user embeddings.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from melochron.baselines.popularity import PopularityScorer, counts_from_sequences
from melochron.baselines.repeat import RepeatScorer
from melochron.data import sessions, splits, vocab
from melochron.eval import protocol, report
from melochron.train import checkpoint, transfer


def _named(pair: str) -> tuple[str, Path]:
    if "=" not in pair:
        raise argparse.ArgumentTypeError(f"expected name=path, got {pair!r}")
    name, _, path = pair.partition("=")
    return name, Path(path)


def _check_vocab(loaded, expected: vocab.Vocab, label: str) -> None:
    """A checkpoint scored against a different vocabulary is silently wrong.

    Ids are positional. If the checkpoint's table and this run's vocabulary
    disagree by a single item, every id after it refers to a different track and
    the model ranks a catalog whose rows mean something else. Nothing crashes
    and the numbers look ordinary, so it is worth an explicit check.
    """
    if len(loaded) != len(expected):
        raise SystemExit(
            f"{label}: checkpoint vocabulary has {len(loaded):,} rows but this run built "
            f"{len(expected):,}; rebuild with the same --min-count and corpus"
        )
    if loaded.id_to_key != expected.id_to_key:
        raise SystemExit(f"{label}: checkpoint vocabulary differs from this run's ordering")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, required=True, help="parquet from build_dataset.py")
    ap.add_argument("--text-vectors", type=Path, help=".npy aligned to this run's vocabulary")
    ap.add_argument(
        "--pretrained",
        type=_named,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="checkpoint to transplant for a zero-shot row, repeatable",
    )
    ap.add_argument(
        "--checkpoint",
        type=_named,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a trained checkpoint to score, repeatable",
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/transfer"))
    ap.add_argument("--stem", default="transfer")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--holdout-user-frac", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--max-per-user", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    started = time.time()

    events = pd.read_parquet(args.path)
    positives = sessions.filter_positives(events)
    positives = vocab.add_item_keys(positives)

    # Vocabulary from the full frame; the fit and the counts from train only.
    # Building it from train would make every in-vocabulary target train-seen by
    # construction and empty the cold-item slice, which is the whole experiment.
    vc = vocab.build_vocab(positives, min_count=args.min_count)
    split = splits.temporal_split(
        positives,
        test_frac=args.test_frac,
        holdout_user_frac=args.holdout_user_frac,
        seed=args.seed,
    )
    splits.assert_no_leakage(split)

    all_seqs = sessions.build_sequences(positives, vc)
    train_seqs = sessions.build_sequences(split.train, vc)
    train_items = {int(i) for s in train_seqs.items for i in np.unique(s)}

    print(f"events                {len(positives):>12,}")
    print(f"users                 {len(all_seqs):>12,}")
    print(f"vocabulary            {vc.n_items:>12,} items (+2 reserved)")
    print(f"repeat rate           {sessions.repeat_rate(all_seqs):>12.1%}")

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
        raise SystemExit("no evaluation instances; nothing to score")

    counts = counts_from_sequences(train_seqs.items, len(vc))
    scorers = [
        RepeatScorer(len(vc), popularity=counts),
        PopularityScorer(vc, train_counts=counts),
    ]

    if args.pretrained:
        if not args.text_vectors:
            raise SystemExit("--pretrained needs --text-vectors for this catalog")
        text = torch.from_numpy(np.load(args.text_vectors)).float()
        # Repeatable, because comparing two pretrained encoders zero-shot on the
        # same catalog is a transfer question and this is the transfer table.
        # Scoring them in one invocation is also the only way to guarantee they
        # meet the same instances rather than two runs that merely used the same
        # flags.
        for name, path in args.pretrained:
            if not path.exists():
                raise SystemExit(f"pretrained checkpoint not found: {path}")
            transplant = transfer.load_for_catalog(
                str(path), text, vc, device=args.device, name=name
            )
            print(f"{name:<21} {transplant.card()}")
            scorers.append(transplant.scorer)

    for name, path in args.checkpoint:
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        loaded = checkpoint.load(path, device=args.device, name=name)
        _check_vocab(loaded.vocab, vc, name)
        print(
            f"{name:<21} variant={loaded.config.get('variant')} epoch={loaded.metrics.get('epoch')}"
        )
        scorers.append(loaded.scorer)

    results = []
    for scorer in scorers:
        t0 = time.time()
        results.append(protocol.evaluate(scorer, instances, batch_size=args.batch_size))
        print(f"scored {scorer.name:<20} {time.time() - t0:>8.1f}s")

    notes = {}
    if len(all_seqs.user_ids) == 1:
        notes["novel"] = (
            "Single-user corpus: novel and cold_start are the same instances here, and "
            "popularity and item-kNN have no collaborative signal, so their zeros are "
            "structural. An ID-embedding model cannot score this slice at all -- those "
            "targets have no trained row -- which is what the text variants exist for."
        )
        notes["cold_item"] = notes["novel"]
        notes["cold_start"] = notes["novel"]

    context = {
        "corpus": str(args.path),
        "events": len(positives),
        "users": len(all_seqs),
        "vocab_items": vc.n_items,
        "min_count": args.min_count,
        "repeat_rate": round(sessions.repeat_rate(all_seqs), 4),
        "cutoff_ts": split.cutoff_ts,
        "max_len": args.max_len,
        "pretrained": {name: str(path) for name, path in args.pretrained} or None,
        "ranking": "full catalog, pessimistic ties",
        "chance_hr10": round(10 / vc.n_items, 6),
        "runtime_s": round(time.time() - started, 1),
    }
    paths = report.write(
        results, args.out, stem=args.stem, context=context, n_items=vc.n_items, notes=notes
    )

    print()
    print(report.format_markdown(results, n_items=vc.n_items, notes=notes))
    print(f"written: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
