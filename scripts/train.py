"""Train a SASRec variant and score it against the baselines.

    python scripts/train.py --config configs/pretrain.yaml
    python scripts/train.py --data synthetic --epochs 2        # smoke test

Three temporal regions, cut in this order, and the order is the point:

    [-------- fit --------|--- val ---|--- test ---]
                          ^           ^
                          inner cut   outer cut

The **outer** cut holds out the test period and, independently, a set of users
withheld from training entirely (the cold-start axis). The **inner** cut then
carves a validation period out of what remains, so early stopping and
checkpoint selection never touch the test period. Selecting a checkpoint on
test data leaks in a way that is invisible afterwards: the model looks like it
generalizes precisely because it was chosen for looking that way.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import yaml

from melochron.baselines.itemknn import ItemKNNScorer
from melochron.baselines.popularity import PopularityScorer, counts_from_sequences
from melochron.baselines.repeat import RepeatScorer
from melochron.data import sessions, splits, synthetic, vocab
from melochron.eval import protocol, report
from melochron.models.scorer import build_scorer
from melochron.train import transfer
from melochron.train.loop import TrainConfig, Trainer


def load_events(args) -> pd.DataFrame:
    if args.data == "synthetic":
        events, _ = synthetic.generate(
            synthetic.SyntheticConfig(n_users=args.users or 40, seed=args.seed)
        )
        return sessions.filter_positives(events)
    if not args.path:
        raise SystemExit("--path is required unless --data synthetic")
    return pd.read_parquet(args.path)


def build_config(args) -> TrainConfig:
    values = {}
    if args.config:
        values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    known = set(TrainConfig.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")

    for key in ("variant", "epochs", "batch_size", "max_len", "d_model", "loss"):
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    if args.no_time:
        values["use_time"] = False
    values.setdefault("seed", args.seed)
    return TrainConfig(**values)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path)
    ap.add_argument("--data", choices=["parquet", "synthetic"], default="parquet")
    ap.add_argument("--path", type=Path, help="parquet from scripts/build_dataset.py")
    ap.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--users", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--holdout-user-frac", type=float, default=0.10)
    ap.add_argument("--max-per-user", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-baselines", action="store_true")
    # Config overrides
    ap.add_argument("--variant", choices=["id", "text_frozen", "text_finetuned"])
    ap.add_argument("--text-vectors", type=Path, help=".npy from scripts/build_embeddings.py")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--d-model", type=int, dest="d_model")
    ap.add_argument("--loss", choices=["sampled_softmax", "bpr"])
    ap.add_argument("--no-time", action="store_true", help="position-only ablation")
    ap.add_argument("--init-from", type=Path, help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--freeze-encoder", action="store_true", help="adapt the item table only")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    started = time.time()

    positives = load_events(args)
    print(f"events                {len(positives):>12,}")

    outer = splits.temporal_split(
        positives,
        test_frac=args.test_frac,
        holdout_user_frac=args.holdout_user_frac,
        seed=args.seed,
    )
    splits.assert_no_leakage(outer)

    inner = splits.temporal_split(
        outer.train, test_frac=args.val_frac, holdout_user_frac=0.0, seed=args.seed
    )
    splits.assert_no_leakage(inner)
    print(f"fit / val / test      {len(inner.train):,} / {len(inner.test):,} / {len(outer.test):,}")

    # Catalog scope: the full frame. Counts and fit stay train-only.
    vc = vocab.build_vocab(positives, min_count=args.min_count)
    print(f"vocabulary            {vc.n_items:>12,} items (+2 reserved)")

    all_seqs = sessions.build_sequences(positives, vc)
    train_period_seqs = sessions.build_sequences(outer.train, vc)
    fit_seqs = sessions.build_sequences(inner.train, vc)
    fit_items = {int(i) for arr in fit_seqs.items for i in arr.tolist()}
    train_items = {int(i) for arr in train_period_seqs.items for i in arr.tolist()}
    print(f"repeat rate           {sessions.repeat_rate(all_seqs):>12.1%}")

    val_instances = protocol.build_instances(
        train_period_seqs,
        cutoff_ts=inner.cutoff_ts,
        train_items=fit_items,
        holdout_users=frozenset(),
        max_len=cfg.max_len,
        max_per_user=args.max_per_user,
        seed=args.seed,
    )
    test_instances = protocol.build_instances(
        all_seqs,
        cutoff_ts=outer.cutoff_ts,
        train_items=train_items,
        holdout_users=outer.holdout_users,
        max_len=cfg.max_len,
        max_per_user=args.max_per_user,
        seed=args.seed,
    )
    print(f"val instances         {len(val_instances):>12,}")
    print(f"test instances        {test_instances.summary()}")

    text_vectors = None
    if cfg.variant != "id":
        if not args.text_vectors:
            raise SystemExit(f"variant {cfg.variant!r} needs --text-vectors")
        import numpy as np

        text_vectors = torch.from_numpy(np.load(args.text_vectors)).float()
        if len(text_vectors) != len(vc):
            raise SystemExit(
                f"text vectors have {len(text_vectors)} rows but vocabulary has {len(vc)}; "
                "rebuild embeddings against this vocabulary"
            )

    scorer_name = f"sasrec-{cfg.variant}" + ("" if cfg.use_time else "-notime")

    if args.init_from:
        # Fine-tuning: start from an encoder pretrained on a different catalog
        # rather than from noise. Only the text variants can do this, and
        # load_for_catalog raises rather than degrading if asked to try.
        if text_vectors is None:
            raise SystemExit("--init-from needs --text-vectors for the new catalog")
        transplant = transfer.load_for_catalog(
            str(args.init_from),
            text_vectors,
            vc,
            device=args.device,
            name=scorer_name,
            freeze_encoder=args.freeze_encoder,
        )
        model, head, scorer = transplant.model, transplant.head, transplant.scorer
        print(f"initialized from {args.init_from} | {transplant.card()}")
        # The architecture is the checkpoint's, not the config's; saying so
        # avoids a report that claims hyperparameters the run did not use.
        cfg.d_model = transplant.config["d_model"]
        cfg.n_heads = transplant.config["n_heads"]
        cfg.n_blocks = transplant.config["n_blocks"]
        cfg.max_len = transplant.config["max_len"]
    else:
        model, head, scorer = build_scorer(
            n_items=len(vc),
            device=args.device,
            name=scorer_name,
            variant=cfg.variant,
            text_vectors=text_vectors,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_blocks=cfg.n_blocks,
            max_len=cfg.max_len,
            dropout=cfg.dropout,
            use_time=cfg.use_time,
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"device {args.device} | variant {cfg.variant} | use_time {cfg.use_time} | "
        f"{n_params / 1e6:.2f}M trainable params"
    )

    counts = counts_from_sequences(fit_seqs.items, len(vc))
    run_name = args.name or f"{cfg.variant}{'' if cfg.use_time else '-notime'}"
    out_dir = args.out / run_name

    trainer = Trainer(model, head, scorer, vc, cfg, device=args.device, train_counts=counts)
    state = trainer.fit(fit_seqs, val_instances, out_dir=out_dir)

    # Reload the selected checkpoint: the in-memory model is whatever the last
    # epoch left behind, which is not the one early stopping chose.
    from melochron.train import checkpoint as ckpt

    best = ckpt.load(out_dir / "best.pt", device=args.device, name=scorer.name)
    print(f"\nevaluating best checkpoint (epoch {state.best_epoch}) on the test period")

    results = [protocol.evaluate(best.scorer, test_instances, batch_size=cfg.eval_batch_size)]

    if not args.skip_baselines:
        test_counts = counts_from_sequences(train_period_seqs.items, len(vc))
        for baseline in (
            PopularityScorer(vc, train_counts=test_counts),
            RepeatScorer(len(vc), popularity=test_counts),
            ItemKNNScorer(len(vc)).fit(train_period_seqs.items, train_period_seqs.times),
        ):
            results.append(protocol.evaluate(baseline, test_instances, batch_size=256))

    context = {
        "corpus": args.data if args.data == "synthetic" else str(args.path),
        "events": len(positives),
        "vocab_items": vc.n_items,
        "repeat_rate": round(sessions.repeat_rate(all_seqs), 4),
        "min_count": args.min_count,
        "best_epoch": state.best_epoch,
        "best_val": round(state.best_metric, 5),
        "ranking": "full catalog, pessimistic ties",
        "runtime_s": round(time.time() - started, 1),
        **{f"cfg.{k}": v for k, v in asdict(cfg).items()},
    }
    paths = report.write(results, out_dir, stem="results", context=context)
    (out_dir / "history.json").write_text(json.dumps(state.history, indent=2), encoding="utf-8")

    print()
    print(report.format_markdown(results))
    print(f"written: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
