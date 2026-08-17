"""Measure request-time inference latency against a real checkpoint.

    python scripts/bench_latency.py --checkpoint artifacts/runs/id-real/best.pt

Reports p50/p95/p99 at a stated history length and catalog size. Both of those
belong next to the numbers: latency here is dominated by scoring the full
catalog, so a figure quoted without the catalog size is not a claim anyone can
check or reproduce.

Measured on CPU by default, because that is what the deployed service runs on.
A GPU number would be faster and would not describe production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from melochron.data.vocab import FIRST_ITEM_ID
from melochron.serving.inference import Recommender, benchmark
from melochron.train import checkpoint as ckpt


def synthetic_histories(
    vocab, n: int, length: int, seed: int = 0
) -> list[list[tuple[str, str, int]]]:
    """Plausible request payloads drawn from the catalog.

    Latency depends on history length and catalog size, not on which specific
    tracks are in the history, so sampled items are a fair stand-in for real
    uploads here.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ids = rng.integers(FIRST_ITEM_ID, len(vocab), size=length)
        base = 1_600_000_000
        rows = []
        for i, item_id in enumerate(ids.tolist()):
            artist, track = vocab.display[item_id] if vocab.display else ("a", "t")
            rows.append((artist, track, base + i * 210))
        out.append(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cpu", help="cpu matches the deploy target")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--history-length", type=int, default=200)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    artifact = ckpt.load(args.checkpoint, device=args.device)
    recommender = Recommender(
        artifact.scorer, artifact.vocab, max_len=artifact.config.get("max_len", 200)
    )

    print(f"checkpoint      {args.checkpoint}")
    print(f"variant         {artifact.config.get('variant')}")
    print(f"catalog         {artifact.vocab.n_items:,} items")
    print(f"device          {args.device}")
    print(f"history length  {args.history_length}")

    histories = synthetic_histories(artifact.vocab, args.requests, args.history_length)
    result = benchmark(recommender, histories, k=args.k)
    result["device"] = args.device
    result["history_length"] = args.history_length
    result["variant"] = artifact.config.get("variant")

    print()
    print(json.dumps(result, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
