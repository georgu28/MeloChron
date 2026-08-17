"""Phase 6: derive drift, session archetypes and attention from a checkpoint.

    python scripts/build_insights.py --data synthetic --checkpoint artifacts/runs/smoke/best.pt
    python scripts/build_insights.py --data parquet --path data/interim/lastfm1k-v1.parquet \
        --checkpoint artifacts/runs/id-real/best.pt

Every number here is read out of an already-trained model; nothing is fitted
except the KMeans over session vectors, which is a description of the learned
representation rather than a second model. That is the whole point of the phase:
the product surface has to ride on the measurable part, not replace it.

Two wiring choices matter:

* **The vocabulary comes from the checkpoint, not from this frame.** Item ids
  are only meaningful against the vocabulary the model was trained on. Rebuilding
  it here would silently renumber every item and produce insights about the
  wrong tracks --- which would not crash and would not look wrong.
* **Item vectors are pulled once and converted to numpy at the boundary.** For a
  projected text representation ``item_vectors()`` is a full matrix product, and
  the drift and archetype modules are numpy by design.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from melochron import schema
from melochron.data import sessions, synthetic, vocab
from melochron.insights import archetypes, attention, drift
from melochron.train import checkpoint


def load_events(args):
    if args.data == "synthetic":
        events, _ = synthetic.generate(
            synthetic.SyntheticConfig(n_users=args.users or 40, seed=args.seed)
        )
        return events
    if args.data == "parquet":
        events = pd.read_parquet(args.path)
        if args.users:
            keep = sorted(events[schema.USER].unique())[: args.users]
            events = events[events[schema.USER].isin(keep)].reset_index(drop=True)
        return events
    raise ValueError(f"unknown corpus {args.data!r}")


def write_json(path: Path, context: dict, rows: list[dict], summary: dict) -> None:
    """Same shape as ``eval.report.write``'s JSON: context first, then rows.

    A payload without its context is not reproducible, which is why the context
    is not optional here even though nothing enforces it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"context": context, "summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )


def sample_histories(seqs, max_len: int, n_users: int, seed: int = 0):
    """One held-back history per user: everything except the final play.

    The last event is withheld so the attention row being read is genuinely
    predicting something rather than looking at the answer.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(seqs.user_ids))[:n_users]
    users, histories, times = [], [], []
    for i in order:
        items = seqs.items[i][:-1][-max_len:]
        stamps = seqs.times[i][:-1][-max_len:]
        if len(items) < 2:
            continue
        users.append(seqs.user_ids[i])
        histories.append(items)
        times.append(stamps)
    return users, histories, times


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="trained artifact (best.pt)")
    ap.add_argument("--data", choices=["synthetic", "parquet"], default="synthetic")
    ap.add_argument("--path", type=Path, help="parquet cache")
    ap.add_argument("--users", type=int, default=None, help="cap on distinct users")
    ap.add_argument("--out", type=Path, default=Path("artifacts/insights"))
    ap.add_argument("--name", type=str, default=None, help="run directory name")
    ap.add_argument("--window-days", type=int, default=drift.DEFAULT_WINDOW_DAYS)
    ap.add_argument("--min-events", type=int, default=drift.DEFAULT_MIN_EVENTS)
    ap.add_argument("--min-session-len", type=int, default=archetypes.DEFAULT_MIN_SESSION_LEN)
    ap.add_argument("--max-sessions", type=int, default=archetypes.DEFAULT_MAX_SESSIONS)
    ap.add_argument("--k-min", type=int, default=archetypes.DEFAULT_K_RANGE[0])
    ap.add_argument("--k-max", type=int, default=archetypes.DEFAULT_K_RANGE[1])
    ap.add_argument("--attention-users", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=attention.DEFAULT_TOP_K)
    ap.add_argument("--min-ms", type=int, default=sessions.DEFAULT_MIN_MS)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.data == "parquet" and args.path is None:
        ap.error("--path is required for --data parquet")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    started = time.time()

    artifact = checkpoint.load(args.checkpoint, device=args.device)
    vc = artifact.vocab
    print(f"checkpoint            {args.checkpoint!s:>12}")
    print(f"variant               {artifact.config.get('variant', 'id'):>12}")
    print(f"vocabulary            {vc.n_items:>12,} items (+2 reserved)")

    events = load_events(args)
    positives = sessions.filter_positives(events, min_ms=args.min_ms)
    positives = vocab.add_item_keys(positives)
    seqs = sessions.build_sequences(positives, vc)
    print(f"events                {len(positives):>12,}")
    print(f"users                 {len(seqs):>12,}")
    print(f"coverage              {vc.coverage(positives['item_key']):>12.1%}")

    item_vectors = artifact.head.items.item_vectors().detach().cpu().numpy()

    context = {
        "corpus": str(args.path) if args.data == "parquet" else "synthetic",
        "checkpoint": str(args.checkpoint),
        "variant": artifact.config.get("variant", "id"),
        "events": len(positives),
        "users": len(seqs),
        "vocab_items": vc.n_items,
        "window_days": args.window_days,
        "seed": args.seed,
    }

    outdir = args.out / (args.name or args.checkpoint.parent.name)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- drift
    timeline = drift.compute(
        seqs,
        item_vectors,
        vocab=vc,
        window_days=args.window_days,
        min_events=args.min_events,
    )
    drift_summary = timeline.summary()
    write_json(outdir / "drift.json", context, timeline.as_rows(), drift_summary)
    print(f"drift windows         {drift_summary['windows']:>12,}")
    print(f"  mean step           {drift_summary['mean_step']:>12.4f}")
    print(f"  mean displacement   {drift_summary['mean_displacement']:>12.4f}")

    # --- archetypes
    table = archetypes.build_sessions(
        seqs,
        item_vectors,
        min_session_len=args.min_session_len,
        max_sessions=args.max_sessions,
        seed=args.seed,
    )
    result = archetypes.compute(table, vocab=vc, k_range=(args.k_min, args.k_max), seed=args.seed)
    arch_summary = result.summary()
    write_json(outdir / "archetypes.json", context, result.as_rows(), arch_summary)
    print(f"sessions clustered    {arch_summary['n_sessions_clustered']:>12,}")
    print(f"  k                   {arch_summary['k']:>12}")
    print(f"  silhouette          {arch_summary['best_silhouette']:>12.4f}")

    # --- attention
    users, histories, times = sample_histories(
        seqs, artifact.model.max_len, args.attention_users, seed=args.seed
    )
    traces = attention.trace(
        artifact.model,
        histories,
        times,
        user_ids=users,
        vocab=vc,
        top_k=args.top_k,
        device=args.device,
        use_time=artifact.config.get("use_time", True),
    )
    attn_summary = attention.summarize(traces)
    write_json(outdir / "attention.json", context, [t.as_row() for t in traces], attn_summary)
    print(f"attention traces      {attn_summary['traces']:>12,}")
    for block, value in attn_summary.get("mean_recency_mass", {}).items():
        print(f"  block {block} recency mass{value:>12.4f}")

    context["runtime_s"] = round(time.time() - started, 1)
    lines = [
        "# Phase 6 insights",
        "",
        "## Run context",
        "",
        *[f"- **{k}**: {v}" for k, v in context.items()],
        "",
        "## Taste drift",
        "",
        *[f"- **{k}**: {v}" for k, v in drift_summary.items()],
        "",
        "## Session archetypes",
        "",
        *[f"- **{k}**: {v}" for k, v in arch_summary.items()],
        "",
        "## Attention",
        "",
        *[f"- **{k}**: {v}" for k, v in attn_summary.items()],
        "",
    ]
    (outdir / "insights.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote                 {outdir!s:>12}")
    print(f"runtime_s             {context['runtime_s']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
