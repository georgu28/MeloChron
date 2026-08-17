"""Parse a raw corpus once and cache it as parquet.

    python scripts/build_dataset.py --data lastfm1k --path data/raw/lastfm-1k

The lastfm-1K TSV is 2.4 GB and ~19M rows. Parsing it is deterministic and
takes on the order of a minute, which is small once but wasteful when every
baseline run, training run and ablation row pays it again. Downstream scripts
read the parquet.

The cache key includes the parser version, so a change to canonicalization or
to the positive filter invalidates it rather than silently serving a frame
built by different rules.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from melochron import schema
from melochron.data import lastfm, sessions, spotify_export, vocab

#: Bump when parsing, canonicalization or positive filtering changes meaning.
PARSER_VERSION = 1

DEFAULT_OUT = Path("data/interim")


def summarize(df: pd.DataFrame) -> dict:
    ts = pd.to_datetime(df[schema.TS], unit="s", utc=True)
    # No-op when item_key is already attached, which it is by the time this runs.
    keyed = vocab.add_item_keys(df)
    return {
        "events": len(df),
        "users": int(df[schema.USER].nunique()),
        "distinct_items": int(keyed["item_key"].nunique()),
        "distinct_artists": int(df[schema.ARTIST].nunique()),
        "first_play": str(ts.min()),
        "last_play": str(ts.max()),
        "has_ms_played": bool(df[schema.MS_PLAYED].notna().any()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", choices=["lastfm1k", "spotify"], default="lastfm1k")
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--users", type=int, default=None, help="cap on distinct users")
    ap.add_argument("--limit", type=int, default=None, help="cap on raw rows")
    ap.add_argument("--min-ms", type=int, default=sessions.DEFAULT_MIN_MS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    suffix = f"-u{args.users}" if args.users else ""
    out = args.out / f"{args.data}{suffix}-v{PARSER_VERSION}.parquet"
    meta_path = out.with_suffix(".json")

    # Both files, not just the parquet. A parquet without its metadata means a
    # previous run died between the two writes, and treating that as a cache
    # hit would serve a frame whose provenance is unrecorded.
    if out.exists() and meta_path.exists() and not args.force:
        print(f"cache hit: {out}")
        print(json.loads(meta_path.read_text(encoding="utf-8"))["summary"])
        print("pass --force to rebuild")
        return 0
    if out.exists() and not meta_path.exists():
        print(f"found {out.name} without metadata; rebuilding")

    started = time.time()
    print(f"parsing {args.path} ...")

    if args.data == "lastfm1k":
        events = lastfm.read_lastfm1k(args.path, users=args.users, limit=args.limit)
    else:
        events = spotify_export.read_export(args.path)

    parsed_at = time.time()
    print(f"  {len(events):,} events in {parsed_at - started:.1f}s")

    positives = sessions.filter_positives(events, min_ms=args.min_ms)
    kept = len(positives) / len(events) if len(events) else 0.0
    print(f"  {len(positives):,} positives ({kept:.1%} kept)")

    # Sessionize here so every consumer sees the same session boundaries, and
    # so the cost is paid once alongside the parse.
    positives = sessions.sessionize(positives)
    positives = vocab.add_item_keys(positives)

    out.parent.mkdir(parents=True, exist_ok=True)
    positives.to_parquet(out, index=False)

    summary = summarize(positives)
    meta_path.write_text(
        json.dumps(
            {
                "parser_version": PARSER_VERSION,
                "source": args.data,
                "source_path": str(args.path),
                "min_ms": args.min_ms,
                "users_cap": args.users,
                "row_limit": args.limit,
                "build_seconds": round(time.time() - started, 1),
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - started:.1f}s")
    for k, v in summary.items():
        print(f"  {k:<18} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
