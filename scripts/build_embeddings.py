"""Build the text-embedding matrix for a vocabulary.

    # names only, no API key needed
    python scripts/build_embeddings.py --path data/interim/lastfm1k-v1.parquet

    # fetch Last.fm tags first (resumable, hours for a large vocabulary)
    python scripts/build_embeddings.py --path ... --fetch-tags

Writes ``[n_items, 384] float32`` aligned to vocabulary ids, which is what
``--text-vectors`` on scripts/train.py expects.

The vocabulary must be built with the **same** ``--min-count`` as the training
run. Row *i* means vocabulary item *i*, and a matrix built against a different
vocabulary gives every item someone else's semantics: no crash, no obviously
wrong metric, just a model that quietly underperforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from melochron.data import sessions, synthetic, vocab
from melochron.data.vocab import FIRST_ITEM_ID
from melochron.features import embed
from melochron.features import tags as tags_mod
from melochron.features import text as text_mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, help="parquet from scripts/build_dataset.py")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--model", default=embed.DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fetch-tags", action="store_true")
    ap.add_argument(
        "--tag-mode",
        choices=["artist", "track"],
        default="artist",
        help="artist: ~8.5x fewer requests for the same item coverage, coarser tags",
    )
    ap.add_argument("--tag-limit", type=int, default=None, help="cap lookups this run")
    ap.add_argument("--tag-cache", type=Path, default=tags_mod.DEFAULT_CACHE)
    ap.add_argument("--artist-cache", type=Path, default=tags_mod.DEFAULT_ARTIST_CACHE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.synthetic:
        events, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=20, seed=0))
        events = sessions.filter_positives(events)
    elif args.path:
        events = pd.read_parquet(args.path)
    else:
        raise SystemExit("pass --path or --synthetic")

    vc = vocab.build_vocab(events, min_count=args.min_count)
    print(f"vocabulary  {vc.n_items:,} items (min_count={args.min_count})")

    track_cache = tags_mod.TagCache(args.tag_cache)
    artist_cache = tags_mod.TagCache(args.artist_cache)

    if args.fetch_tags and args.tag_mode == "artist":
        artists = [vc.display[i][0] for i in range(FIRST_ITEM_ID, len(vc))]
        artist_cache = tags_mod.fetch_artist_tags(
            artists, cache=artist_cache, limit=args.tag_limit, verbose=True
        )
    elif args.fetch_tags:
        items = [
            (vc.id_to_key[i], vc.display[i][0], vc.display[i][1])
            for i in range(FIRST_ITEM_ID, len(vc))
        ]
        track_cache = tags_mod.fetch_for_items(
            items, cache=track_cache, limit=args.tag_limit, verbose=True
        )

    # Track-level tags win where present, so a cheap artist pass can later be
    # refined for the most-played tracks without refetching anything.
    tag_map = tags_mod.tags_for_vocab(vc, artist_cache=artist_cache, track_cache=track_cache)
    coverage = text_mod.tag_coverage(vc, tag_map)
    print(
        f"tag coverage {coverage:.1%} ({len(tag_map):,} items tagged; "
        f"{len(artist_cache):,} artists / {len(track_cache):,} tracks in cache)"
    )

    # Report it, because a 'text' variant built on 5% tag coverage is a
    # names-only variant wearing a different label, and the ablation table
    # would be quietly misleading without the number next to it.
    template = text_mod.DEFAULT_TEMPLATE

    vectors, meta = embed.build_matrix(
        vc,
        tags=tag_map,
        model_name=args.model,
        template=template,
        cache_dir=args.out,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
    )

    report = embed.coverage_report(vectors)
    print(json.dumps({**meta, **report}, indent=2))

    if not report["reserved_rows_zero"]:
        raise SystemExit("PAD/OOV rows are not zero; refusing to write a scorable reserved slot")

    out = Path(args.out) / f"text-{meta['cache_key']}.npy"
    print(f"\nvectors: {out}")
    print(f"train with: --variant text_frozen --text-vectors {out} --min-count {args.min_count}")
    np.save(Path(args.out) / "latest.npy", vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
