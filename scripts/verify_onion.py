"""Check the compact store against the raw archive, user by user.

    python scripts/verify_onion.py --users 5

The counts in the Phase 0 report are aggregates, and aggregates can all agree
while the store is still wrong: a stable sort that is not stable, a batch
boundary that drops a row, an id remapped to the wrong code. Every one of those
preserves the total.

So this re-reads the raw archive once, pulls every row belonging to a few
sampled users straight out of the decompressed text, and asserts that the store
holds exactly those (track, timestamp) pairs in exactly time order. It is the
one check that looks at events rather than at counts.

Costs one full decompression pass, a few minutes. Worth running once after a
build, not on every change.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from melochron.adoption.corpus import CompactCorpus
from melochron.adoption.onion import EVENTS_FILE, read_batches, sniff_schema

DEFAULT_PATH = Path("data/raw/music4all-onion")
DEFAULT_STORE = Path("data/interim/onion-v1")


def to_epoch(stamps: list[str]) -> list[int]:
    """Parse raw timestamps exactly the way the build does, in one batch.

    Going through the same Arrow cast is the point: a check that parsed dates
    its own way would be testing two parsers against each other rather than
    testing the store.
    """
    return pa.array(stamps).cast(pa.timestamp("s")).cast(pa.int64()).to_pylist()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--users", type=int, default=5, help="how many users to check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    events = args.path / EVENTS_FILE
    corpus = CompactCorpus.load(args.store, mmap=True)
    print(f"store: {corpus.n_events:,} events, {corpus.n_users:,} users")

    # Sample across the code space rather than taking the first few: codes are
    # assigned in first-appearance order, so the low codes are all users who
    # happen to sit at the head of the file.
    rng = np.random.default_rng(args.seed)
    codes = rng.choice(corpus.n_users, size=min(args.users, corpus.n_users), replace=False)
    wanted = {str(corpus.users[c]): int(c) for c in codes}
    print(f"checking users: {', '.join(sorted(wanted))}\n")

    schema = sniff_schema(events)
    raw: dict[str, list[tuple[str, str]]] = defaultdict(list)

    print("re-reading the archive (one decompression pass)...")
    value_set = pa.array(sorted(wanted))
    for batch in read_batches(events, schema):
        # Filter in Arrow. Converting every batch to Python lists to test
        # membership would allocate 253M string objects and take longer than
        # the build it is checking.
        mask = pc.is_in(batch.column(schema.user_idx), value_set=value_set)
        kept = batch.filter(mask)
        if kept.num_rows == 0:
            continue
        for user, track, stamp in zip(
            kept.column(schema.user_idx).to_pylist(),
            kept.column(schema.track_idx).to_pylist(),
            kept.column(schema.ts_idx).to_pylist(),
            strict=True,
        ):
            raw[user].append((track, stamp))

    failures = 0
    for user, code in sorted(wanted.items()):
        window = corpus.events_for(code)
        stored = [
            (str(corpus.tracks[t]), int(ts))
            for t, ts in zip(corpus.track_code[window], corpus.ts[window], strict=True)
        ]

        # The raw file is not in time order, so the comparison sorts both sides
        # by the same key. Ties on an identical timestamp are broken by track so
        # that a genuine duplicate still has to appear on both sides.
        expected = sorted(
            zip(
                [track for track, _ in raw[user]],
                to_epoch([stamp for _, stamp in raw[user]]),
                strict=True,
            ),
            key=lambda row: (row[1], row[0]),
        )
        got = sorted(stored, key=lambda row: (row[1], row[0]))

        times = [ts for _, ts in stored]
        ordered = all(a <= b for a, b in pairwise(times))
        ok = expected == got and ordered

        status = "ok" if ok else "FAIL"
        print(
            f"  {user}: {len(stored):,} events in store, {len(raw[user]):,} in archive  [{status}]"
        )
        if not ok:
            failures += 1
            if not ordered:
                print("    stored events are not in ascending time order")
            missing = set(expected) - set(got)
            extra = set(got) - set(expected)
            if missing:
                print(
                    f"    {len(missing)} events in the archive but not the store, e.g. {list(missing)[:3]}"
                )
            if extra:
                print(
                    f"    {len(extra)} events in the store but not the archive, e.g. {list(extra)[:3]}"
                )

    if failures:
        print(f"\n{failures} of {len(wanted)} users do not match")
        return 1
    print(f"\nall {len(wanted)} users match the archive exactly, in time order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
