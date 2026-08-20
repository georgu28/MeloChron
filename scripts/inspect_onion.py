"""Reconcile the assumed Music4All-Onion schema against the real file.

    python scripts/inspect_onion.py

The project brief carries a schema inferred from the dataset paper, and the
dataset README documents row counts and nothing else -- no column names, no
header convention, no timestamp unit. So this reads the actual bytes at the head
of the events file and reports assumed-versus-actual. Where they differ, the
file wins.

Only the first few MB are decompressed, so this runs in seconds against a
2.2 GB archive and is safe to rerun.

Writes ``artifacts/adoption/phase0-schema.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pyarrow as pa

from melochron.adoption.onion import EVENTS_FILE, PUBLISHED, open_stream, sniff_schema

DEFAULT_PATH = Path("data/raw/music4all-onion")
DEFAULT_OUT = Path("artifacts/adoption/phase0-schema.md")

#: What the brief says, so the report can show what changed. These are claims
#: to be tested, not defaults to fall back on.
ASSUMED = {
    "columns": ["user_id", "track_id", "timestamp"],
    "rows": "~253M",
    "users": "~119K",
    "tracks": "~109K",
    "timestamp": "a sortable absolute time (unit unconfirmed)",
    "header": "unstated",
}


def probe_ordering(path, schema, probe_bytes: int = 1 << 22) -> dict:
    """Is the file already sorted, and by what?

    Worth knowing before the build: a file already grouped by user and ordered
    by time would let the sort be skipped entirely. Judged on a prefix, so a
    positive result here is a hint to verify, never a licence to skip the sort.
    """
    with open_stream(path) as stream:
        head = stream.read(probe_bytes)
    if hasattr(head, "to_pybytes"):
        head = head.to_pybytes()

    lines = head.decode("utf-8", errors="replace").splitlines()[:-1]
    if schema.has_header:
        lines = lines[1:]

    users, stamps = [], []
    for line in lines:
        fields = line.split(schema.delimiter)
        if len(fields) != len(schema.columns):
            continue
        users.append(fields[schema.user_idx])
        stamps.append(fields[schema.ts_idx])

    distinct_users = len(set(users))
    runs = 1 + sum(1 for a, b in pairwise(users) if a != b)
    ts_ascending = all(a <= b for a, b in pairwise(stamps))

    # Direction within each contiguous run of one user. Last.fm's API returns
    # most-recent-first, so a corpus scraped from it arrives descending; that
    # is worth recording because it is exactly the order the label builder must
    # not inherit.
    descending = ascending = 0
    start = 0
    for i in range(1, len(users) + 1):
        if i == len(users) or users[i] != users[start]:
            run = stamps[start:i]
            if len(run) > 1:
                if all(a >= b for a, b in pairwise(run)):
                    descending += 1
                elif all(a <= b for a, b in pairwise(run)):
                    ascending += 1
            start = i

    return {
        "rows_probed": len(users),
        "distinct_users_in_probe": distinct_users,
        "user_runs": runs,
        "grouped_by_user": runs == distinct_users,
        "ts_ascending_overall": ts_ascending,
        "runs_descending_in_time": descending,
        "runs_ascending_in_time": ascending,
    }


def render(schema, ordering: dict, sample: list[list[str]]) -> str:
    actual_cols = ", ".join(schema.columns)
    ts_example = sample[0][schema.ts_idx] if sample else "n/a"

    decoded = ""
    if schema.ts_kind in ("seconds", "millis"):
        raw = int(ts_example)
        seconds = raw // 1000 if schema.ts_kind == "millis" else raw
        decoded = datetime.fromtimestamp(seconds, UTC).isoformat()
        ts_described = f"unix {schema.ts_kind}"
    else:
        # "iso" is the parser's bucket for "not an integer", and the real format
        # is worth naming: it is space-separated rather than ISO-8601's T, and
        # it carries no zone.
        decoded = pa.array([ts_example]).cast(pa.timestamp("s")).cast(pa.int64())[0].as_py()
        decoded = datetime.fromtimestamp(decoded, UTC).isoformat()
        ts_described = f"datetime string, `{ts_example}`"

    lines = [
        "# Phase 0 — schema reconciliation",
        "",
        f"Source: `{EVENTS_FILE}`, Zenodo record 15394646.",
        "",
        "The dataset README states row counts and nothing about layout. Everything",
        "below is measured from the head of the file.",
        "",
        "| property | assumed (brief) | actual (file) |",
        "|---|---|---|",
        f"| columns | {', '.join(ASSUMED['columns'])} | {actual_cols} |",
        f"| header row | {ASSUMED['header']} | {'yes' if schema.has_header else 'no'} |",
        f"| delimiter | tab | `{schema.delimiter!r}` |",
        f"| timestamp | {ASSUMED['timestamp']} | {ts_described} |",
        f"| tracks | {ASSUMED['tracks']} | {PUBLISHED['tracks']:,} (README) |",
        f"| users | {ASSUMED['users']} | {PUBLISHED['users']:,} (README) |",
        f"| rows | {ASSUMED['rows']} | {PUBLISHED['events']:,} (README) |",
        "",
        "## Column roles as resolved",
        "",
        f"- `user_id` at index {schema.user_idx}",
        f"- `track_id` at index {schema.track_idx}",
        (
            f"- `ts` at index {schema.ts_idx}, encoded as **{ts_described}**, "
            f"identified by {schema.ts_resolved_by}"
        ),
        "",
    ]

    if decoded:
        lines += [
            f"First timestamp `{ts_example}` decodes to **{decoded}**, a plausible",
            "Last.fm listening date. A wrong unit or format would land in 1970 or far",
            "in the future, so this is the check that the reading is right.",
            "",
            "**The timestamps carry no timezone and the dataset does not document one.**",
            "They are read as UTC. If they are in fact local time the whole corpus",
            "shifts by hours, which would not disturb any label here: encounters,",
            "recurrence and both horizons are all differences between timestamps in",
            "the same column, and a constant offset cancels. It would matter only for",
            "an absolute claim, such as time-of-day analysis.",
            "",
        ]

    lines += [
        "## First rows, as read",
        "",
        "```",
    ]
    for row in sample:
        lines.append(" | ".join(row))
    lines += [
        "```",
        "",
        "## Ordering in the probed prefix",
        "",
        "```json",
        json.dumps(ordering, indent=2),
        "```",
        "",
    ]

    if ordering["grouped_by_user"] and ordering["ts_ascending_overall"]:
        lines.append(
            "The prefix is grouped by user and ascending in time. The build still "
            "sorts: a prefix is not the file."
        )
    elif ordering["grouped_by_user"]:
        lines.append(
            "The prefix is grouped by user but not globally ascending in time, "
            "which is what per-user ordering looks like. The build sorts by "
            "(user, ts) regardless."
        )
    else:
        lines.append(
            "The prefix is not grouped by user, so the build's sort by (user, ts) "
            "is doing real work rather than confirming an existing order."
        )

    desc, asc = ordering["runs_descending_in_time"], ordering["runs_ascending_in_time"]
    if desc > asc:
        lines += [
            "",
            (
                f"**Within a user, the source runs backwards in time** ({desc} of "
                f"{desc + asc} multi-event runs descend). That is the order the Last.fm "
                "API returns, most recent first. Every adoption label depends on which "
                "play came first, so inheriting this order would invert every encounter: "
                "the *last* play of a track would be recorded as the first. The build "
                "sorts ascending, and `CompactCorpus.validate` refuses a store that is "
                "not ascending within each user."
            ),
        ]

    lines += [
        "",
        "## Not in this dataset",
        "",
        "**No artist or song metadata exists anywhere in record 15394646.** All 46",
        "files are keyed by track id; artist names live in the base Music4All",
        "dataset, obtained by request from `contact4music4all@gmail.com`. The",
        "`artist-affinity` baseline and the `new artist` slice cannot be built from",
        "the open files, and are substituted by an item-adoption-rate baseline and",
        "a tag/genre `new_neighborhood` slice until that request is answered.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    events = args.path / EVENTS_FILE
    if not events.exists():
        print(f"missing {events}\nrun: python scripts/download_onion.py")
        return 2

    print(f"sniffing {events}")
    schema = sniff_schema(events)
    print(json.dumps(schema.summary(), indent=2))

    print("\nfirst rows:")
    for row in schema.sample_rows:
        print("  " + " | ".join(row))

    print("\nprobing ordering...")
    ordering = probe_ordering(events, schema)
    print(json.dumps(ordering, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(schema, ordering, schema.sample_rows), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"next: python scripts/build_onion.py --path {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
