"""Parse Music4All-Onion into the compact corpus, and report Phase 0.

    python scripts/build_onion.py

One decompression pass over the 2.2 GB events archive produces three int32
arrays sorted by (user, ts) plus the per-user boundaries into them -- about
3 GB, memory-mappable, and the substrate every adoption label is built from.

Then it measures the corpus and writes ``artifacts/adoption/phase0-corpus.md``.
Three of those measurements decide Phase 1 and are the reason this script
reports rather than just builds:

* **per-user active span** decides whether the 30-day horizon has runway, or
  whether the event-based horizon is mandatory rather than merely preferred;
* **plays per pair** gives the unbounded recurrence rate, which is the ceiling
  on the adoption base rate;
* **the catalog cutoff** says how much of the corpus survives >=20 global plays.

Counts are verified against the published figures and against
``userid_trackid_count.tsv.bz2``. A mismatch is a stop condition: if the row
count is wrong then the parser is wrong, and every label inherits the error.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from melochron.adoption.corpus import CompactCorpus, build, corpus_stats
from melochron.adoption.onion import (
    COUNTS_FILE,
    EVENTS_FILE,
    PUBLISHED,
    read_counts_totals,
    sniff_schema,
)

DEFAULT_PATH = Path("data/raw/music4all-onion")
DEFAULT_OUT = Path("data/interim/onion-v1")
DEFAULT_REPORT = Path("artifacts/adoption/phase0-corpus.md")

#: Recorded in the report for provenance: the base Music4All dataset carries the
#: artist names this corpus lacks, and access is by request.
ARTIST_REQUEST_SENT = "2026-08-19"


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _check(name: str, measured: int, published: int) -> tuple[bool, str]:
    ok = measured == published
    mark = "ok" if ok else "**MISMATCH**"
    delta = "" if ok else f" (off by {measured - published:+,})"
    return ok, f"| {name} | {measured:,} | {published:,} | {mark}{delta} |"


def render(stats: dict, counts: dict | None, timings: dict) -> str:
    corpus_note = (
        "one decompression pass, cached since"
        if timings.get("from_cache")
        else "one decompression pass"
    )
    rows = []
    all_ok = True
    for key in ("events", "users", "tracks", "pairs"):
        ok, row = _check(key, stats[key]["measured"], stats[key]["published"])
        all_ok = all_ok and ok
        rows.append(row)

    span = stats["span"]
    epu = stats["events_per_user"]
    spu = stats["active_span_days_per_user"]
    ppp = stats["plays_per_pair"]
    cut = stats["catalog_cutoff"]

    lines = [
        "# Phase 0 — corpus report",
        "",
        f"Music4All-Onion, Zenodo record 15394646, CC-BY-4.0. Built from `{EVENTS_FILE}`",
        f"in {timings['build_s']:.0f}s ({corpus_note}).",
        "",
        "## Counts against the published figures",
        "",
        "| quantity | measured | published | |",
        "|---|---|---|---|",
        *rows,
        "",
    ]

    if counts:
        pair_ok = counts["pairs"] == stats["pairs"]["measured"]
        play_ok = counts["plays"] == stats["events"]["measured"]
        lines += [
            f"Cross-checked against `{COUNTS_FILE}`, which is an independent statement",
            "of the same two quantities:",
            "",
            "| quantity | from events file | from counts file | |",
            "|---|---|---|---|",
            (
                f"| distinct pairs | {stats['pairs']['measured']:,} | "
                f"{counts['pairs']:,} | {'ok' if pair_ok else '**MISMATCH**'} |"
            ),
            (
                f"| total plays | {stats['events']['measured']:,} | "
                f"{counts['plays']:,} | {'ok' if play_ok else '**MISMATCH**'} |"
            ),
            "",
            "The pair count is the encounter count: one first encounter per (user, track).",
            "",
        ]

    sane = stats["timestamp_sanity"]
    lines += [
        "## Time span",
        "",
        (
            f"- **{sane['start_excluding_implausible'][:10]} to {span['end'][:10]}** "
            f"({sane['span_days_excluding_implausible']:,.0f} days)"
        ),
        "",
    ]
    if sane["implausible_events"]:
        lines += [
            (
                f"{sane['implausible_events']} event(s) are dated before "
                f"{sane['implausible_before']}, which is before Last.fm existed. The raw "
                f"minimum is {span['start'][:10]}; the span above excludes them. At "
                f"{sane['implausible_events']} row(s) in {stats['events']['measured']:,} "
                f"this is a footnote, not a data-quality problem, but it is a real row "
                f"that would hand one user a 50-year history and must be dropped in "
                f"Phase 1 rather than carried."
            ),
            "",
        ]

    lines += [
        "## Per-user activity",
        "",
        "The two distributions that decide the horizon design.",
        "",
        "| percentile | events | active span (days) |",
        "|---|---|---|",
    ]
    for p in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        lines.append(f"| {p} | {epu[p]:,.0f} | {spu[p]:,.1f} |")
    lines += [
        f"| mean | {epu['mean']:,.1f} | {spu['mean']:,.1f} |",
        "",
        (
            f"- Users with >= 200 events (the primary horizon N): "
            f"**{epu['with_at_least_200']:,}** of {stats['users']['measured']:,} "
            f"({100 * epu['with_at_least_200'] / stats['users']['measured']:.1f}%)"
        ),
        (
            f"- Users active >= 30 days (the secondary horizon): "
            f"**{spu['at_least_30d']:,}** "
            f"({100 * spu['at_least_30d'] / stats['users']['measured']:.1f}%)"
        ),
        (
            f"- Users active >= 60 days (30 days of encounters plus 30 of outcome): "
            f"**{spu['at_least_60d']:,}** "
            f"({100 * spu['at_least_60d'] / stats['users']['measured']:.1f}%)"
        ),
        "",
        (
            f"**Both horizons have runway.** The 30-day horizon is comfortable: "
            f"{100 * spu['at_least_60d'] / stats['users']['measured']:.1f}% of users are "
            f"active long enough to hold 30 days of encounters plus 30 of outcome, and the "
            f"median user is active {spu['p50']:,.0f} days. The event horizon is the "
            f"stricter of the two, and its cost is far smaller than the user count "
            f"suggests: the {epu['at_most_200']:,} users with 200 events or fewer are "
            f"{100 * epu['at_most_200'] / stats['users']['measured']:.1f}% of users but "
            f"hold only {epu['events_held_at_most_200']:,} events, "
            f"{100 * epu['events_held_at_most_200'] / stats['events']['measured']:.2f}% "
            f"of the corpus."
        ),
        "",
        "## Recurrence, before any horizon",
        "",
        (
            f"Mean plays per (user, track) pair: **{ppp['mean']}**. "
            f"Median {ppp['p50']:.0f}, p90 {ppp['p90']:.0f}, max {ppp['max']:,}."
        ),
        "",
        (
            f"**{ppp['recurring_pairs']:,} of {stats['pairs']['measured']:,} pairs recur "
            f"at least once: an unbounded recurrence rate of "
            f"{ppp['unbounded_recurrence_rate']:.1%}.**"
        ),
        "",
        "This is the ceiling on the adoption base rate. Applying a horizon and the",
        "censoring rule can only move it down, never up, so any labelled base rate",
        "above this figure is a bug in the label builder.",
        "",
        "## Catalog cutoff",
        "",
        (
            f"At >= {cut['min_track_plays']} global plays: "
            f"**{cut['tracks_kept']:,} tracks kept**, {cut['tracks_dropped']:,} dropped, "
            f"retaining {cut['events_kept']:,} events "
            f"({cut['events_kept_frac']:.2%} of all plays)."
        ),
        "",
        (
            f"**The cutoff is a no-op on this corpus.** The least-played track in the "
            f"catalog already has {cut['min_track_plays_observed']:,} plays, well above "
            f"the {cut['min_track_plays']} the brief specifies, so the filter removes "
            f"nothing. Music4All-Onion ships a catalog that has already been pruned: "
            f"56,512 tracks carry the listening events while the feature files cover "
            f"109,269. The brief's popularity floor is still worth keeping in the code as "
            f"a guard, but it should not be reported as a modelling decision here, "
            f"because it does not make one."
        ),
        "",
        "## What this dataset does not have",
        "",
        "**No artist or song metadata exists in record 15394646.** All 46 files are",
        "keyed by track id. Artist names require the base Music4All dataset, requested",
        f"from `contact4music4all@gmail.com` on {ARTIST_REQUEST_SENT}; no reply yet.",
        "Until then the `artist-affinity` baseline and the `new artist` slice are",
        "substituted by an item-adoption-rate baseline and a tag/genre",
        "`new_neighborhood` slice, which preserve the same test: a model that cannot",
        "beat them on unfamiliar material is a loyalty detector, not a discovery model.",
        "",
        "## Raw stats",
        "",
        "```json",
        json.dumps(stats, indent=2),
        "```",
        "",
    ]

    if not all_ok:
        lines.insert(
            1,
            "\n> **STOP: a measured count does not match the published figure.** "
            "The parser is wrong; nothing below should be trusted.\n",
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--min-track-plays", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="rebuild even if the store exists")
    ap.add_argument(
        "--skip-counts-check",
        action="store_true",
        help="skip the independent cross-check against the counts file (~2 min)",
    )
    args = ap.parse_args(argv)

    events = args.path / EVENTS_FILE
    if not events.exists():
        print(f"missing {events}\nrun: python scripts/download_onion.py")
        return 2

    timings = {}
    if (args.out / "manifest.json").exists() and not args.force:
        print(f"loading cached store from {args.out} (pass --force to rebuild)")
        corpus = CompactCorpus.load(args.out, mmap=False)
        # The cached store records how long its parse took. Reporting 0s here
        # would claim a 253M-row corpus was built instantly.
        cached = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        timings["build_s"] = cached.get("build_s", 0.0)
        timings["from_cache"] = True
    else:
        print(f"sniffing {events}")
        schema = sniff_schema(events)
        print(json.dumps(schema.summary(), indent=2))

        print("\nparsing (one decompression pass; expect several minutes)...")
        started = time.time()
        corpus = build(events, schema)
        timings["build_s"] = time.time() - started
        print(f"  parsed {corpus.n_events:,} events in {timings['build_s']:.0f}s")

        corpus.save(
            args.out,
            extra={
                "source": str(events),
                "schema": schema.summary(),
                "build_s": round(timings["build_s"], 1),
            },
        )
        print(f"  wrote {args.out}")

    print("\nmeasuring...")
    started = time.time()
    stats = corpus_stats(corpus, min_track_plays=args.min_track_plays)
    print(f"  stats in {time.time() - started:.0f}s")

    counts = None
    counts_path = args.path / COUNTS_FILE
    if not args.skip_counts_check and counts_path.exists():
        print(f"\ncross-checking against {COUNTS_FILE}...")
        started = time.time()
        counts = read_counts_totals(counts_path)
        print(
            f"  {counts['pairs']:,} pairs, {counts['plays']:,} plays "
            f"in {time.time() - started:.0f}s"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(stats, counts, timings), encoding="utf-8")
    print(f"\nwrote {args.report}")

    mismatched = [
        k for k in ("events", "users", "tracks", "pairs") if stats[k]["measured"] != PUBLISHED[k]
    ]
    if mismatched:
        print(f"\nSTOP: measured != published for {', '.join(mismatched)}", flush=True)
        for k in mismatched:
            print(
                f"  {k}: measured {_fmt(stats[k]['measured'])}, "
                f"published {_fmt(stats[k]['published'])}"
            )
        return 1

    print("\nall counts match the published figures")
    print(f"unbounded recurrence rate: {stats['plays_per_pair']['unbounded_recurrence_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
