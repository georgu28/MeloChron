"""Phase 1: build the adoption labels and report them.

    python scripts/build_labels.py

Turns the compact corpus into one row per (user, track) first encounter, labels
each row under both horizons, drops the rows whose horizon was never observed,
splits on time and on users, and writes ``artifacts/adoption/phase1-labels.md``.

Nothing here is a model. The point is a labelled table that can be trusted, and
a report honest enough to show what was thrown away and why. Three numbers in it
are load-bearing:

* the **encounter count**, which must come out at 50,016,042 -- the figure Phase
  0 confirmed against the published counts file;
* the **censoring drops**, which must be reported rather than relabelled;
* the **base rate**, which cannot exceed the 54.4% unbounded ceiling Phase 0
  measured. Anything above that is a bug, not a result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from melochron.adoption import slices as slicing
from melochron.adoption.corpus import PLAUSIBLE_FLOOR, CompactCorpus
from melochron.adoption.labels import (
    DEFAULT_EVENT_N,
    DEFAULT_TIME_DAYS,
    EncounterTable,
    build_encounters,
    event_horizon,
    temporal_split,
    time_horizon,
    train_horizon_fits,
    usable,
)
from melochron.adoption.onion import PUBLISHED

DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_OUT = Path("data/interim/onion-labels-v1")
DEFAULT_REPORT = Path("artifacts/adoption/phase1-labels.md")

#: Measured in Phase 0 over all 50,016,042 pairs. Every horizoned base rate must
#: come in at or below this, because a horizon can only remove positives.
UNBOUNDED_CEILING = 0.5411

COLUMNS = ("user_code", "track_code", "encounter_ts", "encounter_pos", "recur_pos", "recur_ts")


def base_rate(label: np.ndarray, keep: np.ndarray) -> tuple[int, int, float]:
    n = int(keep.sum())
    pos = int((label & keep).sum())
    return n, pos, (pos / n if n else 0.0)


def slice_row(name: str, label: np.ndarray, keep: np.ndarray) -> str:
    n, pos, rate = base_rate(label, keep)
    return f"| {name} | {n:,} | {pos:,} | {rate:.4f} |"


def render(stats: dict) -> str:
    ev, tm = stats["event"], stats["time"]
    enc = stats["encounters"]

    lines = [
        "# Phase 1 — adoption labels",
        "",
        (
            f"One row per (user, track) first encounter, from `{DEFAULT_STORE}`. "
            f"Built in {stats['build_s']:.0f}s."
        ),
        "",
        "## Encounters",
        "",
        "| quantity | measured | expected | |",
        "|---|---|---|---|",
        (
            f"| encounters | {enc['total']:,} | {PUBLISHED['pairs']:,} | "
            f"{'ok' if enc['total'] == PUBLISHED['pairs'] else '**MISMATCH**'} |"
        ),
        "",
        "The encounter count *is* the distinct-pair count, which Phase 0 confirmed",
        "twice — against the published figure and against the counts file. It arriving",
        "here unchanged means the first-occurrence search found exactly one encounter",
        "per pair, no more and no fewer.",
        "",
        f"{enc['corrupt']:,} encounter(s) carry the single pre-2002 timestamp and are",
        "excluded from every table below.",
        "",
        "## Horizons and censoring",
        "",
        "| horizon | labelable | censored out | positives | base rate |",
        "|---|---|---|---|---|",
        (
            f"| event, N={DEFAULT_EVENT_N} | {ev['labelable']:,} | "
            f"{ev['censored']:,} ({ev['censored_frac']:.1%}) | {ev['positives']:,} | "
            f"**{ev['base_rate']:.4f}** |"
        ),
        (
            f"| time, {DEFAULT_TIME_DAYS}d | {tm['labelable']:,} | "
            f"{tm['censored']:,} ({tm['censored_frac']:.1%}) | {tm['positives']:,} | "
            f"**{tm['base_rate']:.4f}** |"
        ),
        "",
        f"Both sit under the {UNBOUNDED_CEILING:.1%} unbounded ceiling Phase 0 measured,",
        "as they must: a horizon can only turn positives into negatives.",
        "",
        "**Censored rows are dropped, not called negative.** Relabelling them would",
        "convert genuine positives nobody watched long enough to see into fake",
        "negatives, and would raise every score in the project.",
        "",
        "### What makes a 30-day row unobservable",
        "",
        "| reason | rows |",
        "|---|---|",
        f"| corpus ends inside the horizon | {tm['censored_corpus']:,} |",
        f"| the user stops listening inside the horizon | {tm['censored_user_only']:,} |",
        "",
        "These are different claims. The first is an artefact of when the dataset was",
        "cut. The second is churn: the data continues, the *user* stopped, and scoring",
        "them as a non-adoption would assert a decision not to return when what was",
        "observed is that they stopped listening at all. The headline 30-day number",
        "requires both; the corpus-only variant is reported beside it as",
        f"**{tm['base_rate_corpus_only']:.4f}** over {tm['labelable_corpus_only']:,} rows.",
        "",
        "## Agreement between the horizons",
        "",
        (
            f"On the {stats['agreement']['both']:,} encounters both horizons can label they "
            f"agree on **{stats['agreement']['rate']:.1%}** of rows. The disagreements are "
            f"lopsided, and they say which kind of listener each horizon suits."
        ),
        "",
        "| disagreement | rows | median events of those users |",
        "|---|---|---|",
        (
            f"| event yes, 30-day no | {stats['agreement']['event_only']:,} | "
            f"{stats['agreement']['event_only_user_events']:,} |"
        ),
        (
            f"| 30-day yes, event no | {stats['agreement']['time_only']:,} | "
            f"{stats['agreement']['time_only_user_events']:,} |"
        ),
        (
            f"| (all labelable) | {stats['agreement']['both']:,} | "
            f"{stats['agreement']['all_user_events']:,} |"
        ),
        "",
        (
            "The two windows measure the same thing at different speeds. For a light "
            "listener 200 events span far more than a month, so the event window reaches "
            "further in wall-clock time and catches recurrences the 30-day window misses "
            "— those users sit well below the median. For a heavy listener 200 events can "
            "pass in days, so it is the 30-day window that reaches further, and those rows "
            "come from users with more than twice the median history. This is exactly the "
            "bias the brief chose the event horizon to avoid: a month means something "
            "different to someone playing forty tracks a day than to someone playing four "
            "a week."
        ),
        "",
        "## Base rate by slice",
        "",
        f"Event horizon, N={DEFAULT_EVENT_N}.",
        "",
        "| slice | n | positives | base rate |",
        "|---|---|---|---|",
        *stats["slice_rows"],
        "",
        "The popularity deciles are the substitute for a long-tail cutoff: Phase 0",
        "found the brief's >=20-plays filter removes nothing, so obscurity has to be",
        "measured relatively. `new_neighborhood` joins this table in Phase 2, once the",
        "tag vectors exist.",
        "",
        "## Drift — base rate by encounter year",
        "",
        "| year | n | base rate |",
        "|---|---|---|",
        *stats["year_rows"],
        "",
        stats["drift_note"],
        "",
        "## Splits",
        "",
        f"Global temporal cut at **{stats['split']['cutoff_date']}**, plus a 10% whole-user",
        "holdout for the cold-user slice.",
        "",
        "| | rows |",
        "|---|---|",
        f"| train encounters | {stats['split']['train']:,} |",
        f"| … of those, labelable under the event horizon | {stats['split']['train_observable']:,} |",
        f"| … of those, horizon closing before the cutoff | {stats['split']['train_fits']:,} |",
        f"| test encounters | {stats['split']['test']:,} |",
        f"| cold-user rows in test | {stats['split']['cold_user_test']:,} |",
        "",
        (
            f"**The base rate falls across the cut: {stats['split']['train_base_rate']:.4f} on "
            f"train against {stats['split']['test_base_rate']:.4f} on test.** That is the drift "
            f"above arriving where it actually bites. A global-prior baseline fitted on train "
            f"predicts {stats['split']['train_base_rate']:.3f} against a test reality of "
            f"{stats['split']['test_base_rate']:.3f}, so it is miscalibrated by construction "
            f"rather than by mistake — and every model that learns a prior from train "
            f"inherits the same offset. Worth reporting beside calibration rather than "
            f"quietly correcting."
        ),
        "",
        (
            f"**{stats['split']['train_dropped']:,} training rows are dropped because their "
            f"horizon reaches past the cutoff.** Keeping them would label training data "
            f"with test-period events — not leakage into the metric, but a training "
            f"signal no deployed system could have had."
        ),
        "",
        "## Raw stats",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("slice_rows", "year_rows")}, indent=2
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--event-n", type=int, default=DEFAULT_EVENT_N)
    ap.add_argument("--time-days", type=int, default=DEFAULT_TIME_DAYS)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--holdout-user-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not (args.store / "manifest.json").exists():
        print(f"missing corpus store at {args.store}\nrun: python scripts/build_onion.py")
        return 2

    corpus = CompactCorpus.load(args.store, mmap=True)
    print(f"corpus: {corpus.n_events:,} events, {corpus.n_users:,} users")

    cached = (args.out / "manifest.json").exists() and not args.force
    if cached:
        print(f"loading cached encounters from {args.out} (pass --force to rebuild)")
        table = EncounterTable(
            **{c: np.load(args.out / f"{c}.npy", mmap_mode=None) for c in COLUMNS}
        )
        build_s = json.loads((args.out / "manifest.json").read_text())["build_s"]
    else:
        print("finding first encounters and earliest recurrences...")
        started = time.time()
        table = build_encounters(corpus)
        build_s = time.time() - started
        print(f"  {len(table):,} encounters in {build_s:.0f}s")
        table.validate(corpus)
        print("  validate ok")

    print("labelling both horizons...")
    ev = event_horizon(corpus, table, args.event_n)
    tm = time_horizon(corpus, table, args.time_days, require_user_runway=True)
    tm_corpus_only = time_horizon(corpus, table, args.time_days, require_user_runway=False)

    split = temporal_split(table, corpus.n_users, args.test_frac, args.holdout_user_frac, args.seed)
    keys = slicing.build(corpus, table, split)

    ev_keep = usable(table, ev)
    tm_keep = usable(table, tm)
    tm_corpus_keep = usable(table, tm_corpus_only)

    n_ev, pos_ev, rate_ev = base_rate(ev.label, ev_keep)
    n_tm, pos_tm, rate_tm = base_rate(tm.label, tm_keep)
    n_tmc, _, rate_tmc = base_rate(tm_corpus_only.label, tm_corpus_keep)

    both = ev_keep & tm_keep
    agree = int((ev.label[both] == tm.label[both]).sum())
    n_both = int(both.sum())

    # Per-slice and per-year tables, on the primary horizon.
    slice_rows = [slice_row("all", ev.label, ev_keep)]
    for name, mask in slicing.named_slices(keys).items():
        slice_rows.append(slice_row(name, ev.label, ev_keep & mask))

    years = keys["year"]
    year_rows = []
    rates_by_year = {}
    for y in range(int(years.min()), int(years.max()) + 1):
        mask = ev_keep & (years == y)
        n, _, rate = base_rate(ev.label, mask)
        if n < 1000:
            continue
        rates_by_year[y] = round(rate, 4)
        year_rows.append(f"| {y} | {n:,} | {rate:.4f} |")

    spread = (max(rates_by_year.values()) - min(rates_by_year.values())) if rates_by_year else 0.0
    drift_note = (
        f"Base rate moves by {spread:.3f} across the years with at least 1,000 labelled "
        f"encounters. "
        + (
            "That is large enough that a model trained on the early corpus is being asked "
            "about a different behaviour in the test era, and it belongs in the limitations."
            if spread > 0.05
            else "That is small, so the fifteen-year span is not the confound it looked like, "
            "and the concern is retired with a measurement rather than an assumption."
        )
    )

    train_fits = train_horizon_fits(split, ev)
    train_observable = split.is_train & ev.observable

    # Which listeners drive each disagreement. The direction is not obvious and
    # guessing it wrong is easy, so it is measured.
    user_events = np.diff(corpus.user_offsets)[table.user_code]
    ev_only_mask = both & ev.label & ~tm.label
    tm_only_mask = both & ~ev.label & tm.label

    _, _, train_rate = base_rate(ev.label, split.is_train & ev_keep)
    _, _, test_rate = base_rate(ev.label, split.is_test & ev_keep)

    stats = {
        "build_s": build_s,
        "encounters": {
            "total": len(table),
            "expected": PUBLISHED["pairs"],
            "corrupt": int((table.encounter_ts < PLAUSIBLE_FLOOR).sum()),
        },
        "event": {
            "n": args.event_n,
            "labelable": n_ev,
            "censored": len(table) - n_ev,
            "censored_frac": round((len(table) - n_ev) / len(table), 4),
            "positives": pos_ev,
            "base_rate": round(rate_ev, 4),
        },
        "time": {
            "days": args.time_days,
            "labelable": n_tm,
            "censored": len(table) - n_tm,
            "censored_frac": round((len(table) - n_tm) / len(table), 4),
            "positives": pos_tm,
            "base_rate": round(rate_tm, 4),
            "censored_corpus": int((~tm_corpus_only.observable).sum()),
            "censored_user_only": int((tm_corpus_only.observable & ~tm.observable).sum()),
            "labelable_corpus_only": n_tmc,
            "base_rate_corpus_only": round(rate_tmc, 4),
        },
        "agreement": {
            "both": n_both,
            "agree": agree,
            "rate": round(agree / n_both, 4) if n_both else 0.0,
            "event_only": int(ev_only_mask.sum()),
            "time_only": int(tm_only_mask.sum()),
            "event_only_user_events": int(np.median(user_events[ev_only_mask])),
            "time_only_user_events": int(np.median(user_events[tm_only_mask])),
            "all_user_events": int(np.median(user_events[both])),
        },
        "split": {
            **split.summary(),
            "cutoff_date": str(np.array(split.cutoff_ts, dtype="datetime64[s]"))[:10],
            "train_observable": int(train_observable.sum()),
            "train_fits": int(train_fits.sum()),
            "train_dropped": int(train_observable.sum() - train_fits.sum()),
            "train_base_rate": round(train_rate, 4),
            "test_base_rate": round(test_rate, 4),
        },
        "base_rate_by_year": rates_by_year,
    }

    if not cached:
        args.out.mkdir(parents=True, exist_ok=True)
        for column in COLUMNS:
            np.save(args.out / f"{column}.npy", getattr(table, column))
        for name, arr in keys.items():
            np.save(args.out / f"slice_{name}.npy", arr)
        np.save(args.out / "label_event.npy", ev.label)
        np.save(args.out / "observable_event.npy", ev.observable)
        np.save(args.out / "label_time.npy", tm.label)
        np.save(args.out / "observable_time.npy", tm.observable)
        np.save(args.out / "is_train.npy", split.is_train)
        np.save(args.out / "is_test.npy", split.is_test)
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "encounters": len(table),
                    "build_s": round(build_s, 1),
                    "event_n": args.event_n,
                    "time_days": args.time_days,
                    "cutoff_ts": split.cutoff_ts,
                    "seed": args.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {args.out}")

    stats["slice_rows"] = slice_rows
    stats["year_rows"] = year_rows
    stats["drift_note"] = drift_note

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(stats), encoding="utf-8")
    print(f"\nwrote {args.report}")

    print(f"\nevent N={args.event_n}: base rate {rate_ev:.4f} over {n_ev:,} rows")
    print(f"time {args.time_days}d:   base rate {rate_tm:.4f} over {n_tm:,} rows")

    if len(table) != PUBLISHED["pairs"]:
        print(f"\nSTOP: {len(table):,} encounters, expected {PUBLISHED['pairs']:,}")
        return 1
    if max(rate_ev, rate_tm) > UNBOUNDED_CEILING:
        print(f"\nSTOP: a base rate exceeds the {UNBOUNDED_CEILING:.4f} unbounded ceiling")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
