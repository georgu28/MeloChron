"""Phase 2a: build the genre feature matrix and report its coverage.

    python scripts/build_features.py

Music4All-Onion ships tf-idf matrices, so this is a read-and-align rather than
an embedding job. The coverage report is the point: a vector source that looks
complete overall can still be empty exactly where the discovery slice lives, and
an aggregate percentage hides that. The breakdown is by popularity decile for
that reason.

Genres are the project's vector source. Tags are read only when ``--tags`` is
passed, for the Phase 4 ablation, and never define a slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from melochron.adoption import features
from melochron.adoption.corpus import CompactCorpus

DEFAULT_RAW = Path("data/raw/music4all-onion")
DEFAULT_STORE = Path("data/interim/onion-v1")
DEFAULT_OUT = Path("data/interim/onion-features-v1")
DEFAULT_REPORT = Path("artifacts/adoption/phase2-features.md")


def render(genres: dict, tags: dict | None) -> str:
    lines = [
        "# Phase 2 — feature coverage",
        "",
        "Precomputed tf-idf from the Onion release, aligned to track codes.",
        "",
        "| source | dims | tracks covered | all-zero | mean nonzero |",
        "|---|---|---|---|---|",
        (
            f"| genres | {genres['dims']} | {genres['present']:,} / {genres['tracks']:,} "
            f"({genres['present_frac']:.2%}) | {genres['all_zero']:,} | "
            f"{genres['mean_nonzero']} |"
        ),
    ]
    if tags:
        lines.append(
            f"| tags | {tags['dims']} | {tags['present']:,} / {tags['tracks']:,} "
            f"({tags['present_frac']:.2%}) | {tags['all_zero']:,} | {tags['mean_nonzero']} |"
        )

    lines += [
        "",
        "## Coverage by popularity decile",
        "",
        "The number that decides which source can carry a discovery slice.",
        "",
        "| decile | tracks | genres present | genres mean nonzero |"
        + (" tags present |" if tags else ""),
        "|---|---|---|---|" + ("---|" if tags else ""),
    ]
    for i, row in enumerate(genres["by_popularity_decile"]):
        line = (
            f"| {row['decile']} | {row['tracks']:,} | {row['present_frac']:.1%} | "
            f"{row['mean_nonzero']} |"
        )
        if tags:
            line += f" {tags['by_popularity_decile'][i]['present_frac']:.1%} |"
        lines.append(line)

    lines += [
        "",
        "**Genres carry the slice; tags cannot.** Genre coverage is flat across the",
        "whole popularity range and its density barely moves, so a similarity computed",
        "from it means the same thing for an obscure track as for a famous one.",
        "",
    ]
    if tags:
        first = tags["by_popularity_decile"][0]["present_frac"]
        last = tags["by_popularity_decile"][-1]["present_frac"]
        lines += [
            (
                f"Tag coverage runs from {first:.1%} in the least-played decile to "
                f"{last:.1%} in the most-played. Using tags for the `new_neighborhood`"
            ),
            "slice would make the discovery result least reliable exactly where the",
            "discovery claim lives — the trap the previous project hit with",
            "artist-level tags. They stay available as a Phase 4 ablation.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--tags", action="store_true", help="also read the tag matrix")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    corpus = CompactCorpus.load(args.store, mmap=True)
    tracks = np.load(args.store / "tracks.npy")
    plays = np.bincount(np.asarray(corpus.track_code), minlength=corpus.n_tracks)

    args.out.mkdir(parents=True, exist_ok=True)
    genre_path = args.out / "genres.npy"

    if genre_path.exists() and not args.force:
        print(f"loading cached genre matrix from {genre_path}")
        matrix = np.load(genre_path)
        present = np.load(args.out / "genres_present.npy")
    else:
        print(f"reading {features.GENRES_FILE}...")
        matrix, present = features.load_matrix(args.raw / features.GENRES_FILE, tracks)
        np.save(genre_path, matrix)
        np.save(args.out / "genres_present.npy", present)
        print(f"  wrote {genre_path}  {matrix.shape}")

    genres = features.coverage_report(matrix, present, plays)
    print(
        f"genres: {genres['present']:,}/{genres['tracks']:,} covered, "
        f"{genres['all_zero']:,} all-zero, mean nonzero {genres['mean_nonzero']}"
    )

    tags = None
    if args.tags:
        print(f"reading {features.TAGS_FILE}...")
        tag_matrix, tag_present = features.load_matrix(args.raw / features.TAGS_FILE, tracks)
        np.save(args.out / "tags.npy", tag_matrix)
        np.save(args.out / "tags_present.npy", tag_present)
        tags = features.coverage_report(tag_matrix, tag_present, plays)
        print(f"tags: {tags['present']:,}/{tags['tracks']:,} covered")

    (args.out / "features.json").write_text(
        json.dumps({"genres": genres, "tags": tags}, indent=2), encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(genres, tags), encoding="utf-8")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
