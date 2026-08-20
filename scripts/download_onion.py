"""Fetch the Music4All-Onion files the adoption task needs.

    python scripts/download_onion.py

Zenodo record 15394646, DOI 10.5281/zenodo.15394646, CC-BY-4.0.

The record holds 46 files totalling ~19.9 GB. This fetches the five that the
adoption task actually reads, ~2.8 GB:

    README.md                         the schema source of record
    userid_trackid_timestamp.tsv.bz2  the listening events; every label
    userid_trackid_count.tsv.bz2      an independent check on the label builder
    id_tags_tf-idf.tsv.bz2            text-similarity baseline
    id_genres_tf-idf.tsv.bz2          genre labels

The count file is fetched even though it carries no timestamps and is
superseded by the events file for every modelling purpose. It is here as ground
truth: it publishes 50,016,042 distinct (user, track) pairs, which is exactly
the encounter count the label builder must independently arrive at, and its
counts must sum to the 252,984,396 events. In a 253M-row pipeline an off-by-one
in the encounter logic produces no error and no obviously wrong number, and this
is the cheapest thing that would catch it.

Sizes and md5s are read from the Zenodo API at run time rather than hardcoded,
so a re-versioned record fails loudly on checksum instead of silently parsing
different data.

**There is no artist metadata in this record.** All 46 files are keyed by track
id; artist and song names live in the base Music4All dataset, which is obtained
by emailing contact4music4all@gmail.com. The adoption task's discovery slice is
built from tags/genres instead -- see the Phase 0 report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from melochron.net import USER_AGENT, download_resumable, file_digest, human_bytes

RECORD = "15394646"
API = f"https://zenodo.org/api/records/{RECORD}"
DOI = "10.5281/zenodo.15394646"

#: Fetched by default. Ordered smallest-first so a bad record, a wrong URL or a
#: broken checksum surfaces in seconds rather than after a 2.2 GB transfer.
WANTED = [
    "README.md",
    "id_genres_tf-idf.tsv.bz2",
    "id_tags_tf-idf.tsv.bz2",
    "userid_trackid_count.tsv.bz2",
    "userid_trackid_timestamp.tsv.bz2",
]

DEFAULT_DEST = Path("data/raw/music4all-onion")

CITATION = (
    "Moscati, M., Parada-Cabaleiro, E., Deldjoo, Y., Zangerle, E., Schedl, M. "
    "Music4All-Onion: A Large-Scale Multi-faceted Content-Centric Music "
    "Recommendation Dataset. CIKM 2022. Zenodo record 15394646, CC-BY-4.0."
)


def fetch_manifest(api: str = API) -> dict[str, dict]:
    """Map filename -> {size, md5, url} from the Zenodo record API."""
    request = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        record = json.load(response)

    files = {}
    for entry in record["files"]:
        checksum = entry["checksum"]
        # Zenodo prefixes the algorithm, e.g. "md5:dfe82201...". Anything else
        # means the API changed shape and the verification below is not doing
        # what it claims, so refuse rather than skip the check.
        algorithm, _, digest = checksum.partition(":")
        if algorithm != "md5":
            raise ValueError(f"{entry['key']}: expected an md5 checksum, got {checksum!r}")
        files[entry["key"]] = {
            "size": int(entry["size"]),
            "md5": digest,
            "url": entry["links"]["self"],
        }
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--only", nargs="*", metavar="FILE", help="fetch just these keys")
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="do not md5 files that are already the right size (hashing 2.2GB costs ~10s)",
    )
    args = ap.parse_args(argv)

    print(f"record  {API}")
    print(f"dest    {args.dest}")

    try:
        manifest = fetch_manifest()
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        print(f"could not read the Zenodo record: {exc}", file=sys.stderr)
        return 1

    wanted = args.only or WANTED
    missing = [k for k in wanted if k not in manifest]
    if missing:
        print(f"not in record {RECORD}: {', '.join(missing)}", file=sys.stderr)
        return 2

    total = sum(manifest[k]["size"] for k in wanted)
    print(f"files   {len(wanted)}, {human_bytes(total)}\n")

    results = {}
    for key in wanted:
        entry = manifest[key]
        target = args.dest / key
        print(f"{key}  ({human_bytes(entry['size'])})")

        complete = target.exists() and target.stat().st_size == entry["size"]
        download_resumable(entry["url"], target, expected_bytes=entry["size"], label=key)

        if complete and args.skip_verify:
            print("    md5 skipped (--skip-verify)")
            digest = None
        else:
            digest = file_digest(target, "md5")
            if digest != entry["md5"]:
                print(
                    f"    md5 MISMATCH: got {digest}, record says {entry['md5']}",
                    file=sys.stderr,
                )
                print("    the local file is not the file Zenodo published", file=sys.stderr)
                return 3
            print(f"    md5 ok  {digest}")

        results[key] = {"bytes": entry["size"], "md5": digest or entry["md5"]}

    (args.dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "record": API,
                "doi": DOI,
                "files": results,
                "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "license": "CC-BY-4.0",
                "citation": CITATION,
                "note": (
                    "No artist/song metadata exists in this record; all 46 files are "
                    "keyed by track id. Artist names require the base Music4All "
                    "dataset (contact4music4all@gmail.com)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {args.dest / 'SOURCE.json'}")
    print(f"next: python scripts/inspect_onion.py --path {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
