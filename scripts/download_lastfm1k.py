"""Fetch and unpack the lastfm-dataset-1K corpus.

    python scripts/download_lastfm1k.py

992 users, ~19M timestamped listening events (Celma, 2010). This is the
pretraining corpus: until it is on disk, every number the repo produces comes
from the synthetic generator and measures nothing about real listening.

**HTTPS is mandatory.** The plain-HTTP URL that most references cite returns
403; the HTTPS form returns 200 with ``Content-Length: 672741554``. Verified
against the host. The archive is ~672 MB compressed and ~2.5 GB unpacked.

The download is resumable via HTTP Range, because 672 MB over a flaky
connection otherwise means starting over. Reruns are cheap: an already-extracted
corpus is detected and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
EXPECTED_BYTES = 672_741_554

#: The data file inside the archive, and what the parser expects to find.
TSV_NAME = "userid-timestamp-artid-artname-traid-traname.tsv"

DEFAULT_DEST = Path("data/raw/lastfm-1k")
CHUNK = 1 << 20


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, target: Path, expected: int | None = None) -> Path:
    """Download ``url`` to ``target``, resuming a partial file if present."""
    target.parent.mkdir(parents=True, exist_ok=True)
    have = target.stat().st_size if target.exists() else 0

    if expected and have == expected:
        print(f"archive already complete: {target} ({_human(have)})")
        return target
    if expected and have > expected:
        print(f"local file is larger than expected ({have} > {expected}); refetching")
        target.unlink()
        have = 0

    request = urllib.request.Request(url, headers={"User-Agent": "melochron/0.1"})
    if have:
        # Resume rather than restart. A 200 here (instead of 206) means the
        # server ignored the Range header, so the local prefix must be dropped.
        request.add_header("Range", f"bytes={have}-")
        print(f"resuming from {_human(have)}")

    started = time.time()
    with urllib.request.urlopen(request) as response:
        if have and response.status != 206:
            print("server ignored Range; restarting from zero")
            have = 0

        remaining = response.headers.get("Content-Length")
        total = (int(remaining) + have) if remaining else expected
        mode = "ab" if have else "wb"

        with target.open(mode) as fh:
            done = have
            last_print = 0.0
            while chunk := response.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_print > 1.0:
                    rate = (done - have) / max(now - started, 1e-6)
                    pct = f"{100 * done / total:5.1f}% " if total else ""
                    print(
                        f"\r  {pct}{_human(done)} at {_human(rate)}/s",
                        end="",
                        flush=True,
                    )
                    last_print = now
    print(f"\r  {_human(done)} in {time.time() - started:.0f}s{' ' * 20}")
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, dest: Path) -> Path:
    """Unpack the TSV out of the archive, flattening the leading directory."""
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        wanted = [m for m in members if Path(m.name).name == TSV_NAME]
        if not wanted:
            names = ", ".join(sorted(Path(m.name).name for m in members)[:10])
            raise FileNotFoundError(f"{TSV_NAME} not found in {archive}. Archive contains: {names}")

        for member in wanted + [m for m in members if m.name.lower().endswith(".txt")]:
            # Flatten: the archive nests everything under lastfm-dataset-1K/.
            member.name = Path(member.name).name
            # filter="data" refuses absolute paths, parent traversal and device
            # files. Default behaviour is deprecated and will change.
            tar.extract(member, dest, filter="data")

    return dest / TSV_NAME


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--keep-archive", action="store_true", help="do not delete the .tar.gz")
    ap.add_argument("--force", action="store_true", help="re-extract even if the TSV exists")
    args = ap.parse_args(argv)

    tsv = args.dest / TSV_NAME
    if tsv.exists() and not args.force:
        size = tsv.stat().st_size
        print(f"already present: {tsv} ({_human(size)})")
        print("pass --force to re-extract")
        return 0

    if not args.url.startswith("https://"):
        # Not pedantry: the http:// form of this host 403s.
        print(f"refusing non-HTTPS url {args.url!r}", file=sys.stderr)
        return 2

    archive = args.dest / "lastfm-dataset-1K.tar.gz"
    print(f"source  {args.url}")
    print(f"dest    {args.dest}")

    download(args.url, archive, expected=EXPECTED_BYTES)

    print("hashing...")
    digest = sha256(archive)
    print(f"sha256  {digest}")

    print("extracting...")
    tsv = extract(archive, args.dest)
    size = tsv.stat().st_size
    print(f"extracted {tsv} ({_human(size)})")

    (args.dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "url": args.url,
                "sha256": digest,
                "archive_bytes": archive.stat().st_size,
                "tsv_bytes": size,
                "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "citation": (
                    "Celma, O. Music Recommendation and Discovery in the Long Tail, "
                    "Springer, 2010. Distributed with permission of Last.fm for "
                    "non-commercial use."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.keep_archive:
        archive.unlink()
        print("removed archive (pass --keep-archive to retain)")

    print(f"\nnext: python scripts/build_dataset.py --data lastfm1k --path {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
