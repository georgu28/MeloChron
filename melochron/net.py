"""Resumable HTTP fetching and file digests.

Extracted so that corpus downloaders do not each reimplement the Range-resume
dance. ``scripts/download_lastfm1k.py`` predates this module and still carries
its own copy; it works and is not worth the churn of rewriting a script whose
job is already done.

Everything here is deliberately stdlib-only. A downloader that cannot run
before ``pip install`` has finished is a downloader with a bootstrap problem.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "melochron/0.1"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def file_digest(path: Path, algorithm: str = "md5") -> str:
    """Hex digest of ``path``, read in chunks so a 2 GB file is not resident."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(
    url: str,
    target: Path,
    expected_bytes: int | None = None,
    label: str = "",
) -> Path:
    """Download ``url`` to ``target``, resuming a partial file if one is present.

    A 2.2 GB transfer that dies at 90% must not start over, so an existing
    partial file is continued with a Range request. If the server answers 200
    instead of 206 it ignored the Range header and is about to send the whole
    body, so the local prefix is dropped rather than appended to -- appending
    would produce a file that is the right size only if you are unlucky, and
    corrupt either way.
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS url {url!r}")

    target.parent.mkdir(parents=True, exist_ok=True)
    have = target.stat().st_size if target.exists() else 0

    if expected_bytes and have == expected_bytes:
        print(f"  {label or target.name}: already complete ({human_bytes(have)})")
        return target
    if expected_bytes and have > expected_bytes:
        print(f"  {label or target.name}: local file larger than expected; refetching")
        target.unlink()
        have = 0

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if have:
        request.add_header("Range", f"bytes={have}-")

    started = time.time()
    with urllib.request.urlopen(request) as response:
        if have and response.status != 206:
            print("    server ignored Range; restarting from zero")
            have = 0

        remaining = response.headers.get("Content-Length")
        total = (int(remaining) + have) if remaining else expected_bytes
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
                        f"\r    {pct}{human_bytes(done)} at {human_bytes(rate)}/s",
                        end="",
                        flush=True,
                    )
                    last_print = now

    elapsed = time.time() - started
    print(f"\r    {human_bytes(done)} in {elapsed:.0f}s{' ' * 24}")
    return target
