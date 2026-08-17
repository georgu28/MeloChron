"""Async upload jobs.

History files are large --- an extended Spotify export runs to tens of
megabytes of JSON --- and parsing one is several seconds of pandas work. Doing
that inside the upload request would hold the connection open for the whole
parse, block a worker, and time out behind most proxies. So the upload endpoint
does the one thing that must happen synchronously (stream the bytes to disk),
returns a job id, and hands the parse to a worker thread. The client polls.

The store is in memory, on purpose. The Phase 5 brief is explicit that this
project should not rebuild a database-backed stack that BoilerVault already
demonstrates; what is new here is the ML-specific serving path. A single
process at a concurrency of five needs a dict and a lock, and pretending
otherwise would be resume-driven architecture. The cost is stated rather than
hidden: **a restart drops in-flight jobs**, and clients must be able to
re-upload. That is the correct trade at this scale and would be the wrong one
at a larger one.

Two bounds keep an in-memory store from being a leak. Jobs expire on a TTL, and
the store caps how many it will hold at once, evicting oldest-first. Both sweep
the job's temp directory as they go, because the uploaded file outlives the
job record otherwise and fills the disk instead of the heap.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from melochron.serving.uploads import ParsedHistory

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60
DEFAULT_MAX_JOBS = 64


class JobState(str, Enum):
    """Lifecycle of one upload.

    ``str`` mixin so the value serialises directly to JSON without a custom
    encoder, and so a client comparing against the string literal works.
    """

    QUEUED = "queued"
    PARSING = "parsing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class UploadJob:
    id: str
    filename: str
    size_bytes: int
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    parsed: ParsedHistory | None = None
    workdir: Path | None = None
    parse_seconds: float | None = None

    def status(self) -> dict:
        """Client-facing view. Never includes the history itself, which can be
        20k rows and is not what a status poll is asking for."""
        payload = {
            "job_id": self.id,
            "state": self.state.value,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.error:
            payload["error"] = self.error
        if self.parse_seconds is not None:
            payload["parse_seconds"] = round(self.parse_seconds, 2)
        if self.parsed is not None:
            payload["stats"] = self.parsed.stats
        return payload


class JobStore:
    """Bounded, TTL-expiring, thread-safe store of upload jobs."""

    def __init__(
        self,
        root: Path,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_jobs: int = DEFAULT_MAX_JOBS,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_jobs = max_jobs
        self._jobs: dict[str, UploadJob] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, size_bytes: int = 0) -> UploadJob:
        job_id = uuid.uuid4().hex
        workdir = self.root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = UploadJob(id=job_id, filename=filename, size_bytes=size_bytes, workdir=workdir)
        with self._lock:
            self._jobs[job_id] = job
        self._enforce_bounds()
        return job

    def get(self, job_id: str) -> UploadJob | None:
        self._enforce_bounds()
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> UploadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            self._cleanup(job)

    def _cleanup(self, job: UploadJob) -> None:
        """Remove a job's temp directory. Failure here is logged, not raised:
        a job being evicted is not a request anyone is waiting on, and an
        undeletable temp dir must not take down the sweep for every other job."""
        if job.workdir and job.workdir.exists():
            try:
                shutil.rmtree(job.workdir, ignore_errors=True)
            except OSError:
                log.warning("could not remove workdir %s", job.workdir, exc_info=True)

    def _enforce_bounds(self) -> None:
        """Drop expired jobs, then oldest-first down to ``max_jobs``.

        Jobs still parsing are exempt from the size cap --- evicting one would
        orphan a running worker thread that is about to write into a directory
        the sweep just deleted. They remain subject to the TTL, which is the
        backstop for a parse that has genuinely hung.
        """
        now = time.time()
        evicted: list[UploadJob] = []

        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if now - job.updated_at > self.ttl_seconds:
                    evicted.append(self._jobs.pop(job_id))

            overflow = len(self._jobs) - self.max_jobs
            if overflow > 0:
                candidates = sorted(
                    (j for j in self._jobs.values() if j.state != JobState.PARSING),
                    key=lambda j: j.created_at,
                )
                for job in candidates[:overflow]:
                    evicted.append(self._jobs.pop(job.id))

        for job in evicted:
            self._cleanup(job)

    def stats(self) -> dict:
        with self._lock:
            states: dict[str, int] = {}
            for job in self._jobs.values():
                states[job.state.value] = states.get(job.state.value, 0) + 1
            return {"jobs": len(self._jobs), "by_state": states, "ttl_seconds": self.ttl_seconds}
