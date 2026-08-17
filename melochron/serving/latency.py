"""Latency measured from the running service.

Phase 5 asks for p50/p95 as real numbers. ``scripts/bench_latency.py`` already
measures the scoring path offline and is the right tool for characterising the
*model*. This module measures what that benchmark structurally cannot: what the
service actually did, at whatever concurrency it actually saw, including the
time a request spent waiting for its turn.

Two channels are recorded separately, and keeping them apart is the point:

``inference``
    Time inside the scoring call. Comparable to the offline benchmark, and the
    number that moves when the model, the catalog size, or the device changes.

``request``
    Wall time for the whole handler, including waiting on the inference
    semaphore. This is what a user experiences. Under concurrency it is
    strictly larger than ``inference``, and the gap between the two is queueing.

Publishing only ``inference`` would understate what users feel. Publishing only
``request`` would make the model look slower than it is and hide where the time
actually went. Reporting both, with the gap visible, is the honest version.

Percentiles are computed over a bounded recent window rather than over all time.
An operational latency number should describe how the service is behaving now;
a lifetime average is dominated by whatever happened at startup and gets less
informative the longer the process runs.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Self

import numpy as np

#: Requests retained per channel. At ~8 bytes a sample this is trivial memory,
#: and it is wide enough that a p95 over it is not dominated by a single call.
DEFAULT_WINDOW = 1024


class LatencyRecorder:
    """Thread-safe ring buffers of recent timings, one per channel.

    Thread-safe because inference runs in a worker thread pool while the event
    loop records around it, so ``record`` is genuinely called from several
    threads. ``deque(maxlen=...)`` append is atomic under the GIL, but the
    lifetime counters next to it are not, so the lock covers both.
    """

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._totals: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, channel: str, milliseconds: float) -> None:
        with self._lock:
            self._samples[channel].append(milliseconds)
            self._totals[channel] += 1

    def channel(self, channel: str) -> dict:
        """Percentiles for one channel over the current window."""
        with self._lock:
            values = np.fromiter(self._samples.get(channel, ()), dtype=np.float64)
            total = self._totals.get(channel, 0)

        if not len(values):
            return {"count": 0, "total": total}

        return {
            "count": len(values),
            "total": total,
            "p50_ms": round(float(np.percentile(values, 50)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2),
            "p99_ms": round(float(np.percentile(values, 99)), 2),
            "max_ms": round(float(values.max()), 2),
            "mean_ms": round(float(values.mean()), 2),
        }

    def snapshot(self) -> dict:
        with self._lock:
            channels = list(self._samples)
        return {"window": self.window, "channels": {c: self.channel(c) for c in channels}}

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._totals.clear()


class Stopwatch:
    """Context manager that records elapsed milliseconds into a channel.

    Uses ``perf_counter``, not ``time()``: the wall clock can step backwards
    over NTP adjustment and produce negative durations, which then poison the
    percentiles for the rest of the window.
    """

    __slots__ = ("_started", "channel", "elapsed_ms", "recorder")

    def __init__(self, recorder: LatencyRecorder, channel: str):
        self.recorder = recorder
        self.channel = channel
        self._started = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Self:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        self.recorder.record(self.channel, self.elapsed_ms)
