"""Versioned model artifacts, loaded once at process start.

The most important property of this module is what it refuses to do: load a
checkpoint inside a request. A SASRec artifact over a 50k-item catalog is tens
of megabytes of weights plus the vocabulary; deserializing that per call would
put seconds on every request and would make a measured p50 describe disk I/O
rather than inference. The service loads at startup, holds the artifact for its
lifetime, and serves from memory.

Two consequences worth stating, because both are deliberate:

* **A missing artifact is a hard failure, not a lazy load.** If the checkpoint
  cannot be read the service comes up in a degraded state that answers health
  checks and returns 503 from the scoring routes. Deferring the load to the
  first request would convert a deploy-time error into a user-facing one, and
  would hide it from exactly the probe designed to catch it.

* **Version identity comes from the file's own bytes.** A recommendation is
  only interpretable next to the model that produced it, so every artifact gets
  a content hash and every response carries it. Deriving the id from the
  content rather than from a filename or an mtime means it changes when, and
  only when, the weights change --- redeploying the same file is recognisably
  the same model, and quietly swapping the file behind a stable name is not.

Serving several variants at once is supported because this project's headline
claim is a three-way ablation. Holding ``id``, ``text_frozen`` and
``text_finetuned`` side by side lets the same uploaded history be scored by all
three, which is the comparison the README has to make anyway.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from melochron.serving.inference import Recommender
from melochron.train import checkpoint as ckpt

log = logging.getLogger(__name__)

#: Artifact(s) to serve. Either a bare path, or a comma-separated list of
#: ``name=path`` pairs when serving the ablation variants side by side.
CHECKPOINT_ENV = "MELOCHRON_CHECKPOINT"
#: Deploy target has no GPU; cpu is the default so a benchmark taken here
#: describes production rather than the training box.
DEVICE_ENV = "MELOCHRON_DEVICE"
#: Serve an untrained model so the full request path can be exercised before a
#: real checkpoint exists. Output is meaningless and is labelled as such.
DEV_MODEL_ENV = "MELOCHRON_DEV_MODEL"

#: Torch defaults to one intra-op thread per core, which on a small deploy box
#: makes concurrent requests fight each other for the same cores and inflates
#: tail latency. Bounded explicitly; see ``configure_torch_threads``.
TORCH_THREADS_ENV = "MELOCHRON_TORCH_THREADS"


def file_fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    """Content-addressed id for an artifact file.

    Twelve hex characters of SHA-256. Long enough that a collision is not a
    practical concern for the handful of artifacts one project produces, short
    enough to read in a log line or a response body.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()[:12]


def configure_torch_threads() -> int:
    """Bound torch's intra-op parallelism, returning the value applied.

    Left at its default, torch sizes its thread pool to the whole machine. With
    several requests in flight each one spawns a full-width pool, the pools
    oversubscribe the cores, and p95 degrades far more than throughput improves.
    A small fixed pool gives a flatter, more predictable latency distribution,
    which is what the reported numbers are supposed to characterise.
    """
    requested = os.environ.get(TORCH_THREADS_ENV)
    threads = int(requested) if requested else max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    return threads


@dataclass
class LoadedModel:
    """One artifact, resident in memory, with the identity to describe it."""

    name: str
    version: str
    path: Path
    recommender: Recommender
    config: dict
    metrics: dict
    catalog_size: int
    load_seconds: float
    #: False when this is a synthesised development model rather than a
    #: trained artifact. Surfaced in every response that it touches, because
    #: untrained output that looks like trained output is the single most
    #: expensive kind of confusion this service could create.
    trained: bool = True

    def card(self) -> dict:
        """The model card returned by the API and stamped on every response."""
        return {
            "name": self.name,
            "version": self.version,
            "variant": self.config.get("variant", "id"),
            "trained": self.trained,
            "catalog_size": self.catalog_size,
            "max_len": self.config.get("max_len", 200),
            "d_model": self.config.get("d_model"),
            "use_time": self.config.get("use_time", True),
            "metrics": self.metrics,
            "load_seconds": round(self.load_seconds, 3),
        }


@dataclass
class ModelRegistry:
    """Every artifact this process is holding, plus which one is default."""

    models: dict[str, LoadedModel] = field(default_factory=dict)
    active: str | None = None
    #: Populated when startup could not load what it was asked to. The service
    #: still answers ``/api/health`` so the failure is visible to a probe
    #: rather than presenting as a crash loop with no explanation.
    error: str | None = None

    def __bool__(self) -> bool:
        return bool(self.models)

    def get(self, name: str | None = None) -> LoadedModel:
        """Resolve a model by name, defaulting to the active one."""
        if not self.models:
            raise LookupError(self.error or "no model artifact is loaded")
        key = name or self.active
        if key not in self.models:
            available = ", ".join(sorted(self.models)) or "none"
            raise LookupError(f"unknown model {key!r}; loaded: {available}")
        return self.models[key]

    def cards(self) -> list[dict]:
        return [m.card() for m in self.models.values()]


def _parse_spec(spec: str) -> list[tuple[str, Path]]:
    """Parse ``path`` or ``name=path,name=path`` into named artifact paths.

    The bare-path form is the common case and names the model ``default``. The
    named form exists for serving the ablation variants together, where the
    name is what the API caller selects with.
    """
    entries: list[tuple[str, Path]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, _, raw = chunk.partition("=")
            entries.append((name.strip(), Path(raw.strip())))
        else:
            entries.append(("default", Path(chunk)))
    return entries


def load_artifact(name: str, path: Path, device: str = "cpu") -> LoadedModel:
    """Read one checkpoint into a serving-ready model."""
    started = time.perf_counter()
    artifact = ckpt.load(path, device=device)
    recommender = Recommender(
        artifact.scorer,
        artifact.vocab,
        max_len=artifact.config.get("max_len", 200),
    )
    return LoadedModel(
        name=name,
        version=file_fingerprint(path),
        path=path,
        recommender=recommender,
        config=artifact.config,
        metrics=artifact.metrics,
        catalog_size=artifact.vocab.n_items,
        load_seconds=time.perf_counter() - started,
        trained=True,
    )


def development_model(n_items: int = 512, device: str = "cpu") -> LoadedModel:
    """An untrained model over a synthetic catalog, for exercising the API.

    Phase 5 is being built while the GPU is still occupied by Phase 2 and 3, so
    there is no trained checkpoint to load yet. Rather than leave the request
    path untestable until there is, this synthesises an artifact with the right
    shapes so routing, batching, upload handling and latency measurement can all
    be built and tested now.

    It is emphatically not a model. ``trained=False`` rides on every response it
    produces so that no screenshot of this can ever be mistaken for a result.
    """
    import numpy as np

    from melochron.data.vocab import FIRST_ITEM_ID, Vocab, canonical_key
    from melochron.models.scorer import build_scorer

    display = [("", ""), ("", "")]
    id_to_key = ["<pad>", "<oov>"]
    for i in range(n_items):
        artist, track = f"Artist {i % 64}", f"Track {i}"
        display.append((artist, track))
        id_to_key.append(canonical_key(artist, track))

    vocab = Vocab(
        key_to_id={k: i for i, k in enumerate(id_to_key)},
        id_to_key=id_to_key,
        counts=np.concatenate(
            [np.zeros(FIRST_ITEM_ID, dtype=np.int64), np.ones(n_items, dtype=np.int64)]
        ),
        display=display,
    )

    started = time.perf_counter()
    config = {"variant": "id", "d_model": 64, "max_len": 200, "use_time": True, "n_blocks": 2}
    _, _, scorer = build_scorer(
        n_items=len(vocab),
        device=device,
        variant="id",
        d_model=64,
        n_heads=2,
        n_blocks=2,
        max_len=200,
    )
    return LoadedModel(
        name="development",
        version="untrained",
        path=Path("<synthesised>"),
        recommender=Recommender(scorer, vocab, max_len=200),
        config=config,
        metrics={},
        catalog_size=vocab.n_items,
        load_seconds=time.perf_counter() - started,
        trained=False,
    )


def build_registry(
    spec: str | None = None,
    device: str | None = None,
    allow_development: bool | None = None,
) -> ModelRegistry:
    """Load everything named by the environment, tolerating total failure.

    A registry that failed to load is returned rather than raised, because the
    caller is application startup: raising here takes the process down and with
    it the health endpoint that would have explained why.
    """
    spec = spec if spec is not None else os.environ.get(CHECKPOINT_ENV, "")
    device = device or os.environ.get(DEVICE_ENV, "cpu")
    if allow_development is None:
        allow_development = os.environ.get(DEV_MODEL_ENV, "").lower() in {"1", "true", "yes"}

    registry = ModelRegistry()

    if spec.strip():
        failures = []
        for name, path in _parse_spec(spec):
            if not path.exists():
                failures.append(f"{name}: {path} does not exist")
                continue
            try:
                model = load_artifact(name, path, device=device)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                log.exception("failed to load artifact %s from %s", name, path)
                continue
            registry.models[name] = model
            log.info(
                "loaded model %s version=%s catalog=%d in %.2fs",
                name,
                model.version,
                model.catalog_size,
                model.load_seconds,
            )
        if failures:
            registry.error = "; ".join(failures)
    elif not allow_development:
        registry.error = (
            f"{CHECKPOINT_ENV} is unset. Point it at a trained artifact, or set "
            f"{DEV_MODEL_ENV}=1 to serve an untrained model for development."
        )

    if not registry.models and allow_development:
        model = development_model(device=device)
        registry.models[model.name] = model
        log.warning("serving an UNTRAINED development model; output is meaningless")

    if registry.models:
        registry.active = next(iter(registry.models))

    return registry
