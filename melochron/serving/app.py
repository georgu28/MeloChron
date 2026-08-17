"""The MeloChron serving application.

Phase 5 is deliberately narrow. BoilerVault already demonstrates a FastAPI
stack, so nothing here is included to prove that again; every route exists
because it is ML-specific and would not appear in an ordinary CRUD service:

* the artifact is **versioned and loaded once at startup**, and its identity
  rides on every response, so a recommendation can always be traced to the
  weights that produced it
* uploads are **asynchronous**, because a real streaming-history export is tens
  of megabytes and parsing it inside the request would hold the connection open
  for the whole parse
* scoring is **batched**, exposing the fact that the scorer already ranks the
  full catalog in one pass
* the **cold-start path is explicit**: coverage is measured and returned, and a
  weakly-grounded answer says so instead of looking identical to a confident one
* latency is **measured in the running service**, split into queueing and
  inference, and served as real numbers rather than quoted from a benchmark

Concurrency model. Torch inference is CPU-bound and blocking, so it must never
run on the event loop; every scoring call goes through a worker thread. A
semaphore then bounds how many run at once. Without that bound, five concurrent
requests each spawn a full-width intra-op thread pool, the pools oversubscribe
the cores, and p95 degrades much faster than throughput improves. The semaphore
converts that into an orderly queue, and the queueing time is measured
separately so the trade stays visible rather than being hidden inside one
aggregate number.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from melochron.serving import registry as registry_mod
from melochron.serving.jobs import JobState, JobStore
from melochron.serving.latency import LatencyRecorder, Stopwatch
from melochron.serving.registry import LoadedModel, ModelRegistry
from melochron.serving.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    Coverage,
    HealthResponse,
    ModelCard,
    RecommendationOut,
    RecommendRequest,
    RecommendResponse,
    UploadAccepted,
)
from melochron.serving.uploads import ParsedHistory, UploadError, history_key_set, parse_upload

log = logging.getLogger(__name__)

UPLOAD_DIR_ENV = "MELOCHRON_UPLOAD_DIR"
MAX_CONCURRENCY_ENV = "MELOCHRON_MAX_CONCURRENCY"
MAX_UPLOAD_MB_ENV = "MELOCHRON_MAX_UPLOAD_MB"

DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_UPLOAD_MB = 256

STATIC_DIR = Path(__file__).parent / "static"


def _parse_and_store(store: JobStore, job_id: str, upload_path: Path, workdir: Path) -> None:
    """Parse one upload. Runs on a worker thread, never on the event loop."""
    store.update(job_id, state=JobState.PARSING)
    started = time.perf_counter()
    try:
        parsed = parse_upload(upload_path, workdir)
    except UploadError as exc:
        # UploadError messages are written for the person who uploaded the
        # file, so they pass through verbatim.
        store.update(
            job_id,
            state=JobState.FAILED,
            error=str(exc),
            parse_seconds=time.perf_counter() - started,
        )
        return
    except Exception:
        log.exception("unexpected failure parsing job %s", job_id)
        store.update(
            job_id,
            state=JobState.FAILED,
            error="the file could not be processed. Check it is the export Spotify sent you.",
            parse_seconds=time.perf_counter() - started,
        )
        return

    store.update(
        job_id,
        state=JobState.READY,
        parsed=parsed,
        parse_seconds=time.perf_counter() - started,
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threads = registry_mod.configure_torch_threads()
        app.state.registry = registry_mod.build_registry()
        app.state.jobs = JobStore(Path(os.environ.get(UPLOAD_DIR_ENV, "data/uploads")))
        app.state.latency = LatencyRecorder()
        app.state.max_upload_bytes = (
            int(os.environ.get(MAX_UPLOAD_MB_ENV, DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024
        )
        app.state.inference_slots = asyncio.Semaphore(
            int(os.environ.get(MAX_CONCURRENCY_ENV, DEFAULT_MAX_CONCURRENCY))
        )

        reg: ModelRegistry = app.state.registry
        if reg:
            log.info("serving %d model(s), torch threads=%d", len(reg.models), threads)
        else:
            log.error("no model loaded: %s", reg.error)

        yield

    app = FastAPI(
        title="MeloChron",
        version="0.1.0",
        summary="Sequential music recommendation from listening history",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------- helpers

    def _model(name: str | None) -> LoadedModel:
        try:
            return app.state.registry.get(name)
        except LookupError as exc:
            # 503, not 500: the service is running correctly and simply has
            # nothing to serve with. That distinction is what lets an
            # orchestrator retry instead of alerting on a code bug.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _resolve_history(req) -> tuple[list[tuple[str, str, int]], set[str]]:
        """Get the history to score, from an upload job or from the request."""
        if req.job_id and req.history:
            raise HTTPException(400, "send either job_id or history, not both")

        if req.job_id:
            job = app.state.jobs.get(req.job_id)
            if job is None:
                raise HTTPException(404, f"no job {req.job_id!r}; it may have expired")
            if job.state is JobState.FAILED:
                raise HTTPException(409, job.error or "the upload failed to parse")
            if job.state is not JobState.READY:
                raise HTTPException(409, f"job is {job.state.value}; poll until it is ready")
            parsed: ParsedHistory = job.parsed
            return parsed.history, parsed.keys

        if req.history:
            triples = [(p.artist, p.track, p.ts) for p in req.history]
            return triples, history_key_set(triples)

        raise HTTPException(400, "provide either job_id or history")

    def _response(result, model: LoadedModel, keys: set[str], inference_ms: float):
        return RecommendResponse(
            items=[
                RecommendationOut(
                    item_id=r.item_id,
                    key=r.key,
                    artist=r.artist,
                    track=r.track,
                    score=r.score,
                    repeat=r.key in keys,
                )
                for r in result.items
            ],
            coverage=Coverage(
                coverage=result.coverage,
                history_length=result.history_length,
                matched=result.matched,
                cold_start=result.cold_start,
            ),
            model=ModelCard(**model.card()),
            inference_ms=round(inference_ms, 2),
        )

    # ----------------------------------------------------------------- routes

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness. Always 200, with the real state in the body.

        A degraded service still answers here on purpose: returning an error
        status would make an orchestrator restart a process whose problem is a
        missing artifact, which restarting cannot fix.
        """
        reg: ModelRegistry = app.state.registry
        return HealthResponse(
            status="ok" if reg else "degraded",
            models_loaded=len(reg.models),
            active_model=reg.active,
            error=reg.error,
        )

    @app.get("/api/models")
    async def models() -> dict:
        reg: ModelRegistry = app.state.registry
        return {"active": reg.active, "models": reg.cards(), "error": reg.error}

    @app.post("/api/upload", response_model=UploadAccepted, status_code=202)
    async def upload(
        background: BackgroundTasks, file: Annotated[UploadFile, File()]
    ) -> UploadAccepted:
        """Accept a history export and return immediately with a job id.

        202, not 200: the work has been accepted and has not been done. The
        client polls ``/api/jobs/{id}`` until the job is ready.
        """
        store: JobStore = app.state.jobs
        limit: int = app.state.max_upload_bytes

        # The client-supplied filename is metadata only. The bytes land at a
        # path this service chose, so a hostile name cannot steer the write.
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".zip", ".json"}:
            suffix = ".bin"

        job = store.create(filename=file.filename or "upload", size_bytes=0)
        dest = job.workdir / f"upload{suffix}"

        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(
                            413, f"file exceeds the {limit // (1024 * 1024)} MB upload limit"
                        )
                    out.write(chunk)
        except HTTPException:
            store.delete(job.id)
            raise

        if size == 0:
            store.delete(job.id)
            raise HTTPException(400, "the uploaded file was empty")

        store.update(job.id, size_bytes=size)
        background.add_task(_parse_and_store, store, job.id, dest, job.workdir)

        return UploadAccepted(
            job_id=job.id, state=JobState.QUEUED.value, filename=job.filename, size_bytes=size
        )

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id!r}; it may have expired")
        return job.status()

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def job_delete(job_id: str) -> None:
        app.state.jobs.delete(job_id)

    @app.post("/api/recommend", response_model=RecommendResponse)
    async def recommend(req: RecommendRequest) -> RecommendResponse:
        """Top-k next-track predictions for one history."""
        history, keys = _resolve_history(req)
        model = _model(req.model)
        latency: LatencyRecorder = app.state.latency

        with Stopwatch(latency, "request"):
            async with app.state.inference_slots:
                with Stopwatch(latency, "inference") as timer:
                    result = await run_in_threadpool(
                        model.recommender.recommend, history, req.k, req.exclude_history
                    )

        return _response(result, model, keys, timer.elapsed_ms)

    @app.post("/api/recommend/batch", response_model=BatchRecommendResponse)
    async def recommend_batch(req: BatchRecommendRequest) -> BatchRecommendResponse:
        """Score several histories in a single forward pass."""
        model = _model(req.model)
        latency: LatencyRecorder = app.state.latency

        histories = [[(p.artist, p.track, p.ts) for p in h] for h in req.histories]
        keysets = [history_key_set(h) for h in histories]

        with Stopwatch(latency, "request"):
            async with app.state.inference_slots:
                with Stopwatch(latency, "inference") as timer:
                    results = await run_in_threadpool(
                        model.recommender.recommend_batch,
                        histories,
                        req.k,
                        req.exclude_history,
                    )

        # Per-result inference time is the batch time shared out, and is
        # labelled as such rather than presented as if each was measured alone.
        share = timer.elapsed_ms / max(1, len(results))
        return BatchRecommendResponse(
            results=[_response(r, model, k, share) for r, k in zip(results, keysets)],
            inference_ms=round(timer.elapsed_ms, 2),
        )

    @app.get("/api/metrics/latency")
    async def metrics_latency() -> dict:
        """Live p50/p95 from this process.

        ``inference`` is the scoring call; ``request`` includes waiting for a
        slot. The gap between them is queueing, and it is the number that grows
        first under load.
        """
        latency: LatencyRecorder = app.state.latency
        return {
            **latency.snapshot(),
            "jobs": app.state.jobs.stats(),
            "max_concurrency": app.state.inference_slots._value,
        }

    # --------------------------------------------------------------- frontend

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
