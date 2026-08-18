"""End-to-end tests for the Phase 5 serving path.

These run against an untrained development model. That is the right call here:
what is under test is routing, upload handling, job lifecycle, the cold-start
signal, batching and latency accounting --- none of which depend on the weights
being good. Asserting on *which* tracks come back would be asserting on noise.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from melochron.serving.app import create_app
from melochron.serving.uploads import UploadError, safe_extract_zip

BASE_TS = 1_700_000_000


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MELOCHRON_DEV_MODEL", "1")
    monkeypatch.setenv("MELOCHRON_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("MELOCHRON_CHECKPOINT", raising=False)
    with TestClient(create_app()) as c:
        yield c


def known_pair(i: int) -> tuple[str, str]:
    """An item the development catalog contains.

    Mirrors how ``registry.development_model`` builds its vocabulary. The
    pairing matters: the artist is ``i % 64`` while the track is ``i``, so
    naming the artist any other way produces an out-of-vocabulary item and
    silently turns a coverage test into a cold-start one.
    """
    return f"Artist {i % 64}", f"Track {i}"


def extended_records(n: int = 40, known: bool = True) -> list[dict]:
    """Rows in the Spotify extended-history schema."""
    rows = []
    for i in range(n):
        artist, track = known_pair(i) if known else (f"Unheard Band {i}", f"Unreleased {i}")
        rows.append(
            {
                "ts": f"2023-11-{(i % 28) + 1:02d}T10:{i % 60:02d}:00Z",
                "master_metadata_album_artist_name": artist,
                "master_metadata_track_name": track,
                "ms_played": 210_000,
                "skipped": False,
                "shuffle": False,
                "offline": False,
            }
        )
    return rows


def make_zip(records: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MyData/Streaming_History_Audio_2023_1.json", json.dumps(records))
        zf.writestr("MyData/ReadMeFirst.pdf", b"%PDF-1.4 not history")
    return buf.getvalue()


def inline_history(n: int = 30, known: bool = True) -> list[dict]:
    rows = []
    for i in range(n):
        artist, track = known_pair(i) if known else (f"Unheard Band {i}", f"Unreleased {i}")
        rows.append({"artist": artist, "track": track, "ts": BASE_TS + i * 210})
    return rows


# ------------------------------------------------------------------ health


def test_health_reports_the_development_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 1


def test_missing_checkpoint_degrades_rather_than_crashing(tmp_path, monkeypatch):
    """No artifact must not take the process down, and must not look healthy."""
    monkeypatch.delenv("MELOCHRON_DEV_MODEL", raising=False)
    monkeypatch.setenv("MELOCHRON_CHECKPOINT", "")
    monkeypatch.setenv("MELOCHRON_UPLOAD_DIR", str(tmp_path / "uploads"))

    with TestClient(create_app()) as c:
        health = c.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"

        # Scoring is unavailable, and says so with 503 rather than 500.
        scored = c.post("/api/recommend", json={"history": inline_history()})
        assert scored.status_code == 503


def test_model_card_marks_the_dev_model_untrained(client):
    card = client.get("/api/models").json()["models"][0]
    assert card["trained"] is False
    assert card["catalog_size"] > 0


# ------------------------------------------------------------------ uploads


def test_upload_parses_and_becomes_scoreable(client):
    payload = make_zip(extended_records(40))
    accepted = client.post(
        "/api/upload", files={"file": ("my_spotify_data.zip", payload, "application/zip")}
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "ready"
    assert status["stats"]["plays_after_filter"] == 40
    assert status["stats"]["source"] == "spotify-extended"

    scored = client.post("/api/recommend", json={"job_id": job_id, "k": 5})
    assert scored.status_code == 200
    body = scored.json()
    assert len(body["items"]) == 5
    assert body["coverage"]["history_length"] == 40
    assert body["inference_ms"] > 0


def test_short_plays_are_filtered_like_training_does(client):
    """A skip-only export must fail loudly, not score as if it were listening."""
    records = extended_records(20)
    for row in records:
        row["ms_played"] = 4_000

    accepted = client.post(
        "/api/upload", files={"file": ("skips.zip", make_zip(records), "application/zip")}
    )
    status = client.get(f"/api/jobs/{accepted.json()['job_id']}").json()
    assert status["state"] == "failed"
    assert "30s" in status["error"]


def test_empty_upload_is_rejected(client):
    r = client.post("/api/upload", files={"file": ("empty.zip", b"", "application/zip")})
    assert r.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    r = client.post("/api/recommend", json={"job_id": "nope"})
    assert r.status_code == 404


# ------------------------------------------------------- archive robustness


def test_zip_traversal_is_refused(tmp_path):
    """A member escaping the extraction root must not be written."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.json", json.dumps(extended_records(2)))

    dest = tmp_path / "out"
    # The traversal components are stripped, so the write lands inside dest.
    written = safe_extract_zip(archive, dest)
    assert all(p.parent == dest for p in written)
    assert not (tmp_path.parent / "escaped.json").exists()


def test_archive_without_history_explains_itself(tmp_path):
    archive = tmp_path / "photos.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("cat.png", b"\x89PNG")

    with pytest.raises(UploadError, match="no JSON history"):
        safe_extract_zip(archive, tmp_path / "out")


def test_non_archive_upload_fails_cleanly(client):
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    status = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert status["state"] == "failed"
    assert status["error"]


# --------------------------------------------------------------- cold start


def test_unknown_catalog_is_flagged_cold_start(client):
    body = client.post(
        "/api/recommend", json={"history": inline_history(known=False), "k": 5}
    ).json()

    assert body["coverage"]["cold_start"] is True
    assert body["coverage"]["matched"] == 0
    # Still answers. Zero-shot is the ordinary path, not an error.
    assert len(body["items"]) == 5


def test_known_catalog_is_not_cold_start(client):
    body = client.post("/api/recommend", json={"history": inline_history(known=True)}).json()
    assert body["coverage"]["coverage"] == pytest.approx(1.0)
    assert body["coverage"]["cold_start"] is False


def test_empty_history_is_rejected(client):
    assert client.post("/api/recommend", json={}).status_code == 400


def test_job_and_history_together_is_rejected(client):
    r = client.post("/api/recommend", json={"job_id": "x", "history": inline_history(2)})
    assert r.status_code == 400


# ------------------------------------------------------------ repeat / novel


def test_repeat_flag_marks_tracks_already_in_the_history(client):
    """The repeat/novel split is the project's headline honesty result, so it
    has to be present per item, not only in the README."""
    body = client.post(
        "/api/recommend", json={"history": inline_history(known=True), "k": 50}
    ).json()
    assert any(item["repeat"] for item in body["items"])


def test_exclude_history_removes_every_repeat(client):
    body = client.post(
        "/api/recommend",
        json={"history": inline_history(known=True), "k": 50, "exclude_history": True},
    ).json()
    # exclude_history masks the window the model saw; nothing it returns may
    # come from that window.
    assert not any(item["repeat"] for item in body["items"])


# ------------------------------------------------------------------ batching


def test_batch_scores_every_history(client):
    r = client.post(
        "/api/recommend/batch",
        json={"histories": [inline_history(10), inline_history(12, known=False)], "k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["coverage"]["cold_start"] is False
    assert body["results"][1]["coverage"]["cold_start"] is True


def test_batch_is_bounded(client):
    r = client.post("/api/recommend/batch", json={"histories": [inline_history(2)] * 64})
    assert r.status_code == 422


def test_k_is_bounded(client):
    r = client.post("/api/recommend", json={"history": inline_history(5), "k": 10_000})
    assert r.status_code == 422


# ------------------------------------------------------------------ latency


def test_latency_separates_queueing_from_inference(client):
    for _ in range(5):
        client.post("/api/recommend", json={"history": inline_history(20)})

    metrics = client.get("/api/metrics/latency").json()
    channels = metrics["channels"]

    assert channels["inference"]["count"] == 5
    assert channels["request"]["count"] == 5
    assert channels["inference"]["p50_ms"] > 0
    # Request time contains inference time by construction.
    assert channels["request"]["p95_ms"] >= channels["inference"]["p50_ms"]


# ------------------------------------------------------- context / sample


def test_sample_history_is_fully_in_catalog(client):
    """The sample is the control the cold-start signal is read against, so it
    has to land at full coverage or it teaches the wrong thing."""
    sample = client.get("/api/sample?n=25").json()
    assert len(sample["history"]) == 25

    body = client.post("/api/recommend", json={"history": sample["history"]}).json()
    assert body["coverage"]["coverage"] == pytest.approx(1.0)
    assert body["coverage"]["cold_start"] is False


def test_context_is_omitted_unless_asked_for(client):
    body = client.post("/api/recommend", json={"history": inline_history(10)}).json()
    assert body["context"] is None


def test_context_returns_the_window_that_was_scored(client):
    history = inline_history(12)
    body = client.post("/api/recommend", json={"history": history, "include_context": True}).json()

    context = body["context"]
    assert len(context) == 12
    assert [c["ts"] for c in context] == sorted(c["ts"] for c in context)
    assert all(c["known"] for c in context)
    assert context[0]["track"] == history[0]["track"]


def test_context_marks_unknown_plays_individually(client):
    """Coverage says how much was recognised; context says which parts."""
    history = inline_history(6, known=True) + [
        {"artist": "Nobody At All", "track": "Untracked", "ts": BASE_TS + 9_000}
    ]
    context = client.post(
        "/api/recommend", json={"history": history, "include_context": True}
    ).json()["context"]

    assert [c["known"] for c in context] == [True] * 6 + [False]


def test_context_is_capped_at_the_model_window(client):
    """A long history must not return more context than the model read."""
    max_len = client.get("/api/models").json()["models"][0]["max_len"]
    body = client.post(
        "/api/recommend",
        json={"history": inline_history(max_len + 40), "include_context": True},
    ).json()

    assert len(body["context"]) == max_len
    assert body["coverage"]["history_length"] == max_len
