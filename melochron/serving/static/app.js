/* MeloChron front end.
 *
 * No framework and no build step. The page has one flow --- upload, parse,
 * score, read --- and a dependency-free file that ships as-is keeps the
 * deployment story to "FastAPI serves a directory", which is the whole point
 * of a Phase 5 that refuses to re-prove it can build a web stack.
 *
 * Two behaviours are worth calling out because they are product decisions, not
 * implementation details:
 *
 * - "Hide tracks you have heard" re-requests with exclude_history rather than
 *   filtering the rendered list. Filtering locally would silently return fewer
 *   than k results; re-requesting returns a full k of genuinely novel tracks,
 *   which is what the label promises.
 * - Latency is read back from the server after every scoring call, so the
 *   figures in the footer are this session's real numbers.
 */

"use strict";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const state = {
  source: null,     // { job_id } or { history: [...] }
  model: null,
  maxUploadMb: 256,
  excludeHistory: false,
  ribbonCursor: -1,
  context: [],
};

/* ------------------------------------------------------------------ api */

function detailOf(body) {
  if (!body) return null;
  const d = body.detail;
  if (typeof d === "string") return d;
  // FastAPI validation errors arrive as a list of {loc, msg, type}.
  if (Array.isArray(d) && d.length) return d.map((e) => e.msg).join("; ");
  return null;
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* empty or non-JSON body; handled below */
  }
  if (!res.ok) throw new Error(detailOf(body) || `request failed (${res.status})`);
  return body;
}

/* ------------------------------------------------------------- chrome */

async function loadModel() {
  try {
    const health = await api("/api/health");
    if (health.max_upload_mb) state.maxUploadMb = health.max_upload_mb;

    const info = await api("/api/models");
    const card = info.models && info.models.length ? info.models[0] : null;
    state.model = card;

    $("health-dot").dataset.state = health.status === "ok" ? "ok" : "degraded";

    if (!card) {
      $("model-text").textContent = "no model loaded";
      return;
    }
    $("model-text").textContent =
      `${card.variant} · ${card.version} · ${card.catalog_size.toLocaleString()} items`;
    $("untrained").hidden = card.trained !== false;
  } catch (err) {
    $("health-dot").dataset.state = "degraded";
    $("model-text").textContent = "unreachable";
    showError(err.message);
  }
}

function showError(message) {
  const el = $("error");
  el.textContent = message;
  el.hidden = false;
}

function clearError() {
  $("error").hidden = true;
}

function setStep(name, value, meta) {
  const el = document.querySelector(`.step[data-step="${name}"]`);
  if (!el) return;
  el.dataset.state = value;
  const metaEl = $(`step-${name}-meta`);
  if (metaEl) metaEl.textContent = meta || "";
}

function beginSteps() {
  clearError();
  const steps = $("steps");
  steps.hidden = false;
  steps.setAttribute("aria-busy", "true");
  ["upload", "parse", "score"].forEach((n) => setStep(n, "", ""));
}

function endSteps() {
  $("steps").setAttribute("aria-busy", "false");
}

/* ------------------------------------------------------------- upload */

function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    // fetch() cannot report upload progress, so a large export would show no
    // movement for its entire transfer. XHR is the only way to get the bytes.
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        setStep("upload", "active", `${Math.round((e.loaded / e.total) * 100)}%`);
      }
    });
    xhr.addEventListener("load", () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* handled below */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(detailOf(body) || `upload failed (${xhr.status})`));
    });
    xhr.addEventListener("error", () =>
      reject(new Error("the upload could not reach the server"))
    );
    xhr.open("POST", "/api/upload");
    xhr.send(form);
  });
}

async function waitForJob(jobId) {
  const deadline = Date.now() + 5 * 60 * 1000;
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    if (job.state === "ready") return job;
    if (job.state === "failed") throw new Error(job.error || "the file could not be read");
    if (Date.now() > deadline) throw new Error("parsing took too long and was abandoned");
    await sleep(400);
  }
}

async function handleFile(file) {
  if (!file) return;
  if (file.size > state.maxUploadMb * 1024 * 1024) {
    showError(
      `That file is ${(file.size / 1024 / 1024).toFixed(0)} MB. The limit is ` +
        `${state.maxUploadMb} MB.`
    );
    return;
  }

  beginSteps();
  try {
    setStep("upload", "active", "0%");
    const accepted = await uploadFile(file);
    setStep("upload", "done", `${(file.size / 1024 / 1024).toFixed(1)} MB`);

    setStep("parse", "active", "");
    const job = await waitForJob(accepted.job_id);
    const plays = job.stats ? job.stats.plays_after_filter : 0;
    setStep("parse", "done", `${plays.toLocaleString()} plays`);

    await score({ job_id: job.job_id });
  } catch (err) {
    const active = document.querySelector('.step[data-state="active"]');
    if (active) active.dataset.state = "failed";
    showError(err.message);
  } finally {
    endSteps();
  }
}

/* -------------------------------------------------------------- scoring */

async function score(source) {
  state.source = source;
  setStep("score", "active", "");

  const payload = Object.assign(
    { k: 20, include_context: true, exclude_history: state.excludeHistory },
    source
  );

  const result = await api("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  setStep("score", "done", `${result.inference_ms} ms`);
  render(result);
  refreshLatency();
  return result;
}

async function rescore() {
  if (!state.source) return;
  try {
    clearError();
    await score(state.source);
  } catch (err) {
    showError(err.message);
  }
}

/* ------------------------------------------------------------ rendering */

function render(result) {
  $("results").hidden = false;
  $("latency").hidden = false;

  renderCoverage(result);
  renderRibbon(result.context || []);
  renderPredictions(result.items);

  $("fact-catalog").textContent = result.model.catalog_size.toLocaleString();
  $("fact-variant").textContent = result.model.variant;

  $("results").scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto"
      : "smooth",
    block: "start",
  });
}

function renderCoverage(result) {
  const cov = result.coverage;
  const pct = Math.round(cov.coverage * 100);

  $("cov-pct").textContent = `${pct}%`;
  $("cov-fill").style.width = `${pct}%`;
  $("cov-meter").dataset.cold = String(cov.cold_start);
  $("cov-meter").setAttribute(
    "aria-label",
    `Catalog coverage ${pct} percent: ${cov.matched} of ${cov.history_length} plays recognised`
  );

  $("fact-len").textContent = cov.history_length.toLocaleString();
  $("fact-matched").textContent = cov.matched.toLocaleString();

  const read = $("cov-read");
  if (cov.cold_start) {
    read.innerHTML =
      "<strong>Cold start.</strong> Most of this history sits outside the training " +
      "catalog, so the ranking leans on text embeddings rather than on learned " +
      "item behaviour. Treat it as a weaker signal.";
  } else if (pct >= 85) {
    read.textContent =
      "The model recognises nearly all of this history, so it is predicting " +
      "from learned item behaviour rather than from text alone.";
  } else {
    read.textContent =
      `${cov.history_length - cov.matched} of ${cov.history_length} plays are outside ` +
      "the catalog and were passed through as unknown, which keeps the gaps between " +
      "plays intact.";
  }
}

function humanGap(seconds) {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m`;
  if (seconds < 172800) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function renderRibbon(context) {
  const rib = $("ribbon");
  rib.innerHTML = "";
  state.context = context;
  state.ribbonCursor = -1;
  $("rib-tip").textContent = "";

  if (!context.length) {
    $("rib-count").textContent = "no plays to show";
    return;
  }

  const known = context.filter((p) => p.known).length;
  rib.setAttribute(
    "aria-label",
    `Timeline of the last ${context.length} plays, ${known} of them in the catalog. ` +
      "Use the arrow keys to step through."
  );
  rib.tabIndex = 0;

  context.forEach((play, i) => {
    const gap = i === 0 ? 0 : Math.max(0, play.ts - context[i - 1].ts);
    // Log scale: raw gaps span seconds to months, so a linear axis would
    // collapse a whole listening session into one invisible sliver next to a
    // single three-month break. The caption says the scale is log.
    const weight = Math.log1p(gap / 60) + 0.55;

    const seg = document.createElement("div");
    seg.className = "seg";
    seg.style.flexGrow = String(weight);
    seg.dataset.known = String(play.known);
    seg.dataset.index = String(i);
    seg.title = `${play.track} — ${play.artist}${i ? ` · +${humanGap(gap)}` : ""}`;
    rib.appendChild(seg);
  });

  $("rib-count").textContent = `${context.length} plays · ${known} in catalog`;
  $("rib-first").textContent = new Date(context[0].ts * 1000).toLocaleDateString();
  $("rib-last").textContent = new Date(
    context[context.length - 1].ts * 1000
  ).toLocaleDateString();
}

function describeSegment(i) {
  const play = state.context[i];
  if (!play) return;
  const gap = i === 0 ? null : Math.max(0, play.ts - state.context[i - 1].ts);
  $("rib-tip").textContent =
    `${i + 1}/${state.context.length}  ${play.track} — ${play.artist}` +
    `${gap === null ? "" : `  +${humanGap(gap)}`}` +
    `  ${play.known ? "in catalog" : "unknown"}`;
}

function renderPredictions(items) {
  const list = $("preds");
  list.innerHTML = "";

  if (!items.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No predictions came back for this history.";
    list.appendChild(li);
    return;
  }

  items.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = "pred";

    const rank = document.createElement("span");
    rank.className = "pred__rank";
    rank.textContent = String(i + 1).padStart(2, "0");

    const mark = document.createElement("span");
    mark.className = "pred__mark";
    mark.dataset.repeat = String(item.repeat);

    const names = document.createElement("span");
    names.className = "pred__names";
    const track = document.createElement("span");
    track.className = "pred__track";
    track.textContent = item.track || item.key;
    const artist = document.createElement("span");
    artist.className = "pred__artist";
    artist.textContent = item.artist || "unknown artist";
    names.append(track, artist);

    const tag = document.createElement("span");
    tag.className = "pred__tag";
    tag.textContent = item.repeat ? "heard" : "new";

    const scoreEl = document.createElement("span");
    scoreEl.className = "pred__score";
    scoreEl.textContent = item.score.toFixed(2);

    li.append(rank, mark, names, tag, scoreEl);
    list.appendChild(li);
  });
}

async function refreshLatency() {
  try {
    const m = await api("/api/metrics/latency");
    const inf = m.channels.inference || {};
    const req = m.channels.request || {};
    $("lat-p50").textContent = inf.p50_ms != null ? `${inf.p50_ms} ms` : "—";
    $("lat-p95").textContent = inf.p95_ms != null ? `${inf.p95_ms} ms` : "—";
    // Queueing is what the semaphore adds: total request time minus the
    // scoring call it wraps.
    const queue =
      req.p50_ms != null && inf.p50_ms != null
        ? Math.max(0, req.p50_ms - inf.p50_ms).toFixed(1)
        : null;
    $("lat-queue").textContent = queue != null ? `${queue} ms` : "—";
    $("lat-n").textContent = inf.total != null ? inf.total.toLocaleString() : "—";
  } catch {
    /* the footer is diagnostic; a failed poll must not disturb the results */
  }
}

/* --------------------------------------------------------------- events */

function wire() {
  const drop = $("drop");
  const input = $("file");

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });

  $("browse").addEventListener("click", (e) => {
    e.stopPropagation();
    input.click();
  });

  input.addEventListener("change", () => {
    handleFile(input.files[0]);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((type) =>
    drop.addEventListener(type, (e) => {
      e.preventDefault();
      drop.dataset.drag = "true";
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    drop.addEventListener(type, (e) => {
      e.preventDefault();
      drop.dataset.drag = "false";
    })
  );
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  $("demo").addEventListener("click", async () => {
    beginSteps();
    setStep("upload", "done", "sample");
    setStep("parse", "done", "");
    try {
      const sample = await api("/api/sample?n=40");
      await score({ history: sample.history });
    } catch (err) {
      setStep("score", "failed", "");
      showError(err.message);
    } finally {
      endSteps();
    }
  });

  $("demo-cold").addEventListener("click", async () => {
    beginSteps();
    setStep("upload", "done", "sample");
    setStep("parse", "done", "");
    const base = 1700000000;
    const history = Array.from({ length: 30 }, (_, i) => ({
      artist: `Unlisted Artist ${i}`,
      track: `Unreleased ${i}`,
      ts: base + i * 240,
    }));
    try {
      await score({ history });
    } catch (err) {
      setStep("score", "failed", "");
      showError(err.message);
    } finally {
      endSteps();
    }
  });

  $("novel-only").addEventListener("change", (e) => {
    state.excludeHistory = e.target.checked;
    rescore();
  });

  $("reset").addEventListener("click", () => {
    $("results").hidden = true;
    $("steps").hidden = true;
    state.source = null;
    clearError();
    $("intake").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // Arrow-key walk through the timeline. Without this the ribbon would be
  // mouse-only, and its detail is not repeated anywhere else on the page.
  $("ribbon").addEventListener("keydown", (e) => {
    if (!state.context.length) return;
    let next = state.ribbonCursor;
    if (e.key === "ArrowRight") next = Math.min(state.context.length - 1, next + 1);
    else if (e.key === "ArrowLeft") next = Math.max(0, next - 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = state.context.length - 1;
    else return;

    e.preventDefault();
    state.ribbonCursor = next;
    describeSegment(next);
  });

  $("ribbon").addEventListener("mouseover", (e) => {
    const seg = e.target.closest(".seg");
    if (seg) describeSegment(Number(seg.dataset.index));
  });
}

wire();
loadModel();
