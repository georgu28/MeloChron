# MeloChron — session handoff

You are picking up a finished ML research project with a polished web write-up. Two jobs remain: **(1) build the predicted-vs-actual demo** and fold it into the write-up, and **(2) help the user deploy everything and think through moving training + a live demo to the cloud** (they want the cloud work partly for resume signal). Read this whole file first.

## What the project is

Predicts whether a listener **returns to a track they just met** (binary, one row per first encounter) on **Music4All-Onion** (253M events, 119k listeners, base rate 0.3079). Metric is PR-AUC. The finished story:

- A training-free **in-context running rate** (each listener's own prior return rate, no look-ahead) is the bar: **PR-AUC 0.4212**.
- The best model is a **content sequence model + the rate as a fixed residual base**: **0.4520**, paired user-bootstrap **+0.031 [+0.023, +0.039]**, and it wins every slice (all four slices significance-tested this session: all +0.031, cold_user +0.036, unfamiliar +0.035, new_neighborhood +0.033, every CI clear of 0).
- ID embeddings overfit under temporal drift; content (frozen musicnn audio) generalizes. Adding lyrics+era to audio gave **no** gain (0.4486). Two claims were retracted mid-project ("content is weak", "encoder beats baselines on cold users").

Full decision log: `process.md`. Results summary: `README.md`. Project guide: `CLAUDE.md` (now gitignored but present locally).

## The web write-up (the artifact)

- **Live (private, user-owned):** https://claude.ai/code/artifact/a6a89198-3f6a-4832-8004-9624262198a3
- **Source:** `demo/index.html` (a single self-contained HTML file, no external assets, no build step).
- **CRITICAL — to update the SAME artifact from this new session, you MUST pass `url: "https://claude.ai/code/artifact/a6a89198-3f6a-4832-8004-9624262198a3"` to the Artifact tool.** Publishing `demo/index.html` without that `url` creates a *separate* artifact. Keep `favicon: "🔁"` stable.
- **Structure (research arc), each a `<section id>` with `<h2 class="section-title">`, tracked by a left sticky TOC:** The Question, The model (has an SVG architecture diagram + real param counts), Baselines, Iteration (the ID→content→content+rate→lyrics story, with the forest-plot SVG), Results, Roadblocks, Discussion, Limitations, Conclusion.
- **Hard style constraints the user enforced (do not violate):** system-ui sans-serif only (no serif, no Google Fonts); **no em/en dashes** anywhere; **no gray body text** (body/prose uses `--text`; gray `--faint` only for tiny labels, table headers, SVG ticks); plain first-person human voice. The **humanizer skill** is installed as `humanizer:humanizer` — run it on any new prose. Also `artifact-design` should be loaded before design changes.
- The page is theme-aware (light/dark tokens + a toggle) and the TOC scroll-spy auto-includes any new `.wrap section[id]`, so a new demo section just needs an `id`, a `.section-title`, and a TOC `<li>`.

## Git / how to save + land work

- `origin/main` = `978cffd` — all code + artifact v1. This is canonical; the user is solo and wants main-only.
- Latest artifact is `f6514f8` on branch **`docs-terminal`** (a worktree at `.claude/worktrees/docs-terminal`). To land it on main: `git checkout main && git cherry-pick f6514f8 && git push origin main` (the user runs pushes to main; you do not push to main).
- Do your demo work in the `docs-terminal` worktree (or a fresh worktree off main), commit there, and hand the user a one-line cherry-pick/merge. Never push to main/master or force-push.
- Remote: `github.com/georgu28/MeloChron`. `feat/adoption-onion` was deleted; the only stray is possibly a `docs-terminal-result` worktree (safe to `git worktree remove`).

## Environment

- WSL, **RTX 4050 6 GB VRAM, 5.8 GB RAM** (tight). venv: `source ~/melo-venv/bin/activate` (torch 2.13+cu126). Run from repo root with `export PYTHONPATH=$PWD` (package not pip-installed). Or call the interpreter directly: `/home/georgu/melo-venv/bin/python`.
- **Memory gotcha (hard-won):** long training thrashes swap if RAM is squeezed. Root cause we hit: orphaned python children from `nohup ... &` (kill by name with `pkill -9 -f <script>`, not just `$!`), and VS Code/Pylance eating RAM. Launch long runs with `setsid nohup python -u ... > log 2>&1 < /dev/null &`; kill the whole tree with `pkill -f`. Watch tqdm via `tail -f log | tr '\r' '\n'`.
- Data (gitignored) under `data/interim/`: `onion-v1` (corpus store), `onion-labels-v1`, `onion-features-v1` (`genres.npy`, `musicnn.npy`, plus `lyrics_minilm.npy`, `musicnn_lyrics_era.npy` built this session), `onion-cohort-v1` (the fixed 500,001-row eval cohort). Raw Music4All extracted at `data/raw/music4all/music4all/` (lyrics/, audios/, id_information.csv, id_metadata.csv, listening_history.csv, etc.).
- **Best-model checkpoint:** `artifacts/adoption/runs-content-incontext-residual/residual/best.pt` (content+incontext residual, item_variant `text_frozen`, musicnn; self-contained — `melochron.adoption.train.load_checkpoint` rebuilds it with no feature file needed).

## TASK 1 — build the predicted-vs-actual demo

Goal (the "Still to come" line in the Conclusion): a few **real listeners**, the tracks they met for the first time, the **best model's** predicted P(return), and whether they actually returned. Show real artist/track names, not opaque IDs.

Recipe:
1. **Generate the best model's per-encounter predictions on the cohort.** Mirror the scoring path in `scripts/train_seq_over_incontext.py` (it is the exact template): load `CompactCorpus`, `EncounterTable`, `event_horizon`, `temporal_split`, the cohort rows; `baselines.fit_priors`; compute the in-context column with `baselines.incontext_user_rate(..., test_pool, rows, prior=priors.global_rate, pseudocount=priors.user_pseudocount)` where `test_pool = split.is_test & horizon.observable & (encounter_ts >= PLAUSIBLE_FLOOR)` and `resolution_pos = encounter_pos + event_n` with `resolution_pos[label] = recur_pos[label]`; then `model,_ = train.load_checkpoint(best.pt, device)`, build `train.Examples(users, positions, candidates, labels)` with `.priors = incontext[:,None].astype(float32)`, and `train.predict(model, corpus, examples, model.config["max_len"], device)`. (A working one-off of exactly this ran this session to compute per-slice CIs; the overall PR-AUC reproduced 0.4520 as a sanity check — do the same check.)
2. **Pick 3 real users spanning archetypes.** See `scripts/demo_adoption.py::pick_users` (heavy user, cold user with no history, new-neighborhood listener). Reuse the slice masks from `report.cohort_slices`.
3. **Join real names.** `data/interim/onion-v1/tracks.npy[track_code]` gives the 16-char Music4All track id; `data/raw/music4all/music4all/id_information.csv` maps `id -> artist, song` (tab-separated). The onion ids ARE Music4All ids (verified 100% overlap this session).
4. **Add a section** to `demo/index.html`: `<section id="demo"><h2 class="section-title">Demo</h2>...`, add `<li><a href="#demo">Demo</a></li>` to the `.toc`, and per user show archetype + their adoption rate + a small table (artist — song, predicted P, returned yes/no). Respect every style constraint above (sans-only, no dashes, no gray body). Then republish with the `url` above.
5. Commit on `docs-terminal`, hand the user the cherry-pick.

Note honestly in the demo that per-listener separation is modest (the aggregate lift is ~1.2–1.4×); this is a real but not strong signal.

## TASK 2 — deployment

- The write-up is one static self-contained `demo/index.html`. Easiest public hosting: **GitHub Pages** (repo already on GitHub), or Vercel/Netlify/Cloudflare Pages. For a resume the user wants a public URL (the claude.ai artifact is private/Claude-hosted). This is a static deploy, no backend.
- A **Railway** MCP + `use-railway` skill are available in this environment if the user wants to deploy there.

## TASK 3 — cloud for training + deploying (the user wants to discuss this)

The user wants cloud both for speed/capacity and for resume signal. Help them decide; do not just pick.

- **Training in the cloud.** Current training is capped by the 6 GB laptop GPU (reduced config: 15k users, history len 100; `torch.compile` only ~1.3×; batch >1024 spills VRAM; the real lever is `max_len`). A cloud GPU (Colab Pro, Lambda, RunPod, or a cloud VM with a T4/A10/A100) would allow the planned full config (30k users, len 200), remove the swap-thrash class of problems, and be a clean resume story (training ML on cloud GPUs, ideally containerized + reproducible). Good scope: a `Dockerfile` + a training entrypoint + a cloud GPU run of the headline config, results folded back into `process.md`/README.
- **Deploying a live demo.** The static write-up is trivial. A *live inference* demo (import a listener's history → predict returns) is the stronger resume signal but needs a backend (FastAPI on Cloud Run / Railway / a serverless function) plus a feature pipeline. Real caveats already worked out this session: Spotify deprecated audio-features/preview endpoints (~Nov 2024), and Music4All's catalog tops out at 2019, so a modern listener's tracks mostly miss the feature join — **the in-context rate is the portable part** (computable from any imported history), and the Music4All `spotify_id` join covers the catalog overlap (~50% for mainstream listeners). Frame any live demo as illustrative/out-of-distribution, not calibrated.
- Suggested next step: help the user weigh (a) cloud training run at full config, (b) static public hosting of the write-up, (c) a small live inference service, by effort vs resume payoff, then execute whichever they pick.

## Guardrails carried over
- Baselines before model; every PR-AUC quoted with its base rate; wins are paired user-bootstrap CIs clear of 0 (`metrics.paired_delta_pr_auc`); no look-ahead is sacred; never fabricate a metric/file. Commits are one-line conventional, no AI/co-author trailers. The encoder in `melochron/models/` is imported, never edited.
