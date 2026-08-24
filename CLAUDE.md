# MeloChron — adoption prediction (agent handoff)

Predict whether a listener **returns to a track they just met** — binary
classification, one example per first encounter, on Music4All-Onion (253M events,
base rate ~0.31). Recurrence encodes real intent (unlike shuffle/autoplay next-track
labels). The encoder is reused; the task, labels, baselines, fixed cohort, head, and
audits are the work.

- **Full decision log & every result (incl. retractions):** [`process.md`](process.md) — read it first.
- **Results summary:** [`README.md`](README.md).

## Repo map
- `melochron/adoption/` — the project: `corpus, labels, cohort, baselines, metrics,
  model, windows, train, report, onion, slices, features`.
- `melochron/models/{sasrec, item_repr, time_encoding}.py` — the **reused encoder.
  Import, never edit** (its causal / no-look-ahead behaviour is guarded by the
  adoption tests; it is the transferable core).
- `melochron/net.py` — download helper. `scripts/` — 13 adoption CLIs.
- `tests/` — 110 tests, all green. `data/` (gitignored) — inputs;
  `artifacts/` (gitignored) — outputs (results live in README/process.md).

## Environment (WSL, RTX 4050 6 GB VRAM, 5.8 GB RAM)
- `source ~/melo-venv/bin/activate` (torch 2.13+cu126 + triton installed).
- Run from repo root with `export PYTHONPATH=$PWD` — the package is **not** pip-installed
  here; pytest works from root. (`requires-python>=3.11`; actually run on 3.12.)
- Prebuilt data under `data/interim/`: `onion-v1` (store), `onion-labels-v1`,
  `onion-features-v1` (`genres.npy`, `musicnn.npy`), `onion-cohort-v1`. If missing,
  they can be copied from the Windows build under `/mnt/c/Users/georg/.../MeloChron/`
  or rebuilt via the `scripts/*_onion.py` + `build_*` pipeline.

**Hardware gotchas (hard-won — see process.md):**
- Small RAM → **memmap the label columns** (`mmap_mode="r"`) or per-epoch window
  gathers thrash swap (10× slowdown, D-state, OOM risk). Already applied in the
  scorers/trainer.
- `torch.compile` buys only ~1.3× (model is compute-bound, not launch-bound); batch
  > 1024 collapses (VRAM cliff); **`max_len` is the real speed lever**.
- Hybrid item-rep **NaNs under fp16 AMP** (L2-normalize + log_scale); use the
  residual-free `text_frozen` variant instead.
- Training runs take **45 min–4.5 h**. Foreground Bash caps at 600 s → run
  backgrounded: `nohup python -u scripts/… > artifacts/adoption/<log> 2>&1 &`, then
  monitor the log (tqdm bar writes `\r` frames; `tr '\r' '\n'` to read).

## How to run
```bash
python scripts/train_adoption.py --compile --heads pure priors   # model + baselines, one table
python scripts/score_adoption.py --model NAME PATH … --dump artifacts/adoption/cohort-scores.npz
python scripts/adjudicate_coldstart.py   # the in-context-rate adjudication
python scripts/demo_adoption.py          # per-user predictions vs what happened
python -m pytest -q                      # 110 tests
```

## Conventions / guardrails (non-negotiable)
- **Baselines before the model.** Every PR-AUC is quoted **with its base rate**.
- **Bootstrap over USERS, not rows.** The honest win test is the *paired* difference
  CI: `metrics.paired_delta_pr_auc` (a 95% interval clear of 0). Never fabricate a
  metric/column/file.
- **No look-ahead is sacred:** an encounter never sees an event at/after itself;
  `baselines.incontext_user_rate` is resolution-gated and unit-tested — reuse it, do
  not recompute a second way.
- **Same-cohort invariant:** every column scores the saved `onion-cohort-v1` rows
  (assert the index).
- **Commits:** one-line conventional (`feat(adoption): …`, `docs: …`), **no AI /
  co-author trailers**, granular.
- When you start a training run, hand the user a `tail -f <log>` command.

## Current state / open items
- **In flight:** `scripts/train_seq_over_incontext.py` — the terminal experiment
  *"does the sequence add anything OVER the in-context rate?"* (concat done, residual
  training). Check `artifacts/adoption/phase4-seq-over-incontext.log`. When it lands,
  report the paired Δ (`seq+incontext − incontext-alone`) and replace the "pending"
  line in `README.md` + `process.md`.
- **Open:** rename the package `melochron` → `melo` (user-approved intent; mechanical
  `melochron→melo` replace + `pytest`). Phase 5 finalisation (demo/README) after the
  experiment.
- **Headline so far:** the training-free `incontext-user-rate` (PR-AUC **0.4212** @
  base 0.3079) is the bar to beat; the best *model* is audio content
  `text_frozen(musicnn)` (**0.4139**), which statistically **ties** it. "Content is
  weak" was **retracted** (method artifact). The next-track ranking project that once
  shared this repo was removed (git history at `82d4cbf`).
```
