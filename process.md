# Adoption prediction: process & decision log

A running record of the MeloChron adoption effort: the plan, every decision and its
reasoning, the results, and what those results mean. Written for review. The spine is
the phases in build order (0 to 5), all of them now done, ending in a web write-up
deployed at https://melochron.vercel.app. Cross-cutting engineering (hardware,
`torch.compile`, the swap fix, the scorer) sits inside Phase 3, where it happened.

> Convention: every PR-AUC is quoted **with its base rate**, a precision-recall
> number is meaningless without it. Base rates differ by slice because the label
> rate genuinely differs by slice.

---

## Overview: what this is, and why a rebuild

**Task.** Given a track a user encounters for the **first time**, predict whether
they **return** to it within a horizon. Per-(user, track) **binary
classification**, one example per first encounter, *not* ranking.

**Why rebuild the project around this.** The finished next-track recommender
(SASRec encoder, serving app, insights, everything outside `melochron/adoption/`)
learns from next-track labels that were mostly shuffle/autoplay output: the queue
advanced, the user didn't choose. Recurrence is unambiguously the user's
decision, so an adoption label encodes real intent. The encoder is **reused**;
the task, labels, head, and evaluation are new.

**Dataset.** Music4All-Onion (Zenodo 15394646, CC-BY-4.0). 252,984,396 events,
119,140 users, 56,512 tracks, 50,016,042 distinct (user, track) pairs. **No
artist metadata**, all 46 files are track-id-keyed. That single fact reshaped
the brief: the artist-affinity baseline and new-artist slice became an
item-adoption-rate baseline and tag/genre novelty instead.

Two deliverables, both now delivered. (1) Aggregate PR-AUC per slice against the base
rate (the Results, below). (2) A prediction demo: real listeners, their first-encountered
tracks, the model's predicted return probability, and what actually happened, folded into
a web write-up (Phase 5).

---

## Data and citation

This project uses Music4All-Onion, which extends the Music4All database. Both require
citation, and Music4All is provided for non-commercial research use only. Neither dataset
is redistributed in this repository; everything under `data/` is gitignored.

- Music4All-Onion (Zenodo 15394646, CC BY 4.0): Marta Moscati, Emilia Parada-Cabaleiro,
  Yashar Deldjoo, Eva Zangerle, and Markus Schedl. Music4All-Onion: A Large-Scale
  Multi-faceted Content-Centric Music Recommendation Dataset. CIKM 2022, pages 4339 to
  4343. https://doi.org/10.1145/3511808.3557656
- Music4All (base dataset, research-use agreement): Igor André Pegoraro Santana, Fabio
  Pinhelli, Juliano Donini, Leonardo Catharin, Rafael Biazus Mangolin, Yandre Maldonado e
  Gomes da Costa, Valéria Delisandra Feltrim, and Marcos Aurélio Domingues. Music4All: A
  New Music Database and its Applications. IWSSIP 2020, pages 399 to 404.

---

## Repository scope: the next-track project was removed

This repo began as a *next-track* recommender (a SASRec ranking model, serving app,
transfer/insights work). The adoption effort reused its encoder but is a distinct
task, and keeping both made the repo confusing to read. The next-track code, its
scripts, tests, configs, and README were therefore **removed** (commit
`chore: remove the next-track project, keeping the reused encoder`), recoverable
in full from git history at commit `82d4cbf` and earlier.

**Kept, because adoption reuses it:** the encoder,
`melochron/models/{sasrec, item_repr, time_encoding}.py`, plus `melochron/net.py`
(download helper). Everything else under `melochron/` is now `adoption/`. What the
next-track project established, in one line: the transferable value is the encoder,
and its headline finding (a frozen-text-plus-residual *hybrid* item representation)
is exactly what the adoption **audio-features** section here re-tested and partly
overturned.

---

## The plan (roadmap, phases 0-5)

| Phase | Goal | Output | Status |
|---|---|---|---|
| **0** | Parse 253M events into a sorted, mmap-able compact corpus | `onion-v1` store | ✅ done |
| **1** | Build first-encounter labels (event & time horizons), temporal split, no-look-ahead rule, slices | `onion-labels-v1` | ✅ done |
| **2** | Baselines (fit on train only) + fixed evaluation cohort + scoring harness (PR-AUC/lift/AUROC, user-bootstrap) | `onion-cohort-v1`, baseline table | ✅ done |
| **3** | The model: reused SASRec encoder + binary head, two variants (pure / priors); score beside baselines on the fixed cohort | `phase3-model.md`, checkpoints | ✅ done |
| **4** | Ablations, each a matched pair through the same cohort: pure-vs-priors · max_len 50/100/200 · time on/off · ID-vs-genre-hybrid (+cold-item) · event-horizon N sweep | Phase-4 table | ✅ done |
| **5** | The demo (real listeners, their first encounters, predicted probability, what happened) + a web write-up | `demo/index.html`, deployed | ✅ done |

**Governing rules (from the brief, non-negotiable):** baselines before model ·
PR-AUC always with base rate · never fabricate a metric/column/file · the
no-look-ahead invariant is sacred (an encounter never sees an event at/after
itself) · never touch anything outside `melochron/adoption/` (the encoder,
`sasrec.py`/`time_encoding.py`/`item_repr.py`, are imported, never edited).

---

## Phase 0: the compact corpus  ✅

**Plan:** turn the 253M-event archive into a store the label builder can slice
cheaply, with counts verified against published figures.

**Decisions & findings:**
- **A sorted int32 columnar store, not a pandas frame.** Three parallel arrays
  (`user_code`, `track_code`, `ts`) sorted by (user, ts) + per-user offsets,
  ~3 GB, memory-mappable. Routing through `schema.py` was rejected: it requires an
  `artist` column this dataset lacks and would drop >50% of rows.
- **Source is reverse-chronological** per user (Last.fm API order). Inheriting it
  would invert *every* encounter, so the build sorts by (user, ts) and
  `validate()` enforces ascending time within each user. Load-bearing: the whole
  label depends on event order.
- **Timestamps are datetime strings, not integers.** Converting via an assumed
  resolution (`//1e9`) would silently land everything in 1970. One genuinely
  corrupt 1970 row was found and dropped.
- Counts match published figures exactly and cross-check the 50,016,042-pair file.
  The ≥20-plays catalog cutoff is a **no-op** (least-played track already has 52
  plays).

**Meaning:** the store is trustworthy and order-correct, the foundation every
later number rests on.

---

## Phase 1: the labels  ✅

**Plan:** define "adoption" precisely, with horizons, censoring accounting, a
temporal train/test split, and a no-look-ahead rule.

**Decisions & findings:**
- **Two horizons, side by side.** Event horizon **N=200** (base rate 0.3592, 12.1%
  censored); time horizon **30d** (0.3018, 2% censored). Unbounded recurrence
  ceiling **54.1%**, no labelled rate may exceed it.
- **The base rate is 30-36%, not the 8-15% the brief assumed.** This falsifies the
  premise the brief used to ban AUROC (see Phase 2). Recorded, not accepted.
- **Drift is real and matters.** Base rate falls 0.4464 (2005) → 0.2798 (2020);
  across the split **TRAIN 0.3676 vs TEST 0.3124**. *Every train-fitted prior is
  miscalibrated on test by construction.* Later explains the epoch-0 peak.
- **No look-ahead across the split.** A training encounter whose horizon crosses
  the split boundary is **excluded**, otherwise it's labelled with post-cutoff
  events no deployment could see. Churn dominates time-censoring (836K rows) over
  corpus truncation (152K).
- **Trap noted:** early encounters adopt at 0.5486, a model could cheat by
  learning "is this user new"; the `cold_user` and ordinal-band slices expose it.

**Meaning:** the label encodes real intent, the split is honest, and the known
drift is a measured quantity we must interpret results against, not a surprise.

---

## Phase 2: baselines and the scoring harness  ✅

**Plan:** build baselines *hard enough to be worth beating*, plus one fixed
evaluation cohort and one scoring path every model shares.

**Decisions & findings:**
- **Genres, not tags, as the item vectors.** Genres: 100% catalog coverage, flat
  across popularity. Tags rejected for slicing, 80.7% coverage but collapses
  38% → 99.9% from obscure to popular tracks, which would make the discovery slice
  least reliable exactly where discovery lives.
- **A fixed evaluation cohort:** 500,001 rows, whole users, every model scores
  these **exact** rows. Whole users, because a per-user rate from a fraction of
  someone's encounters isn't the rate a deployment holds.
- **PR-AUC headline + base rate + lift; AUROC secondary** (the brief's AUROC ban
  rests on a rarity premise Phase 1 falsified; reported with the reason recorded).
- **Bootstrap over USERS, not rows**, encounters within a user aren't independent
  draws; row-bootstrap CIs would be several times too narrow.
- **Baselines, the bar (lift over base 0.3079; PR-AUC in parens):**
  - overall: user-prior 1.232 (.3794) · user×item 1.227 (.3777) · genre-sim 1.067
  - unfamiliar: user×item 1.184 (.3385) · genre-sim ≈ chance (content weak alone)
  - cold_user: user-prior 1.000 (structural) · user×item 1.078 (.3125)
- **Self-checks pass:** global-prior PR-AUC == base rate exactly; cold users fall
  back to the global prior.

**Meaning:** `user × item` (two shrunk scalar rates) is a genuinely strong
adversary. The model has to add something *those two numbers can't*, and only on
the slices where they're weak (cold_user, unfamiliar) does beating them mean
anything.

---

## Phase 3: the model  ✅ (this session's headline)

**Plan:** reuse `SASRec.encode_last` unchanged for the history vector; add a
**new** binary head, encounter-anchored windows, and BCE loop (the ranking
machinery in `train/` doesn't fit binary classification). Two heads on **one**
encoder, **pure-sequence** (`[h, c, h⊙c]`, must recover the user rate from the
sequence) and **sequence+priors** (also fed `user-prior` + `item-rate`, learns the
residual). Their gap *is* the experiment. **Pass/fail bar:** beat `user × item` on
**cold_user AND unfamiliar**, not just overall.

### 3a. Environment & hardware: why the config is what it is
- **Machine:** WSL Ubuntu, **RTX 4050 Laptop (6 GB VRAM)**, **5.8 GB RAM + 2 GB
  swap**. The binding constraint on everything below.
- **Setup:** adoption branch pulled into a Linux clone; gitignored interim data
  copied from the Windows build onto ext4; venv torch 2.13+cu126 / Triton 3.7;
  **261 tests pass**.

### 3b. `torch.compile`: kept, but not the lever we hoped
5k-user / 1-epoch A/B: eager **819 s/epoch** → compiled **638 s/epoch** = **~1.3x**
(incl. warmup), no graph breaks. A throughput sweep showed **batch size is a dead
end**, throughput *collapses* above batch 1024 at len 200 (3747 → 406 enc/s) when
the working set spills the 6 GB card.
**Meaning:** the model is **compute-bound on this small GPU, not
launch-bound**, so fusion can't reach 2-4x (the hand-written time-interval
attention doesn't hit tensor cores). The only science-safe speed lever is
**`max_len`** (already a planned ablation); the real 2-4x would need Flash/SDPA in
the frozen `sasrec.py`.

### 3c. Reduced config, and the swap thrash + mmap fix
- **Config: 15k users, len 100** (not 30k/len 200), forced by RAM. The cohort is
  fixed, so the comparison stays exact; only encoder fit changes. The Phase-4
  `max_len` sweep tests whether len 100 changed conclusions.
- **The thrash:** the `EncounterTable` was loaded **resident** (~1.2 GB
  anonymous), starving the memmapped corpus of page cache → window gathers
  page-faulted from disk, anonymous arrays hit a full swap. Process `D`-state, GPU
  0%, **53 min/epoch**, OOM risk.
- **Fix:** mmap the `EncounterTable` columns (train + scorer). The loop copies only
  needed rows into `Examples`, so paging costs nothing per epoch. **Result:** `D`
  → `R`, GPU ~88%, **~12 min/epoch (~4-5x)**, swap flat.

### 3d. A standalone scorer + a paired-difference bootstrap
- `scripts/score_adoption.py` (new): scores checkpoints on the fixed cohort beside
  the baselines through the *same* `report.py` path, identical in construction to
  the trainer's table, without retraining. (The wedged run never reached its own
  scoring; Phase 5 needs checkpoint-loading anyway.)
- **Paired-difference CI:** the harness's *marginal* CIs overlap, which is a
  conservative, slightly wrong test, columns are heavily correlated across users.
  The honest test resamples whole users and takes Δ = model − `user×item` per
  round; a 95% interval excluding 0 is a real win.
- A **tqdm progress bar** was added to the trainer for the many Phase-4 runs; tests
  stay green.

### 3e. Results
Cohort base rate **0.3079**. Baselines reproduced Phase 2 exactly.

| slice | base | user×item | model (pure) | model (priors) |
|---|---|---|---|---|
| all | 0.3079 | 0.3777 | 0.3564 | **0.3895** |
| cold_user | 0.2898 | 0.3125 | **0.3338** | 0.3218 |
| unfamiliar | 0.2860 | 0.3385 | 0.3266 | **0.3446** |

**Paired Δ PR-AUC vs `user × item`** (`*` = 95% CI excludes 0):

| slice | pure − user×item | priors − user×item |
|---|---|---|
| all | −0.0215 [−0.037, −0.007] **\* (worse)** | **+0.0115 [+0.008, +0.015] \*** |
| cold_user | **+0.0216 [+0.006, +0.040] \*** | +0.0095 [−0.004, +0.024] ns |
| unfamiliar | −0.0124 [−0.028, +0.002] ns | **+0.0060 [+0.001, +0.012] \*** |

Files: `artifacts/adoption/phase3-model.md` (full table), `runs/adoption-*/best.pt`
(checkpoints), `cohort-scores.npz` (per-row scores for Phase 5).

### 3f. What the numbers mean (the honest verdict)
- ⚑ **RETRACTED (see Cold-start adjudication).** This was read as "the encoder
  earns its keep on cold users": pure beats `user×item` on `cold_user` by
  **+0.022\*.** True but misleading, `user×item` is *blind* on held-out users. A
  training-free in-context running rate beats **both** heads on cold_user
  (id-pure −0.038\*, id-priors −0.050\*), so the edge was partial rate recovery,
  not adoption understanding.
- ✅ **The priors head is the best overall model**, significantly beating the
  strongest baseline **overall (+0.012\*)** and on **unfamiliar (+0.006\*)**,
  though that discovery win is *small*.
- ⚠️ **No single head clears *both* required slices significantly.** Pure owns
  cold_user (sig) but is worse overall; priors owns unfamiliar+overall (sig) but
  its cold_user edge is directional, not significant. The pass is **split across
  the two heads**, an honest *partial* pass, not a clean sweep.

### 3g. Why did validation peak at epoch 0? (documented per request)
**A yellow flag, not a red one, and close to expected for this model class.**
- **"Epoch 0" = one full pass = ~12,600 updates.** Not undertrained; one pass
  simply generalizes best and a second makes it worse.
- **Signature is textbook overfitting**, train loss keeps falling while val falls
  with it:
  ```
  pure:    loss 0.634→0.605→0.581→0.561   val 0.412→0.412→0.406→0.400
  priors:  loss 0.616→0.597→0.578→0.558   val 0.494→0.483→0.460→0.460
  ```
- **Drivers (in likely order):** (1) enormous ID-embedding capacity (56,512×128 ≈
  7.2M params, most items seen a handful of times → memorizes fast; industrial ID
  recsys often train ~1 epoch on purpose, single-epoch-optimal is *normal*);
  (2) known temporal drift makes the late-train val slice a moving target;
  (3) untuned dropout/wd/lr.
- **Not alarming:** early stopping keeps the epoch-0 weights, which beat the
  baselines. Red flags (val ≈ base rate, val < baseline every epoch, divergence)
  are absent.
- **The real risk it points to:** the numbers may be a **conservative floor**,
  better regularization could peak *later and higher*, making the model look
  **better** vs baselines (esp. the pure head's weak overall).
- **Open diagnostic (deferred):** validate several times *within* epoch 0 + one
  stronger-reg setting. If val still rises at epoch 0's end, the numbers stand; if
  it peaks partway, re-tune before trusting them. Deferred to proceed to Phase 4.

---

## Phase 4: ablations  ✅ done

Each ablation is a **matched pair** through the same fixed cohort and metrics,
added as columns to a Phase-4 table. `pure vs priors` is **already done** (§3e).
Ordered for insight-per-GPU-hour (each run ~45 min with the mmap fix + early stop):

1. **max_len 50 / 100 / 200**, *does the forced len-100 config change any
   conclusion?* Top priority (Phase 3 was forced off the plan's len 200). Priors
   head at all three (re-run len 100 for a clean internal comparison). Watch swap
   at len 200 (2× window memory).
2. **time-delta on / off** (`--no-time`), a clean null in the prior project;
   re-test. One priors run, len 100.
3. **ID vs genre-hybrid**, the headline content question + the **cold-ITEM** test
   (ID can't score a track absent from training; genre can). Needs code: wire the
   genre matrix into `train_adoption.py` as `text_vectors` for
   `item_variant="hybrid"` (currently unwired). Expect ID strong on warm catalog,
   hybrid to *add cold coverage*, not to make content strong where content is all
   there is.
4. **event-horizon N 50 / 100 / 200**, rebuild labels at each N
   (`build_labels.py --event-n`) and re-score. Heaviest (label rebuild over 50M
   encounters). Report whether the model **ranking** is stable across N.

### 4.1 max_len 50 / 100 / 200: ✅ done
Ran both heads at len 50 and len 200 (len 100 from Phase 3); priors-len200 skipped
after len50≈len100 for priors made it a near-certain repeat (and len 200 cost
~4.9 h/head, the validation pass thrashes swap at len 200 even with the mmap fix).
Scored all five on the fixed cohort (`artifacts/adoption/phase4-maxlen-table.md`).

| slice | user×item | pure l50→l100→l200 | priors l50 / l100 |
|---|---|---|---|
| all | 0.3777 | 0.3500 → 0.3564 → 0.3605 | 0.3892 / 0.3895 |
| cold_user | 0.3125 | 0.3314 → 0.3338 → 0.3382 | 0.3186 / 0.3218 |
| unfamiliar | 0.3385 | 0.3176 → 0.3266 → 0.3261 | 0.3467 / 0.3446 |

Paired Δ vs `user×item`: pure beats **cold_user significantly at all three lengths**
(+0.0185\*, +0.0216\*, +0.0254\*) and the edge **grows with history**; priors beats
**unfamiliar + overall significantly at both lengths**.

**Meaning:** the **forced len-100 config changed no conclusion**, every ranking
holds across length. Two findings fall out: (1) **history length helps the `pure`
head monotonically** (it must extract everything from the sequence, so more of it
helps, and its cold-user edge strengthens); (2) **the `priors` head is
length-insensitive** (l50 ≈ l100), anchored to the scalar rates, extra history
adds little. That gap *is* the pure-vs-priors story. The earlier "numbers may be a
conservative floor" worry is mildly confirmed for `pure` (len 200 is its best) but
does not touch the headline `priors` result.

### 4.2 time-delta on / off: ✅ done
Trained both heads with `--no-time` at len 100 and scored vs the time-on
checkpoints (`artifacts/adoption/phase4-notime-table.md`). **Paired Δ = time −
no-time** per head (resampling users; `*` = 95% CI excludes 0):

| slice | pure | priors |
|---|---|---|
| all | +0.0148 \* | +0.0087 \* |
| cold_user | +0.0054 ns | +0.0152 \* |
| unfamiliar | +0.0175 \* | +0.0063 \* |

**Meaning:** time-delta **significantly helps** in 5 of 6 comparisons, a real
**divergence from the prior next-track project, where time was a clean null.** The
interpretation: adoption is about *return* behavior, and the spacing between
listens (recency, bingeing rhythm) is genuinely predictive of whether a user comes
back, signal the immediate-next-track task didn't need. Keep time on.

### 4.3 ID vs genre-hybrid representation (+ cold-item): ✅ done (with a caveat)
Wired the genre matrix into the trainer as `text_vectors` (a zero pad row prepended
to align with track_code+1) and made `load_checkpoint` rebuild the buffer shape for
non-id variants, so a hybrid checkpoint reloads without the genre file. Comparison
run on the **pure head**: hybrid `pure` trained fine (val 0.3546), but hybrid
`priors` **diverged after epoch 1** (val 0.4961 → 0.3936, loss rising) and was
crawling at 1.6 s/batch (swap thrash: the 155 MB text buffer + per-row projection
push this 5.8 GB box over the edge), killed after ~13 h. The pure head is the
cleaner representation test anyway (it isolates the item rep from the prior crutch,
and it is where the cold-item question lives).

Added a **cold_item slice** to the scorer (cohort tracks absent from the training
rows): **157,330 of 500,001 rows (31%)**, base rate **0.3919**, cold items adopt
*more* than average.

| slice (base) | user×item | id-pure | hybrid-pure | id-priors |
|---|---|---|---|---|
| all (0.308) | 0.3777 | 0.3564 | 0.3168 | 0.3895 |
| cold_item (0.392) | 0.4631 | 0.4406 | 0.3952 | 0.4763 |

Paired Δ = **hybrid-pure − id-pure**: all −0.039\*, **cold_item −0.045\***,
cold_user −0.038\*, unfamiliar −0.029\* (all significant). Paired vs `user×item` on
cold_item: id-pure −0.024\*, hybrid-pure −0.069\*, **id-priors +0.012\***.

**Meaning, this inverts the plan's expectation, and the reason is the real finding:**
1. **Hybrid does not rescue cold items, it is worse than ID even there.** Genre
   content is too weak (genre-sim baseline lift 1.067) to beat a learned ID, and
   tying the candidate vector to genre *hurts*.
2. **ID is not "dead" on cold items in this task.** `id-priors` actually **beats
   `user×item` on cold_item (+0.012\*)**. Why: the next-track *ranking* project saw
   cold items score ~0 because you rank the item's (random) embedding against the
   catalog; **binary adoption predicts from `[h, c, h⊙c]` and leans on the user
   history `h`**, so a random cold-item `c` still yields a good prediction. The
   cold-item catastrophe of ranking **does not transfer to the adoption task**,
   a genuinely task-specific result.
3. ~~**Content is weak, confirmed a third way.**~~ ⚑ **RETRACTED, see the
   Audio-features section.** This held only for *hybrid* (genre + residual): the
   residual reintroduces ID-style overfitting, so hybrid < id stands. But
   *content itself is not weak*, a residual-free `text_frozen` projection of even
   genre **beats** id-pure (+0.029\*), and learned audio beats that again. The
   weakness was the method (residual hybrid, raw cosine) + genre coarseness.

**Caveats:** hybrid `priors` did not complete (divergence + thrash), so the hybrid
side rests on the pure head; hybrid `pure` may be mildly undertrained. Genre is
coarse (685 dims), **whether content is *inherently* weak or just coarse-genre
here remains open** (richer id_musicnn/id_bert embeddings untested). Do not
generalize "content can't help cold-start" beyond this genre representation.

**Framing that governs every content/cold-start claim (non-negotiable):** transfer
value lives in the **encoder**, not the item representation. Hybrid helps across a
*mixed* catalog; it does **not** make content strong where content is all there
is. The claim to defend: "the behavioral encoder transfers (testable); content is
weak *in this data* (measured); whether that's inherent or coarse-genre is an open
question." Never "content solves cold-start" or "it works in 2026."

---

### 4.4 event-horizon N sweep (50 / 100 / 200): ✅ done (lightweight)
Full fidelity (rebuild 50M labels at each N *and retrain*) is ~30 h+ on this box,
so this is the **lightweight version**: relabel the cohort at each N, **refit the
baselines at that N**, and re-score the existing N=200-trained id-pure/id-priors
predictions against the new labels. It answers the plan's question ("is the ranking
of models stable across N?") as a horizon-robustness check, not a per-N retrain.

| slice | winner | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| all | id-priors | 0.3144 vs 0.3049 | 0.3529 vs 0.3415 | 0.3895 vs 0.3777 |
| cold_user | id-pure | 0.2526 vs 0.2330 | 0.2955 vs 0.2727 | 0.3338 vs 0.3125 |
| unfamiliar | id-priors | 0.2810 vs 0.2756 | 0.3158 vs 0.3089 | 0.3446 vs 0.3385 |
| cold_item | id-priors | 0.4023 vs 0.3896 | 0.4414 vs 0.4285 | 0.4763 vs 0.4631 |

(each cell: winning model vs `user×item`; base rate scales 0.23 → 0.27 → 0.31 with N.)

**Meaning:** the ranking is **completely stable across the horizon**, priors best
overall/unfamiliar/cold_item, pure best on cold_user, at every N. Lift over base
rate is if anything **slightly stronger at tighter N** (priors overall lift
1.36 / 1.32 / 1.27 for N = 50 / 100 / 200). The N=200 choice did not manufacture
the result. **Caveat:** lightweight, models were trained at N=200, so this tests
robustness of the trained ranking, not a full per-N retrain.

---

## Phase 4: summary of findings
1. pure vs priors: a split pass. Pure wins cold_user (significant), priors wins overall
   and unfamiliar (significant); no single head wins both required slices.
2. max_len: no conclusion changes. History helps `pure`, not `priors`.
3. time on/off: time significantly helps (5 of 6 comparisons), unlike the prior project.
4. ID vs hybrid: ID beats the hybrid everywhere (the residual overfits), and the ranking
   cold-item catastrophe does not transfer to binary adoption. Note: "content is weak" is
   later retracted (see Audio features), since residual-free content (`text_frozen`) beats
   ID and learned audio beats genre.
5. event-N: the ranking is stable across the horizon.
6. Audio features: content is not weak. `text_frozen` content beats ID (architecture,
   +0.03\*) and audio beats genre (+0.03\*); the best content model on its own ties the
   in-context rate and covers cold-start.

The through-line, revised twice (after the cold-start adjudication, then the
audio-features probe):
- A training-free in-context running rate is the single strongest signal. It beat every
  ID model, overall and on cold_user (Cold-start adjudication).
- Content is not weak (the initial finding is retracted): a residual-free `text_frozen`
  projection of content, run through the encoder, beats every ID variant and, with learned
  audio, ties the in-context rate while needing no ID table (Audio features). The item
  representation does matter after all; the earlier "content weak, ID wins" came from raw
  cosine and the residual hybrid.
- Time-delta genuinely helps, a real divergence from the prior project.
- The headline: handed the in-context rate as a fixed base it can only add to, the content
  sequence beats the rate on every slice (+0.031\* overall, +0.036\* cold_user), the first
  model in the project to do so (Content encoder + in-context rate). So the best model is
  content plus the rate, and the most defensible results are the rigorous analysis behind
  it: the in-context rate is the bar, content weakness was a method artifact, audio beats
  genre, and time matters.

---

## Cold-start adjudication: the cold_user win does NOT survive  ⚑ correction

Prompted by a sharp doubt: on `cold_user` the fitted priors are structurally
blind (a held-out user has no train row → `user-prior`/`user×item` fall back to
global), so the model's win there might be nothing but **recovering the user's own
adoption rate from in-context history**, not understanding adoption. Added
`incontext-user-rate` (`baselines.incontext_user_rate`): a training-free, running
per-user adoption rate over that user's prior **resolved** test-period
first-encounters, shrunk to the global prior, **no look-ahead** (a prior counts
only once its horizon closed, `resolution_pos < encounter_pos`; unit-tested).
Scored on the identical cohort (`scripts/adjudicate_coldstart.py` →
`artifacts/adoption/coldstart-adjudication.md`).

**cold_user (base 0.2898):** incontext **0.3776 (lift 1.30)** vs id-pure 0.3338,
id-priors 0.3218, global 0.2898.
**all (base 0.3079):** incontext **0.4212 (lift 1.37)** vs id-priors 0.3895,
user×item 0.3777, id-pure 0.3564.

Paired Δ (resample users; `*` = 95% CI excludes 0):
- **Q1** incontext − global-prior: cold_user **+0.083\***, all **+0.113\*** → a
  running rate alone captures the cold_user signal. **The doubt is real.**
- **Q2** id-pure − incontext: cold_user **−0.038\***; id-priors − incontext:
  cold_user **−0.050\*** (and all: −0.064\* / −0.031\*) → **the models are
  *significantly worse* than the running rate**, on cold_user and overall.

**Fairness (all conservative against the baseline):** no look-ahead (tested);
incontext uses only test-period priors, a *subset* of the encoder's window, and
still wins; its signal is derivable from the same play stream the model sees raw.
Effective evidence on cold_user: median 219 resolved priors, only 2.7% with none.

**Verdict (the plan's "NO" branch):** the Phase-3/§3f claim *"the encoder earns
its keep on cold users"* **is retracted.** The model beats the *fitted* priors on
cold_user only because they are blind there; a direct in-context rate estimator
does the same job **better**. More broadly, **a two-line training-free in-context
rate is the strongest adoption signal measured, and the neural encoder
underperforms it everywhere.** This reframes the project headline and is the
honest result the test existed to be able to reach. §3f, §4-summary and the
Decision index below are corrected accordingly; the demo and README must not claim
a cold-start win.

### The encoder's fair last shot: feed it the rate it lost to

Gave the encoder the feature it was losing to: a **"sequence + user-prior +
item-rate + incontext-rate"** head (`n_prior_features=3`), against two controls on
the same cohort (`scripts/train_incontext_head.py`,
`artifacts/adoption/phase4-incontext-table.md`). incontext here is the
**deployable** variant (pooled over both periods, still no look-ahead), even
stronger than the adjudicator's test-only one.

| slice (base) | user×item | **incontext** | 3-prior logistic (no seq) | model (priors+ic) | id-priors |
|---|---|---|---|---|---|
| all (0.308) | 0.3777 | **0.4123** | 0.3666 | 0.3819 | 0.3895 |
| cold_user (0.290) | 0.3125 | **0.3920** | 0.2652 | 0.3016 | 0.3218 |

Paired Δ (`*` = 95% CI excludes 0):
- **Sequence adds over the same 3 scalars:** priors+ic − 3-prior-logistic =
  **+0.015\*** (all), **+0.037\*** (cold_user). The encoder extracts *something*
  real beyond the scalars.
- **But the full model still loses to the raw rate:** priors+ic − incontext =
  **−0.030\*** (all), **−0.084\*** (cold_user).
- **Training-fit combination actively dilutes the signal:** the no-sequence
  logistic − incontext = **−0.045\*** (all), **−0.120\*** (cold_user), on
  cold_user it drops to lift **0.915, below the base rate**. Fit on train, the
  combiner underweights incontext for the cold-start regime (drift), so the
  trained model is *worse* than using the rate directly. Even feeding incontext to
  the priors head left it slightly below plain id-priors (0.3819 vs 0.3895).

**Verdict (option 2):** the sequence encoder is **not worthless**, it adds
bootstrap-significant signal over the strongest scalar features. **But the neural
approach still does not beat the two-line, training-free in-context rate**, and we
now know *why*: a train-fit model misweights that rate for the cold-start/drift
regime a direct estimator handles cleanly. The thing to beat remains unbeaten.

### Terminal experiment: does the sequence add OVER the in-context rate?  ⚑ clean negative

The fair-last-shot fed incontext to the *priors* head and still lost. This closes the
question directly: hand the encoder the exact `incontext-user-rate` array as an input and
measure the sequence's contribution two ways, both on the fixed cohort
(`scripts/train_seq_over_incontext.py`,
`artifacts/adoption/phase4-seq-over-incontext-{residual,concat}.md`). Run one variant per
process (a single 15k/len100 run fits this box; stacking two orphaned runs is what caused
the earlier swap-thrash, [D10]).

* **residual**, the head sees only `[h, c, h⊙c]` and predicts a *correction*; the output is
  `correction + logit(incontext)` with the base coefficient **fixed at 1**, so it can only
  add, never dilute the rate. The clean, architecturally-fair test.
* **concat**, incontext (+ a `seen/(seen+pseudocount)` confidence feature) is concatenated
  as an ordinary head input, so training is free to weight it, including down-weighting.

| variant | all: incontext → seq+incontext | paired Δ (all) | paired Δ (cold_user) |
|---|---|---|---|
| residual | 0.4212 → 0.3995 | **−0.0213 [−0.0283, −0.0151]\*** | −0.0047 [−0.0194, +0.0119] ns |
| concat | 0.4212 → 0.4141 | **−0.0068 [−0.0114, −0.0022]\*** | +0.0069 [−0.0033, +0.0165] ns |

**Verdict, clean negative.** Both arms lose significantly overall and tie on cold_user;
neither shows a significant gain anywhere. Even the **residual** head, which architecturally
*can only add* to the rate, ends up below it, because the sequence's learned correction, fit
on train under drift, is net harmful to *ranking* (it does **improve calibration**: ECE 0.031
vs 0.049 on all). Concat lands closer to the bar only because it can lean almost entirely on
the incontext feature and nearly ignore the sequence. **The ID sequence encoder adds nothing
over the two-line, training-free in-context rate, the thing to beat remains unbeaten.**

**Scope, resolved below.** This first pass tested the **ID** encoder. The follow-up
(next subsection) runs the identical residual head over the best *content* encoder, and
flips the result.

### Content encoder + in-context rate: the rate is beaten  ✅ headline

Re-ran the **residual** arm with the encoder's item representation swapped from `id` to
`text_frozen(musicnn)`, the best content model, via new `--item-variant`/`--text-matrix`
support in `scripts/train_seq_over_incontext.py`. Everything else identical: same cohort,
the same fixed `logit(incontext)` base the head can only add to, len 100, 15k users
(`artifacts/adoption/phase5-content-incontext-residual.md`).

| slice | base | incontext-alone | content+incontext (residual) | paired Δ (`*` = 95% CI clears 0) |
|---|---|---|---|---|
| all | 0.3079 | 0.4212 | **0.4520** | **+0.0312 [+0.0231, +0.0392]\*** |
| cold_user | 0.2898 | 0.3776 | **0.4106** | **+0.0364 [+0.0245, +0.0499]\*** |
| unfamiliar | 0.2860 | 0.3706 | **0.4048** | wins |
| new_neighborhood | 0.3064 | 0.3730 | **0.4062** | wins |

Wins on **every** slice, significant overall and on cold_user. Unlike the ID models
(epoch-0 overfit), it trained 20 epochs (best val 0.4957).

**Meaning, the first model to beat the in-context rate, and the mechanism is the item
representation.** The residual head is byte-identical across the two runs; only the
encoder's item embedding changed (ID → learned audio). The **ID** sequence's correction
was net harmful (−0.021\*); the **content** sequence's correction is a significant,
everywhere *gain* (+0.031\*). So the sequence *does* carry adoption signal beyond the
in-context rate, but only when items are represented by **content**, not a memorised ID
table that overfits the train period under drift. Because the base is fixed at
coefficient 1, the +0.031 is genuinely additive: **content sequence plus the training-free
rate is the strongest adoption model in the project.**

**Corroborated by concat** (head free to weight the rate): also a significant win,
**+0.0253 [+0.0195, +0.0310]\*** (all), **+0.0349\*** (cold_user), 0.4463 (all), a hair
below residual (0.4520) because residual keeps the full rate as base while concat can
slightly misweight it. Both combination methods flip sign with the item representation:
ID (residual −0.021\*, concat −0.007\*, both losses) → content (residual +0.031\*,
concat +0.025\*, both wins). **The item representation, not the head design, decides
whether the sequence beats the rate.**

**Caveats.** Single seed; the model *consumes* the in-context rate, so this is "content
adds *over* the rate," not "content alone beats it", the rate stays a large component.
Artist/lyrics content (now available via Music4All, every one of the cohort's 56,512
tracks resolves, see Phase 5) is untested and may raise the content contribution further.

---

## Audio features: is content weak *inherently*, or just coarse genre?  ⚑ correction

The project had left this open: genre is weak, but is content *inherently* weak
for adoption, or is genre just coarse? Tested with Music4All-Onion's **learned
audio embedding `musicnn`** (24 MB, 50-dim, downloaded from Zenodo;
`data/interim/onion-features-v1/musicnn.npy`) swapped through the existing
item-representation seam (`--text-matrix` on `train_adoption.py`).

**Probe 1, raw content similarity** (`report.genre_similarity`, cosine to the
user's pre-encounter centroid): audio-sim **worse** than genre-sim, all
0.3201 vs 0.3285 (paired −0.008\*), cold_item 0.4148 vs 0.4358 (−0.021\*). As a
*raw* signal, learned audio is no better than coarse genre.

**Probe 2, hybrid head:** NaN'd at epoch 2 regardless of input scaling, the
hybrid's `L2-normalize + log_scale` is numerically fragile under fp16 AMP (genre's
larger norms had masked it). Pivoted to the residual-free **`text_frozen`**
variant, which is both stable *and* the cleaner content test (no ID-like residual
to memorise).

**Probe 3, `text_frozen(musicnn)`: the correction.** Pure audio content, learned
50→128 projection, run through the encoder, no ID table, no residual. Unlike every
ID model (which peaks at epoch 0 and overfits), it **trained 18 epochs without
overfitting**, a shared projection can't memorise per item. Cohort
(`artifacts/adoption/phase4-audio-table.md`):

| slice | user×item | incontext | id-priors | **tf-musicnn** |
|---|---|---|---|---|
| all | 0.3777 | 0.4212 | 0.3895 | **0.4139** |
| cold_user | 0.3125 | 0.3776 | 0.3218 | **0.3872** |
| unfamiliar | 0.3385 | 0.3706 | 0.3446 | **0.3712** |
| cold_item | 0.4631 | 0.5147 | 0.4763 | 0.4864 |

Paired: tf-musicnn − **id-priors** = +0.024\* / +0.064\* / +0.026\* (all /
cold_user / unfamiliar), **beats every ID model.** tf-musicnn − **incontext** =
−0.007 / +0.015 / +0.001 (**ties** on all/cold_user/unfamiliar) and **−0.027\* on
cold_item (loses).**

**Meaning:** **content is not inherently weak.** A learned-audio content model, run
through the encoder with a learned projection, is the **strongest model in the
project**, it beats every ID variant and is the **first model to statistically
*match* the incontext baseline** overall, while needing no ID table (so it covers
cold users and items). The earlier "content is weak" was an artifact of (a) raw
cosine similarity and (b) the residual-laden, unstable hybrid, not of content
itself. The mechanism is generalisation under drift: ID embeddings memorise and
overfit the train period; a shared content projection cannot, so it transfers.

**Probe 4, `text_frozen(genre)` control: the question resolves into two effects.**
Both `text_frozen` variants scored on the cohort (`phase4-audio-table.md`):

| slice | id-pure | tf-genre | tf-musicnn | incontext |
|---|---|---|---|---|
| all | 0.3564 | 0.3855 | 0.4139 | 0.4212 |
| cold_user | 0.3338 | 0.3627 | 0.3872 | 0.3776 |
| cold_item | 0.4406 | 0.4588 | 0.4864 | 0.5147 |

Paired (all `*` unless noted): **tf-genre − id-pure = +0.029/+0.029/+0.018/+0.016\***
(all/cold_user/cold_item/unfamiliar); **tf-musicnn − tf-genre =
+0.028/+0.023/+0.028/+0.029\***; tf-genre − incontext = −0.035\* (all), −0.008 tie
(cold_user), −0.055\* (cold_item).

**Final answer, it was both, and mostly a method artifact:**
1. **Content is not inherently weak.** Through the encoder as a learned projection,
   *even coarse genre* beats the ID model (+0.029\*). The earlier "content weak"
   was raw cosine + the residual hybrid, not content.
2. **Genre is coarse.** Learned audio adds an equal, significant step over genre
   (+0.028\*) through the identical architecture.
3. **They stack:** id-pure 0.356 → *+architecture* → tf-genre 0.386 → *+audio* →
   tf-musicnn 0.414, each ~+0.03 and bootstrap-significant. Two mechanisms:
   a shared content projection **can't memorise per item, so it doesn't overfit
   under drift** the way the ID table does (architecture); and learned audio
   carries **taste signal genre lacks** (richness).

**Reframes the project headline.** The strongest *model* is now a content-based
(learned-audio) sequence model that **ties the previously-unbeaten in-context rate**
overall/cold_user/unfamiliar (loses cold_item), while needing **no ID table**, so
it also covers cold users and items. The in-context running rate is still the
single strongest signal, but content is competitive and the "content is weak"
finding is retracted: it was measurement method (raw cosine, unstable residual
hybrid) plus genre coarseness, not content itself.

**Caveats:** single seed; the content models train ~4.5 h (18 / 7 epochs, no
overfit) vs the ID models' epoch-0 stop, that content *enables* productive
training is part of the finding, not a confound; the hybrid's fp16 instability was
worked around by using the residual-free `text_frozen` variant, not fixed.

---

## Phase 5: the demo and write-up  ✅ done

**Demo.** `scripts/demo_best_model.py` scores the best model (content sequence over a
fixed in-context base, checkpoint `runs-content-incontext-residual/residual/best.pt`) on
the fixed cohort. It reproduces the 0.4520 headline as a sanity gate, asserts the same
cohort and the aligned score dump, and joins the Onion track ids to real artist and song
names through Music4All's `id_information.csv` (the Onion ids are Music4All ids, 100%
overlap on the cohort's 51,415 tracks). It picks three real listeners by deterministic,
archetype-driven rules, the most active listener in each openness band, chosen by
activity and openness, never by where the model looks good:

- a steady regular (3,450 first-encounters, returned to 26%, model average call 28%),
- a cold-start listener held out of training entirely (1,427 encounters, 39%, average
  call 36%),
- an open enthusiast (1,474 encounters, 76%, average call 57%).

**What the demo shows, at two levels.** Between listeners, the model's average call tracks
each person's real return rate closely (correlation 0.69 across the listeners with a few
encounters). Most of the prediction is just how open the listener is, which the
training-free in-context rate already captures. Within a listener, the model also ranks
its tracks: over the ~1,290 listeners with enough encounters, the top-ranked half come
back 40% of the time against 34% for the bottom half, a real but modest edge. The demo
says this plainly, since a handful of named tracks only illustrates the idea and
per-listener separation is soft. No user ids are shown, only archetype labels, public
track and artist names, and model outputs.

**Write-up.** `demo/index.html` is a single self-contained page (system sans-serif, light
and dark themes, no build step) that tells the research arc: the question, the model,
baselines, the iteration, results, the demo, roadblocks, discussion, limitations, and
conclusion. It is deployed on Vercel at https://melochron.vercel.app, git-linked to `main`
with root directory `demo`, so every push redeploys it. The headline numbers also live in
`README.md`, each PR-AUC beside its base rate.

---

## Phase 6: full-scale cloud run  ✅ done

The reduced config (15k users, history 100) was forced by the 6 GB laptop. To test the
"reduced config is a floor" hypothesis, the winning content + in-context residual model was
retrained at full config on a cloud GPU (RunPod): the whole ~100k-user training pool and
history 200, everything else identical (`--variant residual --item-variant text_frozen
--text-matrix musicnn`, same split seed, same fixed cohort). The checkpoint was scored on
the identical 500k cohort with `scripts/score_checkpoint.py`; the `incontext-alone` column
reproduced 0.4212 exactly, confirming the scoring path.

| slice | base | in-context rate | content+rate laptop (15k/100) | content+rate cloud (100k/200) |
|---|---|---|---|---|
| all | 0.3079 | 0.4212 | 0.4520 | 0.4820 |
| cold_user | 0.2898 | 0.3776 | 0.4106 | 0.4224 |
| unfamiliar | 0.2860 | 0.3706 | 0.4048 | 0.4379 |
| new_neighborhood | 0.3064 | 0.3730 | 0.4062 | 0.4457 |

Paired delta over the in-context rate, all significant: all +0.061 [+0.050, +0.073],
cold_user +0.049, unfamiliar +0.067, new_neighborhood +0.072.

**Meaning:** the reduced-config numbers were a floor, as suspected. More users and a longer
window help the content model, consistent with the mechanism: a shared content projection
generalizes rather than memorizing, so more data is more signal. The headline is now content
plus the rate at 0.4820, +0.061 over the training-free bar (double the laptop margin). The
in-context rate itself does not move, since it needs no training. Reproduce with
`scripts/score_checkpoint.py --checkpoint <best.pt>`.

---

## Limitations carried forward
- **Single seed, single split.** Discovery-slice wins are small (+0.006).
- **Reduced config** (15k/len100) forced by 5.8 GB RAM; the `max_len` sweep tests
  whether it changed conclusions.
- **Drift** miscalibrates every train-fitted prior on test (PR-AUC is rank-based,
  so it hurts calibration more than ranking).
- **Artist-metadata gap** is structural; content signal is genre-only so far.
- **Cold-start honesty:** the ID representation is *dead* on a genuinely new song;
  content (genre) covers any song but is weak alone (≈ chance on unfamiliar). The
  thing that solves new-song coverage is the thing that barely works, that
  tension is the project's real subject, tested in Phase 4.

---

## Decision index (quick reference)

| # | Decision | Why | Status |
|---|---|---|---|
| D1 | Rebuild around adoption, reuse encoder | recurrence = real intent; next-track labels were autoplay | done |
| D2 | int32 columnar corpus, sorted (user, ts) | 253M events; source is reverse-chronological | done |
| D3 | Event-N=200 + time-30d labels; exclude horizon-crossing train rows | no look-ahead across split | done |
| D4 | Genres (not tags) for item vectors & slicing | tag coverage collapses on popular tracks | done |
| D5 | Fixed 500K-row cohort; bootstrap over users | comparison must be exact; rows aren't independent | done |
| D6 | PR-AUC headline, AUROC secondary | base rate 0.36 falsifies the AUROC-ban premise | done |
| D7 | Two heads (pure, priors) on one encoder | their gap = "what order/content adds over 2 rates" | done |
| D8 | Keep `torch.compile` (~1.3x), not the main lever | model is compute-bound, not launch-bound | done |
| D9 | Reduced config 15k/len100 | 5.8 GB RAM; cohort fixed so comparison exact | done |
| D10 | mmap the EncounterTable | frees 1.2 GB → kills swap thrash (~4-5x) | done |
| D11 | Standalone scorer + paired-difference bootstrap | honest significance test; wedged run never scored | done |
| D12 | Accept epoch-0 peak, early-stop on it | normal for ID recsys; numbers may be conservative | done |
| D13 | Phase 4 order: max_len → time → hybrid → event-N | insight-per-GPU-hour; validate forced config first | done |
| D14 | Add `incontext-user-rate` baseline; adjudicate cold_user | test the win against in-context rate recovery, it **loses**; claim shrunk | done |
| D15 | Learned audio (musicnn) via `text_frozen`; genre control | settle coarse-vs-inherent: content **not weak** (arch +0.03\*, audio +0.03\*); "content weak" retracted | done |
| D16 | Content sequence over a fixed `logit(incontext)` base (residual head) | test whether the sequence adds over the rate; content beats it +0.031\* (all), +0.036\* (cold_user), the first model to do so | done |
| D17 | Phase 5: best-model demo with real names + web write-up, deployed to Vercel | deliver the prediction demo and a public write-up; stay honest about the modest per-listener signal | done |
| D18 | Phase 6: full-scale retrain on a cloud GPU (RunPod), 100k users, history 200 | test the reduced-config floor; headline improves 0.4520 to 0.4820, margin over the rate doubles (+0.031 to +0.061\*) | done |
