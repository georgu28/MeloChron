# MeloChron — adoption prediction

When a listener hears a track **for the first time**, will they **come back to
it**? This is that prediction: per-`(user, track)` binary classification, one
example per first encounter, on 253M real listening events. It is deliberately
*not* ranking — next-track labels are mostly shuffle/autoplay, where the queue
advanced and the user didn't choose; **recurrence is unambiguously the user's
decision**, so the label encodes intent.

The encoder is a from-scratch causal self-attention model
(`melochron/models/sasrec.py`); everything else — labels, baselines, the fixed
evaluation cohort, the binary head, the audits — is built around this task in
`melochron/adoption/`. The full decision log, including every result that was
later corrected, is in [`process.md`](process.md).

> This repo previously also held a next-track ranking project; it was removed for
> clarity (it lives in git history). See `process.md` → *Repository scope*.

## What the numbers say

Every PR-AUC is quoted **with its base rate** — a precision-recall number is
meaningless without one. Cohort base rate is **0.3079** (30.8% of first encounters
recur within 200 events); a coin scores exactly the base rate.

- **The bar to beat is not a neural model — it's a two-line, training-free
  in-context rate.** A running per-user adoption rate computed from the listener's
  own prior first-encounters (`incontext-user-rate`, no look-ahead, resolution-gated)
  scores **PR-AUC 0.4212 (lift 1.37)** — stronger than the tuned `user × item`
  baseline (0.3777) and stronger than every ID-embedding model. This was found by
  an adjudication *built to be able to lose* — and the neural encoder lost to it.
- **"Content is weak" turned out to be a measurement artifact.** Run through the
  encoder as a *learned projection* (not raw cosine, not a residual hybrid), even
  coarse genre beats a learned ID table (+0.029 PR-AUC, paired, CI excludes 0), and
  a learned **audio** embedding (musicnn) beats genre again (+0.028). The best
  *model* in the project is pure audio content — **PR-AUC 0.4139** — which
  statistically **ties** the in-context rate overall and covers cold users *and*
  cold items with no ID table.
- **ID embeddings overfit under drift; content generalises.** Adoption drifts down
  over time (train base rate 0.368 → test 0.308). An ID table memorises the train
  period and peaks after a single epoch; a shared content projection can't memorise
  per-item, trains for many epochs without overfitting, and transfers.
- **Time-delta matters here, unlike in ranking.** Feeding the gaps between listens
  helps significantly on 5 of 6 slices — recency and rhythm predict *return*, signal
  the immediate-next-track task never needed.
- **The ranking cold-item catastrophe does not transfer.** In ranking, a track
  absent from training scores ~0 (its random embedding ranks randomly). In binary
  adoption the model predicts from history `[h, c, h⊙c]` and leans on the *listener*,
  so cold items — 31% of the cohort — are scored well even by ID models.

## The task and the data

**Music4All-Onion** (Zenodo 15394646, CC-BY-4.0): 252,984,396 events, 119,140
users, 56,512 tracks, 50,016,042 distinct (user, track) pairs. It ships **no
artist metadata** (every file is track-id-keyed), which is why the brief's
artist-affinity baseline became an item-adoption-rate baseline. The corpus is
parsed into a sorted int32 columnar store (`melochron/adoption/corpus.py`); the
label is *recurrence within the next N=200 of the user's own events*
(`labels.py`), which is fair across light and heavy listeners.

## Protocol

- **Baselines before the model, always.** `global-prior`, `user-prior`,
  `item-rate`, `user × item`, `genre-sim`, and the audited `incontext-user-rate`.
- **One fixed cohort.** 500,001 rows drawn by *whole users* (1,315 of them); every
  model scores the *same* rows, asserted by index — a table is only readable if
  every column met the same instances.
- **PR-AUC headline, with base rate and lift; AUROC secondary.** The brief's
  AUROC ban rests on a rare-label premise this data (base 0.31) falsifies.
- **Bootstrap resamples whole users, not rows** — encounters within a user aren't
  independent draws. Wins are reported as **paired** user-bootstrap deltas (the
  honest test: is model − baseline's 95% interval clear of zero?).

## Results (PR-AUC on the fixed cohort, base rate 0.3079)

| slice | base | user×item | **incontext-rate** | id-pure | id-priors | **tf-musicnn** (audio content) |
|---|---|---|---|---|---|---|
| all | 0.3079 | 0.3777 | **0.4212** | 0.3564 | 0.3895 | 0.4139 |
| cold_user | 0.2898 | 0.3125 | 0.3776 | 0.3338 | 0.3218 | **0.3872** |
| unfamiliar | 0.2860 | 0.3385 | 0.3706 | 0.3266 | 0.3446 | **0.3712** |
| cold_item | 0.3919 | 0.4631 | **0.5147** | 0.4406 | 0.4763 | 0.4864 |

Paired verdicts (all significant): `tf-musicnn − id-priors` +0.024 (all) / +0.064
(cold_user); `tf-genre − id-pure` +0.029; `tf-musicnn − tf-genre` +0.028;
`incontext − user×item` +0.11. `tf-musicnn − incontext` is a statistical **tie**.

**Terminal experiment — the item representation decides it.** Given the in-context
rate *as an input feature* (a **residual** head over a fixed `logit(incontext)` base,
so the sequence can only *add*), does the sequence beat the rate? The answer flips on
what the encoder embeds items with:

- **ID sequence — no.** Residual **−0.0213 [−0.0283, −0.0151]\*** (all), tie on
  cold_user; the concat variant (free to weight the rate) **−0.0068 [−0.0114,
  −0.0022]\***. The ID sequence's learned correction is net harmful under drift.
- **Content (musicnn) sequence — yes, decisively.** The same residual head over the
  best content encoder scores **0.4520 (all)** — paired Δ **+0.0312 [+0.0231,
  +0.0392]\*** over incontext-alone — and wins **every slice**: cold_user **+0.0364\***
  (0.3776 → 0.4106), unfamiliar (0.3706 → 0.4048). **The first model to significantly
  beat the in-context rate.** Because the base is fixed, that gain is genuinely what
  the content sequence adds *on top of* the rate — taste signal the ID table lacks.

(The `concat` arm for the content encoder is training; residual is the decisive test.)

## Honest bottom line

A training-free in-context rate is the strongest *single* adoption signal — no ID
model beats it. But it is **not** the ceiling: the best **content** encoder (learned
audio), handed that rate as a fixed base it can only add to, **significantly beats it**
(+0.031\* overall, +0.036\* cold_user) — the first model to do so, and it needs no ID
table, so it also covers cold users and items. The story is the **item
representation**: an *ID* sequence adds nothing over the rate (its learned correction
overfits under drift); a *content* sequence adds real taste signal on top. The durable
results are the analysis — the in-context rate is the bar, "content is weak" was a
method artifact, learned audio > genre, time-delta matters, the cold-item catastrophe
doesn't transfer — capped by the headline: **content + the rate beats the rate.**

## Limitations

Single seed. Temporal drift miscalibrates every train-fitted prior on test.
Config was reduced (15k users, history length 100) to fit a 6 GB laptop GPU with
5.8 GB RAM — the `max_len` sweep confirms conclusions hold. No artist metadata, so
content is genre/audio only. The hybrid item representation is numerically fragile
under fp16 (worked around with the residual-free `text_frozen` variant).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]" --index-url https://download.pytorch.org/whl/cu126  # or cpu

python scripts/download_onion.py        # fetch Music4All-Onion
python scripts/build_onion.py           # sorted int32 columnar store
python scripts/build_labels.py          # first-encounters, horizons, splits
python scripts/build_features.py        # genre matrix (+ musicnn via --text-matrix)

python scripts/train_adoption.py --compile --heads pure priors   # the model + baselines
python scripts/score_adoption.py  --model id-priors artifacts/adoption/runs/adoption-priors/best.pt
python scripts/adjudicate_coldstart.py  # the in-context-rate adjudication
python scripts/demo_adoption.py         # per-user predictions vs what happened
```

`pytest` runs the suite (tiny fixtures, exact assertions). Data lives under
`data/` and outputs under `artifacts/` — both gitignored; results live here and in
`process.md`.
