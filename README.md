# MeloChron: adoption prediction

When a listener meets a track for the first time, do they come back to it? That is
what this project predicts: a binary label per first encounter, over 253 million real
listening events. It is deliberately not next-track ranking. Next-track labels are
mostly shuffle and autoplay, where the queue advanced and the listener made no choice.
Coming back to a track is a choice, so the label carries real intent.

The encoder is a from-scratch causal self-attention model
(`melochron/models/sasrec.py`), reused unchanged. Everything else, the labels,
baselines, the fixed evaluation cohort, the binary head, and the audits, is built for
this task in `melochron/adoption/`. The full decision log, including every result that
was later corrected, is in [`process.md`](process.md). A visual write-up is live at
**https://melochron.vercel.app** (source: `demo/index.html`).

> This repo once also held a next-track ranking project. It was removed for clarity and
> lives in git history (see `process.md`, "Repository scope").

## What the numbers say

Every PR-AUC is quoted with its base rate, because a precision-recall number means
nothing without one. The cohort base rate is 0.3079 (30.8% of first encounters recur
within 200 events), and a coin scores exactly that.

- The bar to beat is not a neural model. It is a two-line, training-free in-context
  rate: each listener's own running return rate over their prior first encounters, with
  no look-ahead and gated on resolution. It scores PR-AUC 0.4212, above the tuned
  `user × item` baseline (0.3777) and above every ID-embedding model. It was built as a
  test that could lose, and the encoder lost to it.
- "Content is weak" was a measurement artifact. Run through the encoder as a learned
  projection (not raw cosine, not a residual hybrid), even coarse genre beats a learned
  ID table (+0.029 PR-AUC, paired, interval clear of 0), and learned audio (musicnn)
  beats genre again (+0.028). The best content model on its own scores 0.4139, a
  statistical tie with the in-context rate, and it needs no ID table, so it covers cold
  users and cold items.
- The winning model feeds the rate in and lets a content sequence add to it. Handed
  `logit(in-context rate)` as a fixed base it can only add to, the content encoder
  scores 0.4520, a paired +0.031 [+0.023, +0.039] over the rate alone, and it wins every
  slice (cold_user +0.036, unfamiliar +0.035, new_neighborhood +0.033, each interval
  clear of 0). It is the first model to significantly beat the rate.
- ID embeddings overfit under drift; content generalizes. Adoption drifts down over time
  (train base 0.368, test 0.308). An ID table memorizes the training years and peaks
  after one epoch. A shared content projection cannot memorize per item, trains for many
  epochs without overfitting, and transfers.
- Time between listens matters here, unlike in ranking. Feeding the gaps helps on 5 of 6
  slices: recency and rhythm predict return.
- The ranking cold-item catastrophe does not transfer. In ranking, a track absent from
  training scores near zero. In binary adoption the model predicts from `[h, c, h⊙c]`
  and leans on the listener, so cold items (31% of the cohort) are scored well even by ID
  models.

## The task and the data

Music4All-Onion (see Data and citation) holds 252,984,396 events, 119,140 users, 56,512
tracks, and 50,016,042 distinct (user, track) pairs. The Onion release ships no artist
metadata (every file is track-id-keyed), which is why the original artist-affinity
baseline became an item-adoption-rate baseline. Artist and song titles, used only in the
demo, come from the base Music4All `id_information.csv`. The corpus is parsed into a
sorted int32 columnar store (`melochron/adoption/corpus.py`), and the label is recurrence
within the next 200 of the listener's own events (`labels.py`), which is fair across
light and heavy listeners.

## Protocol

- Baselines before the model, always: `global-prior`, `user-prior`, `item-rate`,
  `user × item`, `genre-sim`, and the audited `incontext-user-rate`.
- One fixed cohort: 500,001 rows drawn by whole users (1,315 of them). Every model scores
  the same rows, asserted by index.
- PR-AUC headline, with base rate and lift; AUROC secondary. The usual AUROC ban assumes
  a rare label, which this data (base 0.31) does not have.
- Bootstrap resamples whole users, not rows, since encounters within a listener are not
  independent. Wins are paired user-bootstrap deltas: is (model minus baseline)'s 95%
  interval clear of zero?

## Results (PR-AUC on the fixed cohort, base rate 0.3079)

| slice | base | user×item | incontext-rate | id-pure | id-priors | tf-musicnn (audio content) |
|---|---|---|---|---|---|---|
| all | 0.3079 | 0.3777 | 0.4212 | 0.3564 | 0.3895 | 0.4139 |
| cold_user | 0.2898 | 0.3125 | 0.3776 | 0.3338 | 0.3218 | 0.3872 |
| unfamiliar | 0.2860 | 0.3385 | 0.3706 | 0.3266 | 0.3446 | 0.3712 |
| cold_item | 0.3919 | 0.4631 | 0.5147 | 0.4406 | 0.4763 | 0.4864 |

Paired verdicts, all significant: `tf-musicnn − id-priors` +0.024 (all) and +0.064
(cold_user); `tf-genre − id-pure` +0.029; `tf-musicnn − tf-genre` +0.028; `incontext −
user×item` +0.11. `tf-musicnn − incontext` is a tie. The winning residual model, a
content sequence over a fixed in-context base, scores 0.4520 (all), a paired +0.031
[+0.023, +0.039] over the rate, and wins every slice.

## The demo

`scripts/demo_best_model.py` scores the best model on the cohort and shows three real
listeners with real artist and song names: a steady regular, a cold-start listener the
model never trained on, and an open enthusiast. Two signals show up. Between listeners,
the model's average call tracks each person's real return rate (correlation 0.69), so
most of the prediction is just how open the listener is. Within a listener, the tracks it
ranks highest come back a little more often (top-ranked half 40% versus bottom half 34%),
a real but modest edge. The write-up at https://melochron.vercel.app walks through it.

## Honest bottom line

A training-free in-context rate is the strongest single adoption signal, and no ID model
beats it. But it is not the ceiling. The best content encoder, handed that rate as a
fixed base it can only add to, significantly beats it (+0.031 overall, +0.036 cold_user),
the first model to do so, and it needs no ID table, so it also covers cold users and
items. The story is the item representation: an ID sequence adds nothing over the rate
because its learned correction overfits under drift, while a content sequence adds real
taste signal on top. The most durable results are the analysis: the in-context rate is
the bar, "content is weak" was a method artifact, learned audio beats genre, time between
listens matters, and the cold-item catastrophe does not transfer.

## Limitations

Single seed and split. Temporal drift miscalibrates every train-fitted prior on the test
period. The config was reduced (15k users, history length 100) to fit a 6 GB laptop GPU
with 5.8 GB RAM; a `max_len` sweep confirms the conclusions hold. Content is genre and
audio only, since the Onion release has no artist metadata. The hybrid item
representation is numerically fragile under fp16, worked around with the residual-free
`text_frozen` variant. Training covers 2005 to 2020 listening, so a live account today
would be out of distribution.

## Data and citation

This project uses Music4All-Onion, which extends the Music4All database. Both require
citation, and Music4All is provided for non-commercial research use only. The datasets
themselves are not redistributed here: everything under `data/` is gitignored (see
Reproducing to fetch them from source).

Music4All-Onion (Zenodo 15394646, CC BY 4.0):

> Marta Moscati, Emilia Parada-Cabaleiro, Yashar Deldjoo, Eva Zangerle, and Markus
> Schedl. Music4All-Onion: A Large-Scale Multi-faceted Content-Centric Music
> Recommendation Dataset. In Proceedings of the 31st ACM International Conference on
> Information and Knowledge Management (CIKM 2022), pages 4339 to 4343.
> https://doi.org/10.1145/3511808.3557656

Music4All (the base dataset, research-use agreement):

> Igor André Pegoraro Santana, Fabio Pinhelli, Juliano Donini, Leonardo Catharin, Rafael
> Biazus Mangolin, Yandre Maldonado e Gomes da Costa, Valéria Delisandra Feltrim, and
> Marcos Aurélio Domingues. Music4All: A New Music Database and its Applications. In 27th
> International Conference on Systems, Signals and Image Processing (IWSSIP 2020), pages
> 399 to 404.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]" --index-url https://download.pytorch.org/whl/cu126  # or cpu

python scripts/download_onion.py        # fetch Music4All-Onion (see the citation above)
python scripts/build_onion.py           # sorted int32 columnar store
python scripts/build_labels.py          # first-encounters, horizons, splits
python scripts/build_features.py        # genre matrix (+ musicnn via --text-matrix)

python scripts/train_adoption.py --compile --heads pure priors        # ID model + baselines
python scripts/train_seq_over_incontext.py --item-variant text_frozen \
    --text-matrix data/interim/onion-features-v1/musicnn.npy           # the winning content + rate model
python scripts/adjudicate_coldstart.py                                 # the in-context-rate adjudication
python scripts/demo_best_model.py                                      # the demo: real listeners, predicted vs actual
```

`pytest` runs the suite (small fixtures, exact assertions). Data lives under `data/` and
outputs under `artifacts/`, both gitignored; results live here and in `process.md`.
