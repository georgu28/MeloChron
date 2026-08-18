# MeloChron

Sequential music recommendation from listening history. A causal self-attention
encoder written from scratch in PyTorch, pretrained on 19.1M real plays, then
tested on a personal Spotify export that overlaps the training catalog by 6.5%.

Aggregate metrics in this domain are dominated by replays: a model that
memorizes what you just played scores well overall and has learned nothing. So
every table here is split into *repeat* and *novel*, and the novel column is the
one that counts.

## What the numbers say

- **A model pretrained on 992 strangers ranks tracks it has never seen, in a
  catalog it has never seen, at 26x chance — with zero gradient steps on the
  user's data.** A model trained on that user's own listening scores *exactly
  zero* on the same tracks.
- **Pretraining, not the text representation, is what buys that.** An identical
  architecture with identical text vectors, trained from scratch instead of
  loaded, scores 3.1x worse.
- **Per-user fine-tuning bought 4 extra hits out of 2,601** (+2.8% at k=10). The
  design bar was "adaptation ships only on a measured delta"; at k=10 it fails
  that bar, and clearly helps only on tracks the user already plays.
- **On a public 992-user corpus the transformer does not clearly beat item-kNN
  on novel tracks** — better at HR@10, worse at HR@20. The honest word is tie.

Two corpora, deliberately disjoint. lastfm-1K is where the model is pretrained
and the baselines are set. The personal export is where the transfer claim is
actually tested, because ~90% of its plays are tracks the pretrained model has
never encountered.

---

## Protocol

Every model, neural and not, is scored through one harness. Ranking is against
the **full catalog**, never a sampled subset. Ties break **pessimistically**.
`MRR@k` is the truncated variant. Split is a global temporal cut, with a
separate user holdout on lastfm-1K.

**Slices.** `repeat` = the user has played this track before; `novel` = they
have not. `cold_item` = absent from the training period. `cold_user` = the user
was held out of training entirely. `cold_start` = `novel` ∩ `cold_item`, the
strict transfer test.

**These numbers look low against published SASRec figures, and that is the
protocol, not the model.** Most papers rank against ~100 sampled negatives; this
ranks against every item. Krichene and Rendle (KDD 2020) showed sampled metrics
are not merely noisier but *inconsistent* — they can reverse which model looks
better. Rows here are comparable to each other, not to a sampled-negative paper.

---

## Corpus 1 — lastfm-1K, HR@10

19,150,659 plays, 992 users, 2005-02-14 to 2013-09-29. Catalog capped at items
with ≥20 plays: **171,902 items**, 99 users held out. n=5,334.
**Chance HR@10 = 0.000058.**

| slice | n | popularity | repeat | item-kNN | **SASRec (id)** |
|---|---|---|---|---|---|
| overall | 5,334 | 0.0019 | 0.1468 | 0.2171 | **0.2553** |
| repeat | 3,898 | 0.0023 | 0.2006 | 0.2304 | **0.2804** |
| novel | 1,436 | 0.0007 | 0.0007 | 0.1811 | **0.1873** |
| cold_user | 570 | 0.0035 | 0.1912 | 0.2018 | **0.2386** |
| cold_item | 368 | 0.0000 | **0.3913** | 0.0000 | 0.0000 |

**The transformer wins overall and ties where it matters.** +17.6% relative over
item-kNN overall. On `novel` it is 0.1873 vs 0.1811 and NDCG@10 is 0.0964 vs
0.0959 — a tie — and at HR@20 it is *worse* (0.2124 vs 0.2472). It is more
precise in the top slots and retrieves less across a wider net.

**Repeat scoring 0.0007 on `novel` is the control working.** It is a cache; it
can only rank what you already played. Beating it there by 250x is what
separates a recommender from a lookup table. Item-kNN also clears it, which is
why item-kNN is the number to beat.

### Transfer ablation — item representation only

Same architecture, same data, same harness; only the item vectors change.

| slice | n | id | text (names) | text (+tags) |
|---|---|---|---|---|
| overall | 5,334 | **0.2553** | 0.1376 | 0.1474 |
| repeat | 3,898 | **0.2804** | 0.1632 | 0.1750 |
| novel | 1,436 | **0.1873** | 0.0682 | 0.0724 |
| cold_user | 570 | **0.2386** | 0.1754 | 0.1807 |
| **cold_item** | 368 | 0.0000 | 0.2500 | **0.2745** |
| **cold_start** | 79 | 0.0000 | **0.1392** | 0.0886 |

**Every `0.0000` in an ID column is a coverage statement, not a loss.** A track
absent from training has no row in the item table — it is not ranked badly, it
*cannot be ranked*. (Precisely: untrained rows keep their small init norm, so
they produce low logits and are suppressed.) Reading those cells as "the ID
model is bad at cold items" is wrong.

**The capacity comparison is not fair, and the direction matters.**
`text_frozen` trains **0.47M parameters against id's 22.43M**, because the text
matrix is frozen and only the 384→128 projection and the blocks train. A 47x
smaller trainable model buys unseen-item scoring and pays for it on seen ones.

**Genre tags barely moved ranking, which is the most interesting result here.**
Tags on 98.3% of items cut training loss (3.65 → 3.16 at matched epochs) but
moved HR@10 ~7% relative, and `cold_start` went *down* — 11 hits vs 7 out of 79,
with HR@20 a wash. The honest statement is **no reliable difference**. Likely
mechanism: tags were fetched per *artist*, so every track by an artist gets
identical text. That sharpens between-artist separation and destroys
within-artist discrimination — and much of next-track prediction is choosing
among tracks by the artist already playing.

---

## Corpus 2 — personal Spotify export, HR@10

191,783 parsed plays → **167,277 positives** (≥30s), 2023-03 to 2026-08 plus
eight stray 2017 plays. **One user**, 18,450 tracks, 5,676 artists, 7,175
sessions, repeat rate **0.9435**. Unlike lastfm-1K it carries real `ms_played`,
`skipped` (16.8%) and `shuffle` (66.9%).

**Overlap with the pretraining catalog: 6.51% of items, 9.71% play-weighted.**
On lastfm the `cold_start` slice was n=79; here it is **n=2,601**, so the
transfer claim finally has statistical power.

Catalog 18,450 items (`min_count=1`), 20,000 instances.
**Chance HR@10 = 0.000542.**

| slice | n | popularity | repeat | zero-shot | scratch (id) | scratch (text) | fine-tuned |
|---|---|---|---|---|---|---|---|
| overall | 20,000 | 0.0046 | 0.0191 | 0.0202 | **0.0343** | 0.0204 | 0.0267 |
| repeat | 17,399 | 0.0053 | 0.0219 | 0.0210 | **0.0394** | 0.0228 | 0.0284 |
| **novel** | 2,601 | 0.0000 | 0.0000 | 0.0142 | 0.0000 | 0.0046 | **0.0146** |
| cold_item | 9,795 | 0.0000 | 0.0120 | 0.0175 | 0.0000 | 0.0056 | **0.0177** |
| **cold_start** | 2,601 | 0.0000 | 0.0000 | 0.0142 | 0.0000 | 0.0046 | **0.0146** |

| column | what it is |
|---|---|
| **zero-shot** | the lastfm-pretrained encoder re-pointed at this catalog, **no gradient step on this user** |
| **scratch (id)** | ID embeddings trained on this user alone |
| **scratch (text)** | same architecture and text vectors as fine-tuned, randomly initialized — the control isolating pretraining |
| **fine-tuned** | zero-shot, then trained on this user's train period |

The transplant works because only the item table is catalog-shaped: the learned
384→128 projection and the transformer blocks are the same size for any catalog,
so 37 tensors transfer and the new catalog arrives as a new text matrix
(`melochron/train/transfer.py`).

**The two winners sit in different columns.** On seen tracks, memorizing your
own catalog wins outright — `scratch (id)` takes `repeat` at 0.0394 against
zero-shot's 0.0210. On unseen tracks it scores exactly zero, and the zero-shot
model, which has never seen this person or this catalog, is the only thing that
scores at all.

**Pretraining, not text, is what buys cold start.** `scratch (text)` differs
from zero-shot *only* by initialization and reaches 0.0046 against 0.0142 — a
**3.1x gap attributable to weights learned from 992 other listeners**. Text
representation lifts the floor off zero; pretraining makes the capability
useful rather than nominal. Same ordering on `cold_item`: 0.0056 vs 0.0175.

**Fine-tuning bought almost nothing where it counts.** `novel` moves 0.0142 →
0.0146: four extra hits out of 2,601. Two honest qualifications — at HR@**20**
the same comparison is 0.0238 → 0.0311 (+31%), so the benefit is real but
k-dependent, and fine-tuning clearly helps on *seen* items (+35% on `repeat`).
The deployable reading: zero-shot suffices for cold start; adaptation earns its
keep only on catalog the user already plays.

On unseen tracks the ranking is **pretraining >> text representation >
adaptation** — close to the opposite of where engineering effort naturally goes.

### Why the two corpora are not comparable

Measured, not asserted. The repeat baseline alone falls 0.1797 → 0.0219:

| | lastfm-1K | personal |
|---|---|---|
| median user history | 11,580 | **167,277** |
| repeats visible within `max_len=200` | 45.2% | **29.5%** |
| repeat HR@10 when target **is** in window | 0.3970 | **0.0742** |
| repeat HR@10 when it **is not** | 0.0002 | 0.0000 |
| distinct items per 200-play window | 111 | **181** |

Two effects compound. One listener's history is 14x longer, so a 200-event
context reaches proportionally much less far and **70.5% of `repeat` targets sit
outside it** — invisible to every scorer. And when the target *is* in the window
it is 5.4x harder to rank, because these windows hold 181 distinct tracks per
200 plays against 111, with two-thirds of plays shuffled. `max_len` is the
binding constraint here, not model capacity.

### What one user breaks

At `n_users=1` nothing crashes — it quietly produces a normal-looking table. The
harness was changed to refuse that:

- **`novel` and `cold_start` become identical**, exactly: `is_repeat` spans a
  prefix containing all of training, so `not repeat` implies `not in
  train_items`. On lastfm they differed *because 991 other people trained those
  embeddings*.
- **`cold_user` is unmeasurable** — no second user to hold out. The table names
  it rather than silently dropping the row.
- **Popularity and item-kNN are structurally 0.0000 on `novel`.** Both are
  collaborative; with one listener there is no collaborative signal. On lastfm
  item-kNN scored 0.1811 there purely because other people had played those
  tracks. The report annotates these zeros instead of letting a model look
  decisive against them.
- **Defaults collapse.** `--max-per-user` is a *per-user* cap, so on one user it
  becomes the whole evaluation — defaults would have produced a six-slice table
  computed on **23 instances**. The default stride yields **6 optimizer steps
  per epoch**; `configs/personal.yaml` uses stride 25 for 40 steps, raises
  epochs and patience, and switches to uniform negatives because
  popularity-weighted negatives drawn from one listener's own counts put the
  positive in its own softmax denominator.

Every table prints its **chance level**, because full-catalog HR against 18,450
candidates and against 171,902 are different questions that look alike.

---

## Insights

Derived from the trained model, no separate modelling. `melochron/insights/`.

**Taste drift** — cosine distance between item-vector centroids across 30-day
windows. Over 43 windows the median monthly step is 0.033 while cumulative
displacement reaches 1.66: slow continuous drift, not mood cycling. The
six-year dormancy between a handful of 2017 plays and the 2023 history is
emitted as a marked gap with the jump recorded, not interpolated into a line.

**Session archetypes** — KMeans over pooled session vectors, labelled by *lift*
over global frequency rather than raw count (raw counts name every cluster after
whatever is most popular). They separate far better than on lastfm (silhouette
0.241 vs 0.071), and the playback signals are what make them nameable:

| cluster | share | peak (UTC) | length | dwell | skip | shuffle |
|---|---|---|---|---|---|---|
| late-night deep listening | 11% | 00h | 56.4 | 176s | 0.06 | 0.60 |
| deliberate evening | 31% | 20h | 26.8 | 219s | 0.10 | **0.50** |
| shuffled daytime background | 59% | 14h | 25.5 | 201s | 0.09 | **0.76** |

Shuffle spans 0.50–0.76 and dwell 176–219s across clusters. Neither field exists
in lastfm-1K, which is why these were only measurable once a real export landed.

**Attention** — the last-position attention row, pad-masked and labelled with
the track, how many plays ago, and the time gap. It is sharply peaked on both
corpora (~15x uniform; the top 10 of 200 positions hold ~76% of the mass) but
points somewhere different in each. On lastfm the attention-weighted mean
recency is ~40 of 200 — clearly recency-biased. On the personal corpus it is
**85–99 of 200, which is no recency bias at all**: the model concentrates hard
on specific earlier plays rather than recent ones. That corroborates the repeat
analysis above — against shuffle-heavy, high-diversity listening, recency is a
weak signal and the model learns not to lean on it.

---

## Design decisions

**Pretrain on a population, then decide about adaptation on evidence.** One
person's history is too little to train a transformer, and without a population
the two-stage serving design is impossible. The follow-through is above:
fine-tuning was measured against zero-shot and did not clear the bar on unseen
tracks.

**Text item vectors, not IDs alone.** This is what makes cold start solvable
rather than a special case to apologize for.

**Last.fm tags, not Spotify.** Spotify's February 2026 changes removed batch
metadata endpoints, but the decisive reason is different: Spotify tags would
cover a personal export and not the pretraining corpus, splitting the item
representation into two incompatible spaces. Last.fm covers both.

**Catalog from the full frame; counts and fit from training only.** The catalog
is the universe of rankable items, not a learned parameter, so choosing it
globally is not leakage. Building it from training would make every
in-vocabulary target train-seen by construction and silently empty `cold_item`.

**Global temporal split plus a separate user holdout.** Two orthogonal
questions: does it predict this user's future, and does it work for someone it
never trained on. Validation is cut *inside* the training period — early
stopping on test data leaks invisibly.

**Time-interval-aware attention.** Inter-event gaps are log-bucketed and
injected alongside position, in the spirit of TiSASRec (Li, Wang, McAuley, WSDM
2020). A three-month gap is a different context from thirty seconds.

---

## Serving

Measured on CPU, full 171,902-item catalog, 200-event history:

| p50 | p95 | p99 |
|---|---|---|
| 8.2 ms | 9.35 ms | 15.71 ms |

**Cold start is not a special branch.** SASRec has no per-user parameters: it
conditions on the sequence, not a user id. A new uploader needs no per-user
fitting and no retraining — map their history to item ids, run the shared model,
score.

---

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e ".[dev,serve]"

python scripts/download_lastfm1k.py                     # 672 MB, HTTPS required
python scripts/build_dataset.py --data lastfm1k --path data/raw/lastfm-1k
python scripts/run_baselines.py --data parquet \
    --path data/interim/lastfm1k-v1.parquet --min-count 20
python scripts/build_embeddings.py --path data/interim/lastfm1k-v1.parquet --min-count 20
python scripts/train.py --config configs/pretrain.yaml --data parquet \
    --path data/interim/lastfm1k-v1.parquet --min-count 20
python scripts/bench_latency.py --checkpoint artifacts/runs/id-real/best.pt
```

### On your own export

Drop a Spotify extended streaming history under `data/my_data/` — the same
pipeline runs unchanged. `configs/personal.yaml` carries the settings a
single-user corpus forces.

```bash
P=data/interim/spotify-v1.parquet
V=data/embeddings/latest.npy
FLAGS="--min-count 1 --max-per-user 20000 --holdout-user-frac 0"

python scripts/build_dataset.py --data spotify --path data/my_data --out data/interim
python scripts/run_baselines.py --data parquet --path $P $FLAGS
python scripts/build_embeddings.py --path $P --min-count 1 --fetch-tags --tag-mode artist

# from scratch, then fine-tuned from the lastfm pretrain
python scripts/train.py --config configs/personal.yaml --data parquet \
    --path $P $FLAGS --name personal-id
python scripts/train.py --config configs/personal.yaml --data parquet \
    --path $P $FLAGS --variant text_frozen --text-vectors $V \
    --init-from artifacts/runs/text-frozen-real/best.pt --name personal-finetuned

# every row on identical instances, zero-shot included
python scripts/run_transfer.py --path $P --text-vectors $V \
    --pretrained artifacts/runs/text-frozen-real/best.pt \
    --checkpoint scratch-id=artifacts/runs/personal-id/best.pt \
    --checkpoint fine-tuned=artifacts/runs/personal-finetuned/best.pt

python scripts/build_insights.py --data parquet --path $P \
    --checkpoint artifacts/runs/personal-id/best.pt --window-days 30
```

Genre tags need a free Last.fm API key in `.env` as `LASTFM_API_KEY`. Without
one the pipeline runs on names only — a supported configuration, not a failure.
The tag cache is committed (25,325 artists, 86.7% item coverage on the personal
catalog) because it represents hours of rate-limited requests.

---

## Limitations

- **`cold_start` is n=79 on lastfm-1K** — the mechanism is demonstrated, the
  number is not precise. The personal export gives n=2,601, which is where the
  claim is actually measured.
- **Every run is undertrained.** The lastfm variants were still improving at
  epoch 10; the three personal models early-stopped on CPU at best epochs 18, 26
  and 10 of 60. All numbers are floors.
- **`max_len=200` is the binding constraint on a personal corpus**, not model
  capacity — 70.5% of `repeat` targets fall outside the window.
- **The personal corpus has one user**, so `cold_user` is unmeasurable, `novel`
  and `cold_start` coincide, and collaborative baselines are structurally zero.
  Properties of n=1, not results.
- **Artist-level tags** cover 98.3% of lastfm items for 8.5x fewer requests than
  per-track, but every track by an artist gets identical text, so tags cannot
  distinguish tracks within an artist. ~9% of artists have no tags and skew
  obscure — weakest exactly where cold-start problems live.
- **Positive-threshold asymmetry.** Spotify carries `ms_played` so skips can be
  filtered; Last.fm only scrobbles past ~50% of a track. Different implicit
  thresholds, not filtered versus unfiltered.
- **Single seed.** No confidence intervals; small slices should be read as
  directional.

## Not yet done

The `text_finetuned` and `hybrid` rows on the personal corpus, a `max_len` sweep
(the measured binding constraint), the time-delta ablation, and deployment. The
frontend that would consume the insight JSON is not built.
