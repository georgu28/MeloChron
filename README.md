# MeloChron

Sequential music recommendation from listening history. A causal self-attention
encoder built from scratch in PyTorch, pretrained on 19.1M real listening
events, evaluated against non-neural baselines under a protocol chosen to make
the result hard to flatter.

The headline is not the best number. It is the decomposition: aggregate metrics
in this domain are dominated by replays, and a model that memorizes your recent
plays scores well overall while learning nothing. Every table below is split so
that cannot hide.

Two corpora, deliberately disjoint. The 992-user public dataset is where the
model is pretrained and the baselines are established. A **personal Spotify
export overlapping it by 6.5% of items** is where the transfer claim is actually
tested — and there the result splits cleanly: a model trained on your own
listening wins on tracks you have played and **scores exactly zero on tracks you
have not**, while a model pretrained on 992 strangers and never shown your data
is the only thing that ranks them at all.

---

## Results

**Corpus:** lastfm-dataset-1K (Celma, 2010). 19,150,659 plays, 992 users,
2005-02-14 to 2013-09-29. Catalog capped at items with >= 20 plays: **171,902
items**. Global temporal split, 99 users additionally held out of training
entirely.

**Protocol:** every model, neural and not, is scored through one harness.
Ranking is against the **full 171,902-item catalog**, not a sampled subset.
Ties are broken **pessimistically**. `MRR@k` is the truncated variant.

### Baselines and model, HR@10

| slice | n | popularity | repeat | item-kNN | **SASRec (id)** |
|---|---|---|---|---|---|
| overall | 5,334 | 0.0019 | 0.1468 | 0.2171 | **0.2553** |
| repeat | 3,898 | 0.0023 | 0.2006 | 0.2304 | **0.2804** |
| novel | 1,436 | 0.0007 | 0.0007 | 0.1811 | **0.1873** |
| cold_user | 570 | 0.0035 | 0.1912 | 0.2018 | **0.2386** |
| cold_item | 368 | 0.0000 | **0.3913** | 0.0000 | 0.0000 |

**Slices.** `repeat` = the user has played this track before. `novel` = they
have not. `cold_user` = the user was held out of training entirely, which is
the situation a new uploader is in. `cold_item` = the track is absent from the
training period.

### Reading this honestly

**The transformer beats the baselines overall, and ties on the slice that
matters.** Overall it is +17.6% relative over item-kNN. On `novel` it is
0.1873 vs 0.1811, and NDCG@10 is 0.0964 vs 0.0959, which is a tie. At HR@20 it
is *worse* (0.2124 vs 0.2472). It is more precise in the top few slots and
retrieves less across a wider net. Against a co-occurrence baseline on genuinely
new tracks, the sequence model has not clearly won.

**The repeat baseline scoring 0.0007 on `novel` is the control working.** It is
a cache: it can only rank what you have already played, so on new tracks it has
nothing to say. That the transformer beats it there by 250x is what
distinguishes a recommender from a lookup table. That item-kNN also does is why
item-kNN, not repeat, is the number to beat.

**Popularity at 0.0019 is not a bug.** Against 171,902 candidates, chance is
0.000058. Popularity is ~30x chance and still useless for personal next-track
prediction.

**These numbers look low against published SASRec figures, and that is the
protocol, not the model.** Most papers rank the target against ~100 sampled
negatives. This ranks against all 171,902 items. Krichene and Rendle (KDD 2020)
showed sampled metrics are not merely noisier but *inconsistent*: they can
reverse which of two models looks better. The full-catalog numbers are lower and
comparable across rows here; they are not comparable to a sampled-negative
paper.

---

## The transfer ablation

Item representation is the only thing that changes between these rows. Same
architecture, same data, same harness.

| slice | n | id | text (names only) | text (+ genre tags) |
|---|---|---|---|---|
| overall | 5,334 | **0.2553** | 0.1376 | 0.1474 |
| repeat | 3,898 | **0.2804** | 0.1632 | 0.1750 |
| novel | 1,436 | **0.1873** | 0.0682 | 0.0724 |
| cold_user | 570 | **0.2386** | 0.1754 | 0.1807 |
| **cold_item** | 368 | **0.0000** | 0.2500 | **0.2745** |
| **cold_start** | 79 | **0.0000** | 0.1392 | 0.0886 |

Tag coverage in the third column is 98.3% of items, from 20,191 artists.

**The `0.0000` is a coverage statement, not a model comparison.** With learned
per-ID embeddings, a track absent from training has no row in the item table. It
is not ranked badly; it cannot be ranked at all. Any reading of that cell as
"the ID model is bad at cold items" is wrong, and a README that did not say so
would be presenting a rigged baseline.

**`cold_start` is the honest version of the transfer claim.** `cold_item` alone
conflates two populations: evaluation context legitimately includes a user's
earlier test-period plays, so a user can already know a track the model never
trained on. `cold_start` is the intersection with `novel`: absent from training
*and* never played by this user. It is n=79, small, and it is the number worth
defending.

**The capacity comparison is not fair, and the direction matters.**
`text_frozen` has **0.47M trainable parameters against id's 22.43M**, because
the text matrix is frozen and only the 384->128 projection and the transformer
blocks train. A 47x smaller trainable model buys the ability to score unseen
items and pays for it on seen ones. That trade is the finding, not a footnote.

**Genre tags barely moved ranking quality, and that is the most interesting
result here.** Adding tags to 98.3% of items improved training loss
substantially (3.65 -> 3.16 at matched epochs) but moved HR@10 by only ~7%
relative on most slices, and on `cold_start` it went *down*, 0.1392 to 0.0886.
That last figure is 11 hits versus 7 out of 79, and HR@20 is a wash
(0.2532 vs 0.2405), so the honest statement is **no reliable difference**, not
a regression.

The likely mechanism is the tagging strategy itself. Tags were fetched per
*artist*, so every track by an artist gets identical text. That sharpens the
separation between artists and destroys discrimination *within* an artist, and
a large share of next-track prediction is choosing among tracks by an artist
you are already listening to. Lower training loss with flat ranking quality is
what you would expect if the model got better at a distinction that the metric
does not reward. Per-track tags would test this directly and cost 8.5x more
requests.

**Neither variant is the answer alone.** A deployed system wants ID embeddings
for catalog it knows and text for everything else.

---

## The personal export: transfer to a catalog the model has never seen

Everything above is one corpus. This section is a second, genuinely disjoint
one: a personal Spotify extended streaming history, which is what the project
was sequenced around waiting for.

**Corpus.** 191,783 parsed plays, **167,277 positives** at the 30-second
threshold, 2023-03-26 to 2026-08-16 (plus eight stray plays from 2017).
**One user.** 18,450 distinct tracks, 5,676 artists, 7,175 sessions. Repeat
rate **0.9435**. Unlike lastfm-1K this export carries real `ms_played`,
`skipped` (16.8%) and `shuffle` (66.9%).

**Overlap with the pretraining catalog is 6.51% of items, 9.71% play-weighted**
— roughly 90% of these plays are on tracks the pretrained model has never seen.
That is not a data problem. It is the cold-start condition the transfer design
has claimed to solve since Phase 2, and the first time it could be tested with
real statistical power: on lastfm the `cold_start` slice was n=79, here it is
**n=2,601**.

### HR@10 on the personal corpus

Catalog 18,450 items (`min_count=1`), full-catalog ranking, pessimistic ties,
20,000 evaluation instances. **Chance HR@10 = 0.000542.**

| slice | n | popularity | repeat | zero-shot | scratch (id) | scratch (text) | fine-tuned |
|---|---|---|---|---|---|---|---|
| overall | 20,000 | 0.0046 | 0.0191 | 0.0202 | **0.0343** | 0.0204 | 0.0267 |
| repeat | 17,399 | 0.0053 | 0.0219 | 0.0210 | **0.0394** | 0.0228 | 0.0284 |
| **novel** | 2,601 | 0.0000 | 0.0000 | 0.0142 | 0.0000 | 0.0046 | **0.0146** |
| cold_item | 9,795 | 0.0000 | 0.0120 | 0.0175 | 0.0000 | 0.0056 | **0.0177** |
| **cold_start** | 2,601 | 0.0000 | 0.0000 | 0.0142 | 0.0000 | 0.0046 | **0.0146** |

Four models, one harness, identical instances:

- **zero-shot** — the lastfm-pretrained `text_frozen` encoder re-pointed at this
  catalog by `melochron/train/transfer.py`, with **no gradient step taken on
  this user**. Only the item table is catalog-shaped; the learned 384->128
  projection and the transformer blocks are the same size for any catalog, so 37
  tensors transfer and the new catalog arrives as a new text matrix.
- **scratch (id)** — ID embeddings, trained on this user's data alone.
- **scratch (text)** — same architecture and text vectors as `fine-tuned`, but
  randomly initialized. The control that isolates pretraining from
  representation.
- **fine-tuned** — the zero-shot model, then trained on this user's train period.

### The two winners are in different columns

**On seen tracks, learning your own catalog wins.** `scratch (id)` takes
`repeat` at 0.0394 against zero-shot's 0.0210, nearly 2x. Nothing about text
representations beats simply memorizing what this person actually replays.

**On unseen tracks, it scores exactly zero — and that is a coverage statement,
not a loss.** A track absent from the training period keeps its random-init
embedding row, and because those rows have small norm (std 0.02) they produce
systematically low logits. They are not ranked badly; they are *suppressed*.
The ID model cannot participate in the `novel`, `cold_item` or `cold_start`
columns at all.

**The zero-shot model, which has never seen this person or this catalog, is the
only thing that scores there — 0.0142 against 0.0000, at 26x chance.** That is
the transfer claim, tested against real data at a sample size that supports it.

**Pretraining, not the text representation, is what buys cold start.**
`scratch (text)` is the control that separates the two: identical architecture,
identical text vectors, identical config, randomly initialized instead of
loaded. It reaches 0.0046 on `novel` against zero-shot's 0.0142 — a **3.1x gap
attributable entirely to weights learned from 992 other listeners**, because
nothing else differs. Text representation alone is what lifts the floor off zero,
since an ID model cannot rank an unseen track at all; but pretraining on other
people's listening is what makes the capability useful rather than nominal. The
same ordering holds on `cold_item`: 0.0056 against 0.0175.

Read together with the fine-tuning result, the ranking on unseen tracks is
**pretraining >> text representation > adaptation**, which is close to the
opposite of where the engineering effort naturally wants to go.

**Fine-tuning bought almost nothing on the slice that matters.** `novel` moves
0.0142 -> 0.0146: four extra hits out of 2,601, 2.8% relative. The plan of
record set the bar explicitly — *"per-user fine-tuning ships only if a measured
delta justifies it"* — and at k=10 it does not. Two honest qualifications: at
HR@**20** the same comparison is 0.0238 -> 0.0311, a 31% gain, so the benefit is
real but k-dependent; and fine-tuning clearly helps on *seen* items, +35% on
`repeat`. The deployable reading is that zero-shot is sufficient for cold start
and adaptation earns its keep only on catalog the user already plays.

### Why these numbers are an order of magnitude below the lastfm table

They are not comparable, and the reason is measurable rather than a matter of
opinion. The repeat baseline alone falls from 0.1797 to 0.0219 between corpora:

| | lastfm-1K | personal |
|---|---|---|
| median user history | 11,580 | **167,277** |
| repeats visible within `max_len=200` | 45.2% | **29.5%** |
| repeat HR@10 when the target **is** in the window | 0.3970 | **0.0742** |
| repeat HR@10 when it **is not** | 0.0002 | 0.0000 |
| distinct items per 200-play window | 111 | **181** |

Two effects compound. One listener's history is 14x longer than a typical
lastfm user's, so a 200-event context reaches proportionally much less far and
**70.5% of "repeat" targets were last played outside it** — invisible to every
scorer, neural or not. And when the target *is* in the window it is 5.4x harder
to rank, because these windows hold 181 distinct tracks per 200 plays against
lastfm's 111, with two thirds of plays shuffled. Recency is a far weaker signal
against varied, shuffled listening.

Neither is a modelling regression. Both are properties of predicting one
person's next track over a long, diverse, shuffle-heavy history, and `max_len`
is the binding constraint rather than model capacity.

### What one user breaks, and what the tables do about it

A single-user corpus degrades several things silently, so the harness was
changed to refuse to hide them:

- **`novel` and `cold_start` are the same instances**, exactly. `is_repeat` is
  computed over a prefix containing the whole training period, so `not repeat`
  implies `not in train_items`. On lastfm these were different populations
  *because 991 other people trained those item embeddings*.
- **`cold_user` cannot be populated** — there is no second user to hold out. The
  table names it as not measurable rather than dropping the row.
- **Popularity and item-kNN are structurally 0.0000 on `novel`.** Both are
  collaborative; with one listener there is no collaborative signal, so a novel
  target has count zero and an all-zero similarity column. On lastfm item-kNN
  scored 0.1811 there — entirely because other people had played those tracks.
  The report annotates these zeros rather than letting a model look decisive
  against them.
- **`--max-per-user` is a per-user cap**, so on one user it silently becomes the
  size of the whole evaluation. Script defaults would have produced a
  normal-looking six-slice table computed on **23 instances**. It now warns.
- **Training defaults collapse too.** At the default stride the fit split yields
  640 windows and **6 optimizer steps per epoch** against lastfm's ~750;
  `configs/personal.yaml` uses stride 25 for 5,119 windows and 40 steps, raises
  epochs and patience to match, and switches to uniform negatives because
  popularity-weighted negatives drawn from a single listener's own counts put
  the positive in its own softmax denominator.
- Every table prints its **chance level**, because full-catalog HR against
  18,450 candidates and against 171,902 are different questions that look alike.

### Insights on a personal history

The Phase 6 modules run against this corpus with the signals lastfm never had.

**Taste drift**, 30-day windows over 43 months. Median monthly step is 0.033
while cumulative displacement reaches 1.66 — slow continuous drift rather than
mood cycling. The three-year dormancy before 2023 is emitted as a marked gap
with `since_previous` recording the jump, not interpolated into a smooth line.

**Session archetypes** separate far more cleanly than on lastfm (silhouette
0.241 against 0.071), and the playback signals are what make them nameable:

| cluster | share | peak (UTC) | length | dwell | skip | shuffle |
|---|---|---|---|---|---|---|
| late-night deep listening | 11% | 00h | 56.4 | 176s | 0.06 | 0.60 |
| deliberate evening | 31% | 20h | 26.8 | 219s | 0.10 | **0.50** |
| shuffled daytime background | 59% | 14h | 25.5 | 201s | 0.09 | **0.76** |

Shuffle rate spans 0.50 to 0.76 across clusters and dwell spans 175s to 219s.
Neither field exists in lastfm-1K, which is why these were declined in Phase 6
and are only now measurable.

---

## Design decisions that shaped the numbers

**Pretrain on 992 strangers, fine-tune on you.** One person's history is too
little data to train a transformer, and it makes the two-stage serving design
impossible: there is no population to pretrain on. It also decouples the project
from a 30-day Spotify export wait.

**Text item vectors, not IDs alone.** This is what makes cold start solvable at
all rather than a special case to apologize for.

**Last.fm tags, not Spotify.** Spotify's February 2026 changes removed the batch
metadata endpoints and dropped search `limit` from 50 to 10. But the decisive
reason is different: Spotify tags would cover a personal export and not the
pretraining corpus, splitting the item representation into two incompatible
spaces. Last.fm covers both.

**Global temporal split, plus a separate user holdout.** Two orthogonal axes.
Time answers "does it predict this user's future". User holdout answers "does it
work for someone it never trained on". A global cut alone puts every user's
pre-cutoff events into training, which silently turns the cold-start slice into
a warm-start slice.

**Catalog built from the full frame; counts and fit from training only.** The
catalog is the universe of rankable items, not a learned parameter, so choosing
it globally is not leakage. Building it from training instead would make every
in-vocabulary target train-seen by construction and silently empty the
`cold_item` slice. Popularity counts and all model fitting stay train-only,
which is what keeps that defensible.

**Validation cut inside the training period.** Early stopping on test data leaks
invisibly: the model looks like it generalizes because it was selected for
looking that way.

**Time-interval-aware attention.** Inter-event gaps are bucketed on a log scale
and injected alongside position, in the spirit of TiSASRec (Li, Wang, McAuley,
WSDM 2020). A three-month gap is a different context from thirty seconds.

---

## Serving

Measured on the deploy target (CPU), full 171,902-item catalog, 200-event
history:

| p50 | p95 | p99 |
|---|---|---|
| 8.2 ms | 9.35 ms | 15.71 ms |

**Cold start is not a special branch.** SASRec has no per-user parameters: it
conditions on the sequence, not a user id. A new uploader needs no per-user
fitting, no user embedding, no retraining. Map their history to item ids, run
the shared model, score. Calling that "per-user adaptation" would be an
overclaim; if adaptation is added it arrives with a measured delta against this
path.

---

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e ".[dev,serve]"

python scripts/download_lastfm1k.py                     # 672 MB, HTTPS required
python scripts/build_dataset.py --data lastfm1k --path data/raw/lastfm-1k
python scripts/run_baselines.py --data parquet --path data/interim/lastfm1k-v1.parquet --min-count 20

python scripts/build_embeddings.py --path data/interim/lastfm1k-v1.parquet --min-count 20
python scripts/train.py --config configs/pretrain.yaml --data parquet \
    --path data/interim/lastfm1k-v1.parquet --min-count 20
python scripts/bench_latency.py --checkpoint artifacts/runs/id-real/best.pt
```

### The personal export

Drop a Spotify extended streaming history under `data/my_data/` and the same
pipeline runs on it unchanged. `configs/personal.yaml` carries the settings a
single-user corpus forces; the defaults would produce a plausible table computed
on 23 instances.

```bash
python scripts/build_dataset.py --data spotify --path data/my_data --out data/interim
python scripts/run_baselines.py --data parquet --path data/interim/spotify-v1.parquet     --min-count 1 --max-per-user 20000 --holdout-user-frac 0

python scripts/build_embeddings.py --path data/interim/spotify-v1.parquet     --min-count 1 --fetch-tags --tag-mode artist

# from scratch, and fine-tuned from the lastfm pretrain
python scripts/train.py --config configs/personal.yaml --data parquet     --path data/interim/spotify-v1.parquet --min-count 1 --max-per-user 20000     --holdout-user-frac 0 --name personal-id
python scripts/train.py --config configs/personal.yaml --data parquet     --path data/interim/spotify-v1.parquet --min-count 1 --max-per-user 20000     --holdout-user-frac 0 --variant text_frozen     --text-vectors data/embeddings/latest.npy     --init-from artifacts/runs/text-frozen-real/best.pt --name personal-finetuned

# every row on identical instances, zero-shot included
python scripts/run_transfer.py --path data/interim/spotify-v1.parquet     --text-vectors data/embeddings/latest.npy     --pretrained artifacts/runs/text-frozen-real/best.pt     --checkpoint scratch-id=artifacts/runs/personal-id/best.pt     --checkpoint fine-tuned=artifacts/runs/personal-finetuned/best.pt

python scripts/build_insights.py --data parquet --path data/interim/spotify-v1.parquet     --checkpoint artifacts/runs/personal-id/best.pt --window-days 30
```

Genre tags need a free Last.fm API key in `.env` as `LASTFM_API_KEY`. Without
one the pipeline runs on names only, which is a supported configuration rather
than a failure. The fetched tag cache is committed, because it represents hours
of rate-limited requests: 25,325 artists, 86.7% item coverage on the personal
catalog.

---

## Limitations

- **`cold_start` is n=79 on lastfm-1K.** The mechanism is demonstrated there;
  the number is not precise. The personal export gives the same slice at
  **n=2,601**, which is where the transfer claim is actually measured.
- **Both variants are undertrained.** Validation was still improving at epoch 10
  in each run. These are floors.
- **Artist-level tags.** 20,208 artists cover 171,902 items at 98.3% coverage
  for 8.5x fewer requests than per-track, but every track by an artist gets
  identical tags, so tags do not distinguish tracks within an artist.
- **9% of artists have no tags**, and they skew obscure, so tag signal is
  weakest exactly where cold-start problems live.
- **Corpus era.** lastfm-1K ends in 2013. Measured item overlap with the 2020s
  Spotify export is **6.51% of items, 9.71% play-weighted**, which makes it a
  hard transfer test rather than a flattering one.
- **`max_len=200` is the binding constraint on a personal corpus**, not model
  capacity. 70.5% of `repeat` targets there were last played outside the window
  and are invisible to every scorer. Every personal-corpus number is a floor set
  by context length.
- **The personal corpus has one user**, so `cold_user` is unmeasurable, `novel`
  and `cold_start` are the same instances, and the collaborative baselines are
  structurally zero on both. Those are properties of n=1, not results.
- **The personal models are undertrained by wall-clock, not by schedule.** All
  three early-stopped on CPU (best epochs 18, 26 and 10 of 60).
- **Positive-threshold asymmetry.** Spotify exports carry `ms_played` so skips
  can be filtered; Last.fm only scrobbles past ~50% of a track. Different
  implicit thresholds, not filtered versus unfiltered.

## Not yet done

The `text_finetuned` and `hybrid` ablation rows on the personal corpus, the
time-delta ablation, a `max_len` sweep (the measured binding constraint), and
deployment. The frontend that would consume the insight JSON is not built.
