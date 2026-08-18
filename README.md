# MeloChron

Sequential music recommendation from listening history. A causal self-attention
encoder built from scratch in PyTorch, pretrained on 19.1M real listening
events, evaluated against non-neural baselines under a protocol chosen to make
the result hard to flatter.

The headline is not the best number. It is the decomposition: aggregate metrics
in this domain are dominated by replays, and a model that memorizes your recent
plays scores well overall while learning nothing. Every table below is split so
that cannot hide.

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

**Metric:** HR@10 (hit rate in top 10 recommendations). Bold marks the best result per row, except `Cold item`: SASRec structurally can't score there (no ID for unseen tracks), so repeat wins by default, not by skill.

**Slice definitions:**
- `Overall`: all test cases combined
- `Repeat`: user has played this track before
- `Novel`: user has not played this track before
- `Cold user`: user held out entirely from training
- `Cold item`: track absent from the training period

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

Genre tags need a free Last.fm API key in `.env` as `LASTFM_API_KEY`. Without
one the pipeline runs on names only, which is a supported configuration rather
than a failure. The fetched tag cache is committed (372 KB), because it
represents 84 minutes of rate-limited requests.

---

## Limitations

- **`cold_start` is n=79.** The mechanism is demonstrated; the number is not
  precise.
- **Both variants are undertrained.** Validation was still improving at epoch 10
  in each run. These are floors.
- **Artist-level tags.** 20,208 artists cover 171,902 items at 98.3% coverage
  for 8.5x fewer requests than per-track, but every track by an artist gets
  identical tags, so tags do not distinguish tracks within an artist.
- **9% of artists have no tags**, and they skew obscure, so tag signal is
  weakest exactly where cold-start problems live.
- **Corpus era.** lastfm-1K ends in 2013. Item overlap with a 2020s Spotify
  export is near zero, which makes it a hard transfer test rather than a
  flattering one.
- **Positive-threshold asymmetry.** Spotify exports carry `ms_played` so skips
  can be filtered; Last.fm only scrobbles past ~50% of a track. Different
  implicit thresholds, not filtered versus unfiltered.

## Not yet done

Per-user fine-tuning on a personal export, the `text_finetuned` ablation row,
the time-delta ablation, the insights surface (drift, session archetypes,
attention visualization), and deployment.
