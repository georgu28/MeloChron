# MeloChron

Sequential music recommendation from listening history. A causal self-attention
encoder written from scratch in PyTorch, pretrained on 19.1M real plays, then
tested on a personal Spotify export that overlaps the training catalog by 6.5%.

The item representation that works is a **frozen text prior plus a learned
per-item residual**. Neither half is sufficient and the ablation below shows why:
learned ID embeddings cannot score a track they never saw, text embeddings can
but give up most of the accuracy on tracks they did.

Aggregate metrics in this domain are dominated by replays: a model that
memorizes what you just played scores well overall and has learned nothing. So
every table here is split into *repeat* and *novel*, and the novel column is the
one that counts.

## What the numbers say

- **A model pretrained on 992 strangers ranks tracks it has never seen, in a
  catalog it has never seen, at 35x chance — with zero gradient steps on the
  user's data.** A model trained on that user's own listening scores *exactly
  zero* on the same tracks.
- **Pretraining, not the text representation, is what buys that.** An identical
  architecture with identical text vectors, trained from scratch instead of
  loaded, scores 4.2x worse.
- **Per-user fine-tuning is the largest single win in the project, and it is
  negative on the one slice that motivated the design.** Against its own
  starting point it gains **+52% overall and +60% on replays** — 87% of the test
  set, at 11 standard errors — and loses 24% on tracks the user has never
  played, 38 hits against 50 out of 2,601. Not a marginal result in either
  direction, which is why the deployable answer is two artifacts routed per
  request rather than one.
- **Every non-neural baseline scores exactly 0.0000 on the strict transfer
  slice.** Popularity, repeat and item-kNN are all collaborative or
  memorization-based, so a track absent from training is unrankable for all of
  them. The hybrid representation scores 0.2658 there.
- **The margin over item-kNN on novel tracks is real but narrows with k** —
  +57% at HR@5, +17% at HR@10, +1.7% at HR@20. The model is much better in the
  top few slots and the co-occurrence baseline has caught up by twenty.
- **The one bespoke architectural idea here does nothing, and the reason is in
  the data.** Time-interval-aware attention is a clean null on a matched pair at
  n=25,950 — because 85.6% of consecutive plays sit in two of its 24 gap buckets.
  People play one song after another; the gap is almost always the length of a
  track.
- **Genre tags were nearly worthless alone and are load-bearing in combination.**
  Tags moved HR@10 by ~7% as a representation on their own; the same tags under
  a learned residual give the best model in the project. The signal was there,
  it just could not be used without a per-item correction.

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

| slice | n | popularity | repeat | item-kNN | SASRec (id) | **SASRec (hybrid)** |
|---|---|---|---|---|---|---|
| overall | 5,334 | 0.0019 | 0.1468 | 0.2171 | 0.2553 | **0.3020** |
| repeat | 3,898 | 0.0023 | 0.2006 | 0.2304 | 0.2804 | **0.3353** |
| novel | 1,436 | 0.0007 | 0.0007 | 0.1811 | 0.1873 | **0.2117** |
| cold_user | 570 | 0.0035 | 0.1912 | 0.2018 | 0.2386 | **0.3211** |
| cold_item | 368 | 0.0000 | **0.3913** | 0.0000 | 0.0000 | 0.3315 |
| cold_start | 79 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.2658** |

**Metric:** HR@10, the hit rate in the top 10. Bold marks the best result in
each row. `hybrid` is the deployable representation, defined in the ablation
below; `id` is kept in the table because it is the control the hybrid is built
out of, not because it is a second product.

**The hybrid model wins every slice except `cold_item`,** where the repeat
baseline holds 0.3913 against 0.3315. That row is not the upset it looks like:
`cold_item` means absent from *training*, and evaluation context legitimately
includes a user's earlier test-period plays, so a user can already know a track
the model never trained on. Repeat is very good at exactly that. `cold_start`,
the intersection with `novel`, is the row without that loophole.

**Every non-neural model scores exactly 0.0000 on `cold_start`.** Popularity,
repeat and item-kNN all reduce to counting co-occurrences among tracks someone
has already played, so a track absent from the training period is not ranked
badly by them — it cannot be ranked at all. This is the row the whole transfer
argument is about, and it is the row where the baselines have nothing to say.

**Against item-kNN on `novel` the win is real and shrinks with k.**

| | HR@5 | HR@10 | HR@20 | NDCG@10 |
|---|---|---|---|---|
| item-kNN | 0.1058 | 0.1811 | 0.2472 | 0.0959 |
| hybrid | **0.1657** | **0.2117** | **0.2514** | **0.1067** |
| relative | +56.6% | +16.9% | +1.7% | +11.2% |

Reported at three k rather than one because one k would let either story be
told. The model is much better in the top five slots, and by twenty a plain
co-occurrence baseline has all but caught it. For a product that shows
ten recommendations that is the favourable half of the range, which is worth
saying out loud rather than leaving to the reader to notice.

**Repeat scoring 0.0007 on `novel` is the control working.** It is a cache; it
can only rank what you already played. Beating it there by 300x is what
separates a recommender from a lookup table. Item-kNN also clears it, which is
why item-kNN is the number to beat.

### Transfer ablation — item representation only

Same architecture, same data, same harness; only the item vectors change.

| slice | n | id | text (names) | text (+tags) | **hybrid** |
|---|---|---|---|---|---|
| overall | 5,334 | 0.2553 | 0.1376 | 0.1474 | **0.3020** |
| repeat | 3,898 | 0.2804 | 0.1632 | 0.1750 | **0.3353** |
| novel | 1,436 | 0.1873 | 0.0682 | 0.0724 | **0.2117** |
| cold_user | 570 | 0.2386 | 0.1754 | 0.1807 | **0.3211** |
| **cold_item** | 368 | 0.0000 | 0.2500 | 0.2745 | **0.3315** |
| **cold_start** | 79 | 0.0000 | 0.1392 | 0.0886 | **0.2658** |

`hybrid` is `item_vectors = projection(text) + residual[id]`, and its text half
is the **same tagged matrix as the `text (+tags)` column** — verified by
comparing the buffers in the two checkpoints, not by trusting the run names. So
that pair isolates one thing: what a learned per-item residual adds to a fixed
text prior.

**The residual does not merely recover what text gave up, it overtakes ID
embeddings.** Against its own text prior it is 2.05x overall and 3.00x on
`cold_start`; against `id` it is +18.3% overall while going from a structural
zero to 0.2658 on `cold_start`. The two representations are not competing
accounts of an item. Text says what an item *is*, interaction data says what
listeners *do* with it, and neither is recoverable from the other.

**The residual is zero-initialized, and that is the whole mechanism.** A row
that never receives gradient stays exactly zero, so the item falls back to pure
text with nothing corrupting it. Cold items never receive gradient: they are
never positives, by definition of being absent from the training period, and
never negatives, because popularity sampling weights by training count and a
count of zero is drawn with probability zero. Random initialization would add
noise to precisely the items with no signal to override it. On the trained
artifact, **7,306 of 171,904 rows are still exactly zero**, which is the
guarantee observed rather than argued.

Item vectors are also L2-normalized before scoring. That sounds cosmetic and is
not: in the first hybrid run every cold item scored a flat 0.0000 despite the
residual guarantee holding exactly. Trained items had grown to norm 1.52 against
0.56 for pure-text items, and since scoring is a dot product that 2.7x length
difference alone kept cold items out of the top 10 against 164k trained
competitors. Their direction had been right the whole time.

**Every `0.0000` in an ID column is a coverage statement, not a loss.** A track
absent from training has no row in the item table — it is not ranked badly, it
*cannot be ranked*. (Precisely: untrained rows keep their small init norm, so
they produce low logits and are suppressed.) Reading those cells as "the ID
model is bad at cold items" is wrong.

**The capacity comparison is not fair, and the direction matters.**
`text_frozen` trains **0.47M parameters against id's 22.43M**, because the text
matrix is frozen and only the 384→128 projection and the blocks train. A 47x
smaller trainable model buys unseen-item scoring and pays for it on seen ones.

`hybrid` trains **22.48M**, so against `id` it is not a capacity story at all:
essentially the same budget, spent on correcting a prior instead of on learning
a table from scratch, for +18.3% overall and the cold-start slice going from
impossible to 0.2658.

**Genre tags barely moved ranking on their own, and are load-bearing in
combination.** As a representation, tags on 98.3% of items cut training loss
(3.65 → 3.16 at matched epochs) but moved HR@10 ~7% relative, and `cold_start`
went *down* — 11 hits vs 7 out of 79, with HR@20 a wash. Read alone that is **no
reliable difference**.

The diagnosis at the time was that tags are fetched per *artist*, so every track
by an artist gets identical text: sharper between-artist separation, destroyed
within-artist discrimination, and much of next-track prediction is choosing
among tracks by the artist already playing. The hybrid column is that diagnosis
tested. Give the model a per-item residual — exactly the within-artist
distinction artist-level tags cannot express — and the same tagged vectors go
from 0.1474 to 0.3020. The tag signal was real and simply unusable without
somewhere to put the correction, which is a more useful conclusion than "tags
did not help" and would have been invisible without running the combination.

---

## Corpus 2 — personal Spotify export, HR@10

191,783 parsed plays → **167,277 positives** (≥30s), 2023-03 to 2026-08 plus
eight stray 2017 plays. **One user**, 18,450 tracks, 5,676 artists, 7,175
sessions, repeat rate **0.9435**. Unlike lastfm-1K it carries real `ms_played`,
`skipped` (16.8%) and `shuffle` (66.9%).

**Overlap with the pretraining catalog: 6.51% of items, 9.71% play-weighted.**
On lastfm the `cold_start` slice was n=79; here it is **n=2,601**, so the
transfer claim finally has statistical power.

Catalog 18,450 items (`min_count=1`), 20,000 instances, every row below scored
in one invocation of `scripts/run_transfer.py` so all eight models meet the same
instances. **Chance HR@10 = 0.000542.**

| slice | n | popularity | repeat | scratch (id) | zero-shot | **fine-tuned** |
|---|---|---|---|---|---|---|
| overall | 20,000 | 0.0046 | 0.0191 | 0.0343 | 0.0286 | **0.0435** |
| repeat | 17,399 | 0.0053 | 0.0219 | 0.0394 | 0.0300 | **0.0479** |
| novel | 2,601 | 0.0000 | 0.0000 | 0.0000 | **0.0192** | 0.0146 |
| cold_item | 9,795 | 0.0000 | 0.0120 | 0.0000 | **0.0216** | 0.0197 |
| **cold_start** | 2,601 | 0.0000 | 0.0000 | 0.0000 | **0.0192** | 0.0146 |

`zero-shot` and `fine-tuned` are both the **hybrid** representation.

| column | what it is |
|---|---|
| **scratch (id)** | ID embeddings trained on this user alone, from noise |
| **zero-shot** | the lastfm-pretrained hybrid encoder re-pointed at this catalog, **no gradient step on this user** |
| **fine-tuned** | that same transplant, then trained on this user's train period |

The transplant works because only the item table is catalog-shaped: the learned
384→128 projection and the transformer blocks are the same size for any catalog,
so 38 tensors transfer and the new catalog arrives as a new text matrix plus an
all-zero residual (`melochron/train/transfer.py`).

**Fine-tuned hybrid is the first model here to be good at both jobs.** It takes
`overall` at 0.0435 against `scratch (id)`'s 0.0343 — +27.0%, 6.4 standard
errors at n=20,000 — while scoring 0.0146 on tracks `scratch (id)` cannot rank
at all. Every earlier configuration forced a choice between the two.

**Measured as adaptation rather than against a different model, the effect is
larger still.** The honest question about fine-tuning is what it adds to the
checkpoint it started from, and against zero-shot it is +52% on `overall` and
**+60% on `repeat`** at 11 SE. Replays are 87% of this test set, so that is most
of what a user of this system would actually experience:

| slice | share | zero-shot | fine-tuned | |
|---|---|---|---|---|
| repeat | 87.0% | 0.0300 | **0.0479** | +60% |
| novel | 13.0% | **0.0192** | 0.0146 | −24% |

**And on cold start it is still beaten by doing nothing.** Zero-shot scores
0.0192 there against fine-tuned's 0.0146, a 24% drop from adapting to the user
(2.0 SE). The same thing happens to the text variant, −19.1%, and text has no
residual, so this is the general cost of specializing an encoder on one
listener's catalog rather than anything specific to the hybrid.

Worth keeping the two statements apart, because collapsing them says something
false. "Fine-tuning did not work" is wrong: it worked enormously, on 87% of the
test set. What is true is narrower — fine-tuning does not buy cold start, and
costs a little there.

The deployable reading is unchanged in shape and sharper in detail: **fine-tune
for the catalog someone already plays, keep the zero-shot model for discovery.**
Two artifacts, chosen per request, is a real design conclusion drawn from a
measurement rather than an architecture diagram.

### A correction to what this section used to say

It previously reported that fine-tuning "bought 4 extra hits out of 2,601" on
`cold_start`. That comparison was against a broken baseline and the sign is
wrong.

The zero-shot row was transplanted from `text-frozen-real`, whose projection was
trained on text with `tag_coverage: 0.0` — bare `"Track by Artist"` strings —
and then pointed at personal vectors with `tag_coverage: 0.8672`, which carry
`Genre: ...` clauses that projection had never seen. Nothing errors; the numbers
just come out low. Scored from a pretrain whose text matches, `zero-shot` on
`cold_start` is **0.0181, not 0.0142** — 47 hits rather than 37.

Against a matched baseline, fine-tuning does not buy 4 hits. It **loses 9**.

| | cold_start HR@10 | hits of 2,601 |
|---|---|---|
| zero-shot, names-only pretrain (what was published) | 0.0142 | 37 |
| zero-shot, matched tagged pretrain | **0.0181** | **47** |
| fine-tuned text | 0.0146 | 38 |

The design bar was "adaptation ships only on a measured delta". It does not
merely fail to clear the bar; measured properly it is negative on the slice that
motivated the whole transfer design.

### Text against hybrid, matched on both sides

Same pretraining corpus, same tagged text, same instances; only the item
representation differs.

| | overall | cold_item | cold_start |
|---|---|---|---|
| zero-shot text | 0.0261 | 0.0210 | 0.0181 |
| **zero-shot hybrid** | **0.0286** | **0.0216** | **0.0192** |
| fine-tuned text | 0.0267 | 0.0177 | 0.0146 |
| **fine-tuned hybrid** | **0.0435** | **0.0197** | 0.0146 |

Before running this the prediction recorded in this README was that zero-shot
hybrid should beat zero-shot text, because on a new catalog every residual row
is zero and hybrid's per-row normalization is exactly the correction that regime
needs. **The direction held at every k and only the aggregate is out of the
noise**: +9.6% on `overall` is 2.1 SE, while `cold_start` is 50 hits against 47,
which is 0.4 SE and means nothing on its own. A prediction that comes true by
three hits has not been confirmed, and is recorded here as directional.

**Pretraining, not text representation, is what buys cold start.**
`scratch (text)` differs from `zero-shot` only by initialization and reaches
0.0046 against 0.0192 — a **4.2x gap attributable to weights learned from 992
other listeners**. Text lifts the floor off zero; pretraining makes the
capability useful rather than nominal.

On unseen tracks the ranking is **pretraining >> item representation >
adaptation**, with adaptation actively negative — close to the reverse of where
engineering effort naturally goes.

### The zero-residual guarantee has a precondition, and this config broke it

`item_repr.py` argues that a cold item falls back to its pure text vector
because its residual never receives gradient: never a positive, since it is
absent from the training period, and never a negative, since popularity sampling
weights by training count and a count of zero is drawn with probability zero.

**That second clause is a property of the negative sampler, not of the model.**
`configs/personal.yaml` sets `popularity_negatives: false`, for a good and
unrelated reason — with a single listener, popularity-weighted negatives drawn
from that listener's own counts put the positive into its own softmax
denominator. Under uniform sampling every row is reachable:

| | negatives | residual rows still exactly zero |
|---|---|---|
| lastfm pretrain | popularity^0.75 | 7,306 of 171,904 |
| personal fine-tune | uniform | **3 of 18,452** |

At 40 steps per epoch, 512 shared negatives and 60 epochs, that is 1.2M draws
over 18,452 items: the chance a given row is never touched is ~1e-29, and the
three survivors are the rows that structurally cannot receive gradient. Cold
items are never positives, so what they collect is exclusively *negative*
gradient — pushed away from every context they appear in and never pulled toward
one.

Two configuration choices, each defensible alone, whose interaction disables the
mechanism the representation is built around. Neither file mentions the other.
This is offered as a diagnosis and not as the explanation of the cold-start drop
above, because the text variant has no residual at all and dropped too.

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

**Text item vectors, not IDs alone — then both.** Text is what makes cold start
solvable rather than a special case to apologize for, and the ablation was
supposed to settle which representation to ship. It settled something better:
both, because the failure modes are complementary rather than ranked. The
deployed representation is a frozen text prior with a zero-initialized per-item
residual on top, which is the only variant that is never structurally unable to
score an item and never gives up accuracy on items it knows.

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

**Time-interval-aware attention, measured and not retained as a claim.**
Inter-event gaps are log-bucketed into 24 buckets from one second to one year
and injected alongside position, in the spirit of TiSASRec (Li, Wang, McAuley,
WSDM 2020). The reasoning was that a three-month gap is a different context from
thirty seconds. The reasoning is fine and the feature does not pay.

Two hybrid runs, identical in every setting except `use_time`, both scored
through `scripts/evaluate.py` on one instance set of **25,950** — not on their
own training-time tables, which were built over different instance counts and
are not comparable:

| slice | n | time | no time | delta |
|---|---|---|---|---|
| overall | 25,950 | 0.3131 | 0.3139 | −0.2% |
| repeat | 19,466 | 0.3361 | 0.3363 | −0.1% |
| novel | 6,484 | 0.2441 | 0.2465 | −0.9% |
| cold_user | 2,840 | 0.3430 | 0.3454 | −0.7% |
| cold_item | 1,644 | 0.3613 | 0.3534 | +2.2% |
| cold_start | 311 | 0.3151 | 0.2958 | +6.5% |

HR@10. **Removing the time encoding changes nothing.** Four of six slices are
very slightly better without it, and every difference on the large slices is
under a percent — against a standard error of ~0.3% on `overall`, which is to
say inside the noise floor. The two positive cells do not survive contact with
other k: `cold_start` is +6.5% at k=10, −3.7% at k=5 and −3.1% at k=20, and at
n=311 that is 98 hits against 92. Six hits is not a finding.

**The interesting part is why, and it is not subtle.** The encoding spans one
second to one year, and the corpus does not:

| | lastfm-1K | personal |
|---|---|---|
| median inter-event gap | 240 s | 204 s |
| share in the top 2 buckets | 85.6% | 82.8% |
| buckets holding >1% of gaps | 5 of 24 | 8 of 24 |
| bucket entropy | 1.71 bits | 1.51 bits |
| gaps over 1 hour | 3.88% | 2.78% |

People listen to music one track after another. The gap between consecutive
plays is the length of a song — 91.1% of all gaps fall between thirty seconds
and ten minutes — so a feature designed to distinguish thirty seconds from three
months spends almost all of its time reading the same value. Of 4.58 bits
available across 24 buckets the realised distribution carries 1.71, and the
personal export is *worse*, not better, so this is not an artifact of the
pretraining corpus being old or scrobble-shaped.

It costs 3,072 parameters, 0.014% of the model, so keeping it is close to free
and removing it would buy nothing either. It stays in the code and stays on by
default; what changes is that it is described here as a measured null instead of
as a design feature, because the alternative is a README crediting an idea for
accuracy it did not produce.

Worth being clear about what this does not show. Inter-event time is not
useless *in principle* — it is uninformative in a corpus where the gap is nearly
constant. A dataset with genuine dormancy structure, or a formulation using
absolute time-of-day and day-of-week rather than gaps between consecutive plays,
would be a different experiment. Session archetypes below do separate cleanly on
hour-of-day, which is evidence that temporal signal exists in this data and that
the gap between adjacent plays is simply the wrong place to look for it.

---

## Serving

A FastAPI service loads a versioned artifact at startup, parses uploads
asynchronously, and scores the full catalog in one batched pass.
`scripts/bench_latency.py`, CPU, 4 torch threads, full 171,902-item catalog,
200-event history, 400 requests:

| variant | p50 | p95 | p99 |
|---|---|---|---|
| id | 24.00 ms | 27.70 ms | 31.72 ms |
| text_frozen | 24.04 ms | 29.25 ms | 30.04 ms |
| **hybrid** | **24.38 ms** | **31.60 ms** | **34.05 ms** |

**Read the comparison, not the absolute.** This is a loaded development laptop,
and across repeated trials the same artifact measured anywhere from 14 to 24 ms
p50 depending on what else was running. What is stable across every trial is
that the three variants land within about a millisecond of each other at p50.
The deployable representation costs essentially nothing over the ID control,
which is the claim worth making; a p50 measured on a laptop is not a production
SLO and is not offered as one.

**The benchmark was measuring a configuration nothing runs.** The service bounds
torch's intra-op pool to four threads, because several concurrent requests each
spawning a full-width pool oversubscribe the cores and degrade p95 far more than
they improve throughput. `bench_latency.py` did not apply that bound, so it let
torch size the pool to the whole machine: 23.2 ms p50 for an artifact the
service itself served in 9.6 ms. It now applies the same bound.

### What it cost to serve the model the project argues for

The hybrid artifact first measured **538 ms p50** per recommendation through the
endpoint, against 9.6 ms for `id` in the same session. Both causes were the same
mistake — doing catalog-scale work to answer a question about 200 plays:

| | p50 |
|---|---|
| as merged | 538.30 ms |
| item table materialized once at load | 271.02 ms |
| gather before projecting, not after | **12.01 ms** |

The item table was being rebuilt per request. `SASRecScorer.score` hoists it out
of its batch loop, which amortizes to nothing over a full-catalog evaluation of
thousands of instances and to nothing at all when a request carries one history.
It is now materialized once, at load, with invalidation handled structurally
rather than by convention: `train()` and `_apply()` both drop the cached table,
so entering training mode or moving device cannot serve vectors that no longer
match the weights.

The larger half was subtler. `ProjectedTextEmbedding` inherited its `forward`
from the base class, which builds the entire `[171904, 128]` table and then
indexes 200 rows out of it. Gathering first is the same arithmetic — indexing
rows commutes with an unbiased right multiplication — and 860x less of it. That
one was also being paid on every training step of the text and hybrid variants,
where the sequence and its sampled negatives are both looked up this way: 175 ms
to 46 ms per forward-and-backward over a 128x200 batch, 4x.

Worth stating plainly because it is the kind of thing an offline metric never
catches. Both variants had been producing correct numbers the whole time.

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
python scripts/build_embeddings.py --path data/interim/lastfm1k-v1.parquet \
    --min-count 20 --fetch-tags --tag-mode artist

P=data/interim/lastfm1k-v1.parquet
V=data/embeddings/latest.npy

# the rows of the ablation; --variant selects one
python scripts/train.py --config configs/pretrain.yaml --data parquet --path $P \
    --min-count 20 --variant id --name id-real
python scripts/train.py --config configs/pretrain.yaml --data parquet --path $P \
    --min-count 20 --variant text_frozen --text-vectors $V --name text-tagged
python scripts/train.py --config configs/pretrain.yaml --data parquet --path $P \
    --min-count 20 --variant hybrid --text-vectors $V --name hybrid-norm

python scripts/bench_latency.py --checkpoint artifacts/runs/hybrid-norm/best.pt

# serve the artifact the tables are about
MELOCHRON_CHECKPOINT=artifacts/runs/hybrid-norm/best.pt \
    uvicorn melochron.serving.app:app --port 8000
```

Baseline output is named after the corpus rather than the `--data` flag, so a
personal export and the pretraining corpus cannot overwrite each other's
tables, which they silently did once.

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
  directional. `cold_start` at n=79 is the one to treat most cautiously: the
  hybrid result there is 21 hits, against 7 for the identical text vectors used
  without a residual.
- **The time ablation is a single matched pair, one seed.** It is a clean null
  at n=25,950 and the mechanism explains it, but two runs cannot distinguish a
  true zero from a small effect this corpus cannot express.
- **Latency is measured on a development laptop**, not on a deploy target, and
  the absolute figures move with machine load. The variant comparison is stable;
  the absolute is not an SLO.

## Not yet done

- **A negative sampler that keeps the zero-residual guarantee on one user.**
  Popularity weighting is wrong at n=1 and uniform sampling touches every row;
  something that excludes count-zero items without weighting by a single
  listener's counts would let the hybrid keep its cold-start fallback while
  adapting. Not attempted.
- **A `max_len` sweep**, the measured binding constraint on a personal corpus.
- **Deployment.** The service runs locally and is not hosted anywhere.
- **A frontend for the insight JSON.** The upload-and-recommend surface exists;
  drift, archetypes and attention are computed but only ever written to disk.
