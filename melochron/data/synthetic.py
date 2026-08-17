"""Synthetic listening-history generator.

Stands in for the Spotify export until it arrives, and stays useful afterwards
as fast, deterministic test data. It is built to reproduce the four properties
that the modelling decisions actually hinge on:

1. **Repeat structure.** Plays are heavily recency-biased, so a repeat baseline
   is genuinely hard to beat. If the synthetic data lacked this, Phase 1 would
   report a flattering number and the central honesty check of the project
   would be untestable until real data landed.
2. **Session bursts.** Plays cluster into sittings separated by long gaps, so
   sessionization and the inter-event time encoding have signal to find.
3. **Taste drift.** Each user's preference vector random-walks over time, so
   the Phase 6 drift metric has ground truth to be checked against.
4. **Skips.** ``ms_played`` is short for low-affinity items, so the >30s
   positive filter removes something real rather than being a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from melochron import schema

_ADJ = [
    "velvet",
    "midnight",
    "electric",
    "paper",
    "golden",
    "hollow",
    "crystal",
    "quiet",
    "neon",
    "wandering",
    "bitter",
    "slow",
    "crooked",
    "silver",
    "distant",
    "amber",
    "restless",
    "cardinal",
    "pale",
    "iron",
    "glass",
    "northern",
    "lonesome",
    "violet",
]
_NOUN = [
    "circuit",
    "arithmetic",
    "harbor",
    "machine",
    "orchard",
    "signal",
    "atlas",
    "tide",
    "chapel",
    "engine",
    "static",
    "meridian",
    "cassette",
    "lantern",
    "avenue",
    "ledger",
    "mountain",
    "parade",
    "wireless",
    "foxglove",
    "compass",
    "hymn",
    "corridor",
    "ember",
]
_VERB = [
    "waiting",
    "burning",
    "falling",
    "running",
    "drifting",
    "breaking",
    "turning",
    "holding",
    "leaving",
    "chasing",
    "sinking",
    "rising",
    "counting",
    "fading",
]

#: One tag vocabulary per latent genre cluster, so the synthetic path exercises
#: the same text-embedding code as real Last.fm tags.
_GENRE_TAGS = [
    ["indie rock", "alternative", "guitar"],
    ["electronic", "ambient", "downtempo"],
    ["hip hop", "rap", "beats"],
    ["folk", "singer-songwriter", "acoustic"],
    ["jazz", "improvisation", "piano"],
    ["metal", "hardcore", "heavy"],
    ["pop", "synthpop", "dance"],
    ["classical", "orchestral", "strings"],
]


@dataclass
class SyntheticConfig:
    n_users: int = 50
    n_artists: int = 300
    tracks_per_artist: int = 8
    latent_dim: int = 16
    n_genres: int = len(_GENRE_TAGS)

    #: Roughly how many months of history per user.
    n_months: int = 24
    sessions_per_month: float = 22.0
    mean_session_len: float = 9.0

    #: How many items a user is plausibly reaching for in a given month.
    library_size: int = 260
    #: Fraction of the library drawn at random from outside the top affinity
    #: band, so discovery happens and the novel slice is never empty.
    library_explore_frac: float = 0.25
    #: Fraction of the catalog released partway through the timeline rather than
    #: being available from the start. Drives the cold-item slice.
    late_release_frac: float = 0.35
    #: Per-month random-walk step on the taste vector. Drives Phase 6 drift.
    drift_scale: float = 0.18

    #: Score weights: taste affinity, global popularity, recency repeat bonus.
    w_taste: float = 3.0
    w_pop: float = 0.8
    w_repeat: float = 1.8
    #: How many recent plays carry a repeat bonus, and its decay.
    repeat_window: int = 40
    repeat_halflife: float = 12.0

    start_ts: int = 1_600_000_000  # 2020-09-13 UTC
    seed: int = 0


def _make_names(rng: np.random.Generator, n_artists: int, tracks_per_artist: int):
    """Readable fake names so text embeddings have real strings to encode."""
    artists, seen = [], set()
    while len(artists) < n_artists:
        name = f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}".title()
        if name not in seen:
            seen.add(name)
            artists.append(name)

    tracks = []
    for _ in range(n_artists):
        per, used = [], set()
        while len(per) < tracks_per_artist:
            title = f"{rng.choice(_VERB)} {rng.choice(_NOUN)}".title()
            if title not in used:
                used.add(title)
                per.append(title)
        tracks.append(per)
    return artists, tracks


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9, None)


def generate(config: SyntheticConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic events plus the item catalog backing them.

    Returns ``(events, catalog)``. ``events`` is in canonical schema form.
    ``catalog`` carries artist, track, and genre tags per item, standing in for
    what ``features/tags.py`` will fetch from Last.fm for real data.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    n_items = cfg.n_artists * cfg.tracks_per_artist
    artist_names, track_names = _make_names(rng, cfg.n_artists, cfg.tracks_per_artist)

    # Genre centroids -> artist centroids -> item latents. The nesting is what
    # makes "same artist" and "same genre" both meaningful in latent space.
    genre_centroids = _unit(rng.normal(size=(cfg.n_genres, cfg.latent_dim)))
    artist_genre = rng.integers(0, cfg.n_genres, size=cfg.n_artists)
    artist_latents = _unit(
        genre_centroids[artist_genre] + 0.55 * rng.normal(size=(cfg.n_artists, cfg.latent_dim))
    )

    item_artist = np.repeat(np.arange(cfg.n_artists), cfg.tracks_per_artist)
    item_latents = _unit(
        artist_latents[item_artist] + 0.30 * rng.normal(size=(n_items, cfg.latent_dim))
    )

    # Zipf popularity over a shuffled ranking, so popularity is uncorrelated
    # with item id and a model cannot cheat off id ordering.
    order = rng.permutation(n_items)
    log_pop = np.empty(n_items)
    log_pop[order] = -np.log(np.arange(1, n_items + 1))
    log_pop = (log_pop - log_pop.mean()) / log_pop.std()

    duration_ms = rng.integers(120_000, 320_000, size=n_items)

    # Catalogs grow. A fraction of items is released partway through the
    # timeline and is unplayable before then, which is what puts genuinely new
    # items in the test period. Without this the catalog is fixed for all time,
    # every test target was necessarily available during training, and the
    # cold-item evaluation slice is structurally impossible to populate no
    # matter how the split is drawn.
    horizon_s = cfg.n_months * 30 * 86_400
    release_ts = np.full(n_items, cfg.start_ts, dtype=np.int64)
    is_late = rng.random(n_items) < cfg.late_release_frac
    release_ts[is_late] = cfg.start_ts + rng.integers(0, horizon_s, size=int(is_late.sum()))

    catalog = pd.DataFrame(
        {
            "item_index": np.arange(n_items),
            "artist": [artist_names[a] for a in item_artist],
            "track": [track_names[a][i % cfg.tracks_per_artist] for i, a in enumerate(item_artist)],
            "genre": [_GENRE_TAGS[g][0] for g in artist_genre[item_artist]],
            "tags": [", ".join(_GENRE_TAGS[g]) for g in artist_genre[item_artist]],
            "duration_ms": duration_ms,
            "release_ts": release_ts,
        }
    )

    rows = []
    month_s = 30 * 86_400
    decay = np.log(2.0) / cfg.repeat_halflife

    for u in range(cfg.n_users):
        user_id = f"synth_{u:04d}"
        # Taste starts as a mixture of two genres, so users are not uniformly
        # spread and some share overlapping libraries.
        g1, g2 = rng.choice(cfg.n_genres, size=2, replace=False)
        taste = _unit(
            genre_centroids[g1] + 0.6 * genre_centroids[g2] + 0.3 * rng.normal(size=cfg.latent_dim)
        )

        ts = cfg.start_ts + int(rng.integers(0, month_s))
        recent: list[int] = []

        for _ in range(cfg.n_months):
            taste = _unit(taste + cfg.drift_scale * rng.normal(size=cfg.latent_dim))

            # A user reaches into a bounded library in any given month. This is
            # what produces realistic repeat rates: without it, sampling over
            # the full catalog makes every play effectively novel.
            affinity = item_latents @ taste
            # Most of the library is what the user currently likes best, but a
            # slice is drawn from outside it. Without that exploration slice the
            # library is a deterministic function of taste, users never
            # encounter anything new, and the novel evaluation slice comes out
            # empty.
            avail = np.nonzero(release_ts <= ts)[0]
            n_top = min(
                max(1, int(cfg.library_size * (1.0 - cfg.library_explore_frac))), len(avail)
            )
            local = np.argpartition(-(affinity[avail] + 0.35 * log_pop[avail]), n_top - 1)[:n_top]
            ranked = avail[local]

            rest = np.setdiff1d(avail, ranked)
            n_explore = min(cfg.library_size - n_top, len(rest))
            if n_explore > 0:
                library = np.concatenate([ranked, rng.choice(rest, size=n_explore, replace=False)])
            else:
                library = ranked

            base = cfg.w_taste * affinity[library] + cfg.w_pop * log_pop[library]
            lib_pos = {int(it): j for j, it in enumerate(library)}

            n_sessions = rng.poisson(cfg.sessions_per_month)
            for _ in range(max(1, n_sessions)):
                # Between-session gap: mostly hours, with a heavy tail so long
                # dormancies exist for the time-delta encoding to represent.
                gap_h = rng.lognormal(mean=2.6, sigma=1.1)
                ts += int(gap_h * 3600)

                session_len = 1 + rng.poisson(cfg.mean_session_len)
                for _ in range(session_len):
                    # Recency bonus over the last N plays, decayed by age, and
                    # capped at one bonus per item.
                    #
                    # Summing a bonus per occurrence instead is a rich-get-richer
                    # loop: an item played ten times in the window collects ten
                    # bonuses, feeding a softmax that then makes an eleventh play
                    # near-certain. That collapses each user's effective catalog
                    # to a handful of tracks and drives the repeat rate to ~100%,
                    # which makes the repeat baseline unbeatable and leaves the
                    # novel slice empty. Iterating newest-first and taking only
                    # the first hit per item is exactly a max, since age only
                    # grows.
                    bonus = np.zeros(len(library))
                    for age, item in enumerate(reversed(recent[-cfg.repeat_window :])):
                        j = lib_pos.get(item)
                        if j is not None and bonus[j] == 0.0:
                            bonus[j] = cfg.w_repeat * np.exp(-decay * age)

                    scores = base + bonus
                    scores -= scores.max()
                    p = np.exp(scores)
                    p /= p.sum()
                    item = int(library[rng.choice(len(library), p=p)])

                    # Skip probability falls with affinity: unloved tracks get
                    # cut short, which is exactly what the >30s filter removes.
                    aff = float(affinity[item])
                    p_skip = float(np.clip(0.34 - 0.30 * aff, 0.03, 0.6))
                    skipped = bool(rng.random() < p_skip)
                    if skipped:
                        ms = int(rng.integers(1_000, 30_000))
                    else:
                        ms = int(duration_ms[item] * rng.uniform(0.85, 1.0))

                    rows.append(
                        (
                            user_id,
                            ts,
                            catalog.at[item, "artist"],
                            catalog.at[item, "track"],
                            ms,
                            skipped,
                            bool(rng.random() < 0.35),
                            bool(rng.random() < 0.05),
                        )
                    )

                    recent.append(item)
                    if len(recent) > 4 * cfg.repeat_window:
                        del recent[: cfg.repeat_window]

                    # Advance by what was actually listened to, plus a beat.
                    ts += ms // 1000 + int(rng.integers(1, 8))

    events = pd.DataFrame(
        rows,
        columns=[
            schema.USER,
            schema.TS,
            schema.ARTIST,
            schema.TRACK,
            schema.MS_PLAYED,
            schema.SKIPPED,
            schema.SHUFFLE,
            schema.OFFLINE,
        ],
    )
    return schema.conform(events, source="synthetic"), catalog
