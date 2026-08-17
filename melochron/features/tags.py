"""Last.fm tag client with a committed, resumable disk cache.

Genre tags are the enrichment that makes text item-vectors carry musical
meaning rather than orthographic similarity between titles. They come from
Last.fm because that source covers **both** corpora: Spotify's artist genres
would cover the personal export but not the pretraining corpus, which would
split the item representation into two incompatible spaces and break the
transfer story the project is built on.

Three properties this needs and the reasons they are not optional:

* **Resumable.** A vocabulary of ~170k items at the ~4 req/s courtesy limit is
  roughly twelve hours. Any fetch will be interrupted.
* **Committed cache.** Those twelve hours must not be repeated by a fresh
  clone. ``.gitignore`` un-ignores ``data/tags/`` specifically for this.
* **Degrades without a key.** No ``LASTFM_API_KEY`` means names-only item
  strings and a warning, never a crash. Tags improve the text variant; they are
  not required for it to run.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from melochron.data import vocab as vocab_mod

API_ROOT = "https://ws.audioscrobbler.com/2.0/"

#: Per-item cache, keyed by canonical item key.
DEFAULT_CACHE = Path("data/tags/lastfm_tags.json.gz")
#: Per-artist cache, keyed by normalized artist name. Separate file because the
#: two are populated by different fetch strategies with very different costs.
DEFAULT_ARTIST_CACHE = Path("data/tags/lastfm_artist_tags.json.gz")

#: Last.fm's published courtesy limit is ~5 requests/second averaged over 5
#: minutes. Four leaves headroom rather than riding the boundary.
DEFAULT_RATE = 4.0

#: Tags applied by very few users describe the listener, not the music
#: ("seen live", "albums i own"). Last.fm normalizes count to 0-100.
MIN_TAG_COUNT = 10


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=value`` lines from a dotenv file into ``os.environ``.

    Hand-rolled rather than pulling in ``python-dotenv``, which lives in the
    ``serve`` extra: the tag fetch is a data-preparation step and should not
    require the serving dependencies to be installed. Existing environment
    variables win, so an explicitly exported key is never silently overridden
    by a stale file.
    """
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


class TagCache:
    """Item key -> list of tag names, persisted as gzipped JSON.

    Misses are recorded explicitly as an empty list. Without that, an item with
    genuinely no tags is indistinguishable from one never looked up, and every
    rerun refetches the entire long tail of untagged obscurities.
    """

    def __init__(self, path: str | Path = DEFAULT_CACHE):
        self.path = Path(path)
        self.data: dict[str, list[str]] = {}
        if self.path.exists():
            with gzip.open(self.path, "rt", encoding="utf-8") as fh:
                self.data = json.load(fh)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str) -> list[str]:
        return self.data.get(key, [])

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False)
        tmp.replace(self.path)  # atomic: a killed process cannot truncate the cache
        return self.path

    @property
    def tagged_fraction(self) -> float:
        if not self.data:
            return 0.0
        return sum(1 for v in self.data.values() if v) / len(self.data)


class LastfmClient:
    def __init__(
        self,
        api_key: str | None = None,
        rate: float = DEFAULT_RATE,
        timeout: float = 15.0,
        min_count: int = MIN_TAG_COUNT,
        max_tags: int = 10,
    ):
        if api_key is None and "LASTFM_API_KEY" not in os.environ:
            load_dotenv()
        self.api_key = api_key or os.environ.get("LASTFM_API_KEY")
        self.min_interval = 1.0 / rate if rate > 0 else 0.0
        self.timeout = timeout
        self.min_count = min_count
        self.max_tags = max_tags
        self._last_call = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call(self, method: str, **params) -> dict | None:
        if not self.enabled:
            return None
        self._throttle()

        query = urllib.parse.urlencode(
            {"method": method, "api_key": self.api_key, "format": "json", **params}
        )
        request = urllib.request.Request(
            f"{API_ROOT}?{query}", headers={"User-Agent": "melochron/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Backing off rather than dropping the item: a rate-limit is a
                # "try later", and treating it as "no tags" would poison the
                # cache with a permanent empty for a taggable track.
                time.sleep(5.0)
                return self._call(method, **params)
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extract(payload: dict | None, container: str) -> list[tuple[str, int]]:
        if not payload:
            return []
        tags = (payload.get(container) or {}).get("tag") or []
        if isinstance(tags, dict):  # single-tag responses are not wrapped in a list
            tags = [tags]
        out = []
        for tag in tags:
            name = (tag.get("name") or "").strip()
            if name:
                try:
                    count = int(tag.get("count") or 0)
                except (TypeError, ValueError):
                    count = 0
                out.append((name.lower(), count))
        return out

    def _keep(self, raw: list[tuple[str, int]]) -> list[str]:
        return [name for name, count in raw if count >= self.min_count][: self.max_tags]

    def artist_tags(self, artist: str) -> list[str]:
        """Tags for one artist. One request covers every track they appear on."""
        return self._keep(self._extract(self._call("artist.gettoptags", artist=artist), "toptags"))

    def top_tags(self, artist: str, track: str) -> list[str]:
        """Track tags, falling back to artist tags when the track has none.

        The fallback matters more than it sounds: most of a long-tail catalog
        has no track-level tags at all, and artist-level genre is still a far
        better signal than nothing.
        """
        raw = self._extract(self._call("track.gettoptags", artist=artist, track=track), "toptags")
        if not raw:
            raw = self._extract(self._call("artist.gettoptags", artist=artist), "toptags")
        return self._keep(raw)


def fetch_for_items(
    items: list[tuple[str, str, str]],
    cache: TagCache | None = None,
    client: LastfmClient | None = None,
    save_every: int = 500,
    limit: int | None = None,
    verbose: bool = True,
) -> TagCache:
    """Populate ``cache`` for ``items`` given as ``(item_key, artist, track)``.

    Already-cached keys are skipped, so this is safe to interrupt and rerun.
    """
    cache = cache or TagCache()
    client = client or LastfmClient()

    if not client.enabled:
        if verbose:
            print(
                "LASTFM_API_KEY not set: skipping tag fetch. Item strings will use "
                "names only, which is a supported configuration, not a failure."
            )
        return cache

    todo = [row for row in items if row[0] not in cache]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"{len(cache):,} cached, {len(todo):,} to fetch")

    started = time.time()
    for i, (key, artist, track) in enumerate(todo, 1):
        cache.data[key] = client.top_tags(artist, track)
        if i % save_every == 0:
            cache.save()
            if verbose:
                rate = i / max(time.time() - started, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6)
                print(f"  {i:,}/{len(todo):,}  {rate:.1f}/s  eta {eta / 60:.0f}m")

    cache.save()
    return cache


def artist_cache_key(artist: str) -> str:
    """Normalized artist name, so casing and spacing variants share one entry."""
    return vocab_mod.normalize_field(artist)


def fetch_artist_tags(
    artists: list[str],
    cache: TagCache | None = None,
    client: LastfmClient | None = None,
    save_every: int = 250,
    limit: int | None = None,
    verbose: bool = True,
) -> TagCache:
    """Populate an artist-keyed cache. One request covers all their tracks.

    This is the strategy to prefer on a large catalog. On the lastfm-1K
    vocabulary at ``min_count=20`` there are 171,902 items but only 20,208
    distinct artists, so artist-level fetching is ~8.5x fewer requests: about
    1.4 hours at the courtesy rate against about 11.9, for the same 100% item
    coverage.

    The cost is resolution. Every track by an artist receives identical tags, so
    tags stop discriminating *within* an artist and only separate artists from
    each other. For genre signal that is most of what matters, but it is a real
    limitation and belongs next to any ablation number derived from it.
    """
    cache = cache or TagCache(DEFAULT_ARTIST_CACHE)
    client = client or LastfmClient()

    if not client.enabled:
        if verbose:
            print(
                "LASTFM_API_KEY not set: skipping tag fetch. Item strings will use "
                "names only, which is a supported configuration, not a failure."
            )
        return cache

    seen: set[str] = set()
    todo: list[tuple[str, str]] = []
    for artist in artists:
        key = artist_cache_key(artist)
        if not key or key in seen or key in cache:
            continue
        seen.add(key)
        todo.append((key, artist))

    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"{len(cache):,} artists cached, {len(todo):,} to fetch")

    started = time.time()
    for i, (key, artist) in enumerate(todo, 1):
        cache.data[key] = client.artist_tags(artist)
        if i % save_every == 0:
            cache.save()
            if verbose:
                rate = i / max(time.time() - started, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6)
                print(
                    f"  {i:,}/{len(todo):,}  {rate:.1f}/s  eta {eta / 60:.0f}m  "
                    f"tagged {cache.tagged_fraction:.0%}"
                )

    cache.save()
    return cache


def tags_for_vocab(
    vocab: vocab_mod.Vocab,
    artist_cache: TagCache | None = None,
    track_cache: TagCache | None = None,
) -> dict[str, list[str]]:
    """Build ``item_key -> tags``, preferring track tags over artist tags.

    Either cache may be absent. Track-level entries win where they exist, so a
    cheap artist-level pass can later be refined for the most-played tracks
    without refetching or rebuilding anything.
    """
    out: dict[str, list[str]] = {}
    for item_id in range(vocab_mod.FIRST_ITEM_ID, len(vocab)):
        key = vocab.id_to_key[item_id]

        if track_cache is not None:
            tags = track_cache.get(key)
            if tags:
                out[key] = tags
                continue

        if artist_cache is not None and vocab.display and item_id < len(vocab.display):
            tags = artist_cache.get(artist_cache_key(vocab.display[item_id][0]))
            if tags:
                out[key] = tags

    return out
