"""Composing the text string that represents an item.

This is the whole transfer mechanism in one function. A track the model never
trained on has no usable id, but it does have a name, an artist, and possibly
tags. Turning that into a sentence and embedding it is what lets the model
score an item it has never seen, which is the cold-start case every new
uploader lands in.

The template is a parameter because it is cheap to ablate and it matters more
than it looks. ``"Song by Artist"`` and ``"Artist - Song"`` produce different
neighbourhoods in sentence-embedding space, and tags change the geometry
substantially: without them the space is dominated by orthographic similarity
between names, which is close to noise for recommendation.
"""

from __future__ import annotations

from melochron.data.vocab import FIRST_ITEM_ID, SEP, Vocab

DEFAULT_TEMPLATE = "{track} by {artist}"
TAGGED_TEMPLATE = "{track} by {artist}. Genre: {tags}"

#: How many tags to include. Last.fm returns a long tail of tags applied by a
#: handful of users each ("seen live", "favourite"); past the first few they
#: describe the listener rather than the music.
DEFAULT_MAX_TAGS = 5


def compose(
    artist: str,
    track: str,
    tags: list[str] | None = None,
    template: str = DEFAULT_TEMPLATE,
    tagged_template: str = TAGGED_TEMPLATE,
    max_tags: int = DEFAULT_MAX_TAGS,
) -> str:
    """Build the sentence for one item."""
    artist = (artist or "").strip()
    track = (track or "").strip()

    if tags:
        joined = ", ".join(t.strip() for t in tags[:max_tags] if t and t.strip())
        if joined:
            return tagged_template.format(track=track, artist=artist, tags=joined)
    return template.format(track=track, artist=artist)


def strings_for_vocab(
    vocab: Vocab,
    tags: dict[str, list[str]] | None = None,
    template: str = DEFAULT_TEMPLATE,
    tagged_template: str = TAGGED_TEMPLATE,
    max_tags: int = DEFAULT_MAX_TAGS,
) -> list[str]:
    """One string per vocabulary id, aligned to ``vocab.id_to_key``.

    Reserved rows (PAD, OOV) get the empty string. Their embeddings are zeroed
    downstream rather than being whatever the encoder makes of ``""``, because
    a reserved slot must never be scorable.

    Display names come from ``vocab.display`` when available, which preserves
    original casing and punctuation. Falling back to the canonical key would
    feed the encoder a casefolded, suffix-stripped string, losing exactly the
    surface detail a language model is good at using.
    """
    tags = tags or {}
    out: list[str] = [""] * len(vocab)

    for item_id in range(FIRST_ITEM_ID, len(vocab)):
        key = vocab.id_to_key[item_id]

        if vocab.display and item_id < len(vocab.display):
            artist, track = vocab.display[item_id]
        else:
            artist, _, track = key.partition(SEP)

        out[item_id] = compose(
            artist=artist,
            track=track,
            tags=tags.get(key),
            template=template,
            tagged_template=tagged_template,
            max_tags=max_tags,
        )

    return out


def tag_coverage(vocab: Vocab, tags: dict[str, list[str]]) -> float:
    """Fraction of real items that have at least one tag.

    Worth reporting next to the ablation: a text variant built on 20% tag
    coverage is mostly a names-only variant wearing a different label.
    """
    total = len(vocab) - FIRST_ITEM_ID
    if total <= 0:
        return 0.0
    hits = sum(1 for i in range(FIRST_ITEM_ID, len(vocab)) if tags.get(vocab.id_to_key[i]))
    return hits / total
