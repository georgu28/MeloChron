import pandas as pd
import pytest

from melochron import schema
from melochron.data import vocab as V


@pytest.mark.parametrize(
    "raw",
    [
        "Song",
        "Song (feat. Someone)",
        "Song (Remastered 2011)",
        "Song - 2011 Remaster",
        "Song (feat. A) - 2011 Remaster",
        "Song [ft. B] (Deluxe Edition)",
        "Song - Radio Edit",
        "  Song   With   Spaces  ",
        "Sóng Ñame",
        "",
    ],
)
def test_normalize_is_idempotent(raw):
    """Canonicalization must be a fixed point after one application.

    Without this, an item's key depends on how many times the pipeline touched
    it, and the same track lands in two vocabulary slots.
    """
    once = V.normalize_field(raw)
    assert V.normalize_field(once) == once


def test_canonical_key_is_idempotent_end_to_end():
    key = V.canonical_key("The Band (feat. X)", "Song - 2011 Remaster")
    artist, track = key.split(V.SEP)
    assert V.canonical_key(artist, track) == key


def test_packaging_variants_collapse():
    """The same recording, packaged differently, is one item."""
    base = V.canonical_key("Radiohead", "Creep")
    assert V.canonical_key("Radiohead", "Creep - Remastered") == base
    assert V.canonical_key("Radiohead", "Creep (2011 Remaster)") == base
    assert V.canonical_key("radiohead", "  Creep  ") == base
    assert V.canonical_key("Radiohead", "Creep (feat. Nobody)") == base


def test_different_recordings_stay_distinct():
    """Under-merging is recoverable; over-merging silently corrupts labels."""
    base = V.canonical_key("Nirvana", "Come As You Are")
    assert V.canonical_key("Nirvana", "Come As You Are - Live") != base
    assert V.canonical_key("Nirvana", "Come As You Are - Acoustic") != base
    assert V.canonical_key("Weezer", "Come As You Are") != base


def _events(pairs):
    df = pd.DataFrame(
        {
            schema.USER: ["u"] * len(pairs),
            schema.TS: list(range(1_600_000_000, 1_600_000_000 + len(pairs))),
            schema.ARTIST: [a for a, _ in pairs],
            schema.TRACK: [t for _, t in pairs],
        }
    )
    return schema.conform(df, source="test")


def test_build_vocab_respects_min_count_and_reserves_ids():
    df = _events([("A", "x")] * 5 + [("B", "y")] * 2)
    vocab = V.build_vocab(df, min_count=5)

    assert vocab.id_to_key[V.PAD_ID] == "<pad>"
    assert vocab.id_to_key[V.OOV_ID] == "<oov>"
    assert vocab.n_items == 1

    ids = vocab.encode([V.canonical_key("A", "x"), V.canonical_key("B", "y")])
    assert ids[0] >= V.FIRST_ITEM_ID  # kept
    assert ids[1] == V.OOV_ID  # below threshold, not dropped


def test_coverage_reports_out_of_vocabulary_fraction():
    df = _events([("A", "x")] * 5)
    vocab = V.build_vocab(df, min_count=5)
    keys = [V.canonical_key("A", "x"), V.canonical_key("Z", "unheard")]
    assert vocab.coverage(keys) == 0.5
