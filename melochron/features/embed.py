"""Sentence-embedding matrix over the item vocabulary.

Produces the ``[n_items, d_text]`` array that
``models.item_repr.ProjectedTextEmbedding`` consumes. Row *i* is vocabulary id
*i*, and that alignment is the entire contract: a matrix built against a
different vocabulary silently gives every item someone else's semantics, which
does not crash and does not look wrong in any metric except a mysteriously bad
one.

The cache key covers the model name, the template, and the vocabulary contents,
so any of those changing produces a different file rather than a stale hit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from melochron.data.vocab import FIRST_ITEM_ID, OOV_ID, PAD_ID, Vocab
from melochron.features import text as text_mod

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


def cache_key(vocab: Vocab, model_name: str, template: str, tags_fingerprint: str = "none") -> str:
    """Stable digest of everything that changes the resulting matrix."""
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(template.encode())
    digest.update(tags_fingerprint.encode())
    digest.update(str(len(vocab)).encode())
    # The vocabulary contents, not just its length: two vocabularies of equal
    # size with different items must not share a cache entry.
    for key in vocab.id_to_key:
        digest.update(key.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def embed_strings(
    strings: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    device: str | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode strings with a sentence-transformer. Imports lazily."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        strings,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_matrix(
    vocab: Vocab,
    tags: dict[str, list[str]] | None = None,
    model_name: str = DEFAULT_MODEL,
    template: str = text_mod.DEFAULT_TEMPLATE,
    cache_dir: str | Path | None = "data/embeddings",
    batch_size: int = 256,
    device: str | None = None,
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """Return ``([n_items, d_text] float32, metadata)`` aligned to vocab ids.

    PAD and OOV rows are zeroed. ``ProjectedTextEmbedding`` re-zeros them too,
    which is deliberate belt-and-braces: a nonzero reserved row would make
    "unknown track" a scorable recommendation with real semantics behind it.
    """
    tags = tags or {}
    fingerprint = (
        hashlib.sha256(json.dumps(sorted(tags.items()), ensure_ascii=False).encode()).hexdigest()[
            :16
        ]
        if tags
        else "none"
    )

    key = cache_key(vocab, model_name, template, fingerprint)
    path = Path(cache_dir) / f"text-{key}.npy" if cache_dir else None
    meta = {
        "model": model_name,
        "template": template,
        "n_items": len(vocab),
        "cache_key": key,
        "tag_coverage": round(text_mod.tag_coverage(vocab, tags), 4) if tags else 0.0,
    }

    if path and path.exists() and not force:
        vectors = np.load(path)
        if len(vectors) == len(vocab):
            meta["cached"] = True
            return vectors, meta
        # Length mismatch under a matching key should be impossible, but a
        # silently misaligned matrix is the worst failure available here.
        print(f"cache {path} has {len(vectors)} rows, expected {len(vocab)}; rebuilding")

    strings = text_mod.strings_for_vocab(vocab, tags=tags, template=template)
    vectors = embed_strings(strings, model_name=model_name, batch_size=batch_size, device=device)

    vectors[PAD_ID] = 0.0
    vectors[OOV_ID] = 0.0

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    meta["cached"] = False
    return vectors, meta


def coverage_report(vectors: np.ndarray) -> dict:
    """Sanity checks worth printing before a multi-hour training run."""
    real = vectors[FIRST_ITEM_ID:]
    norms = np.linalg.norm(real, axis=1)
    return {
        "rows": len(vectors),
        "dim": int(vectors.shape[1]),
        "reserved_rows_zero": bool(not vectors[PAD_ID].any() and not vectors[OOV_ID].any()),
        "zero_rows": int((norms == 0).sum()),
        "mean_norm": float(norms.mean()) if len(norms) else 0.0,
    }
