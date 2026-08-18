"""Invariants for re-pointing a pretrained encoder at a new catalog.

The transplant is the operation the transfer-learning claim rests on, and it
has one failure mode that would be invisible in every metric: quietly keeping
something that belonged to the old catalog. A stale ``[n_items]`` head bias or a
half-loaded projection does not crash and does not produce obviously wrong
numbers, it just makes the zero-shot row mean something other than what the
table says it means.

So these tests assert what *moved* and what did not, rather than only that the
call returned an object.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melochron.data.vocab import FIRST_ITEM_ID, Vocab
from melochron.models.scorer import build_scorer
from melochron.train import checkpoint, transfer

D_TEXT = 24
D_MODEL = 16
N_OLD = 40
N_NEW = 25


def _vocab(n_items: int, prefix: str) -> Vocab:
    keys = [f"{prefix} {i} :: track {i}" for i in range(n_items - FIRST_ITEM_ID)]
    id_to_key = ["<pad>", "<oov>"] + keys
    return Vocab(
        key_to_id={k: i + FIRST_ITEM_ID for i, k in enumerate(keys)},
        id_to_key=id_to_key,
        counts=np.zeros(len(id_to_key), dtype=np.int64),
        display=[("", ""), ("", "")] + [(f"{prefix} {i}", f"track {i}") for i in range(len(keys))],
    )


def _text(n_items: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_items, D_TEXT, generator=g)


def _write_checkpoint(tmp_path, variant: str):
    torch.manual_seed(0)
    text = _text(N_OLD) if variant != "id" else None
    model, head, _ = build_scorer(
        n_items=N_OLD,
        variant=variant,
        text_vectors=text,
        d_model=D_MODEL,
        n_heads=2,
        n_blocks=2,
        max_len=32,
    )
    config = {
        "variant": variant,
        "d_model": D_MODEL,
        "n_heads": 2,
        "n_blocks": 2,
        "max_len": 32,
        "dropout": 0.1,
        "use_time": True,
    }
    path = tmp_path / f"{variant}.pt"
    checkpoint.save(path, model, head, _vocab(N_OLD, "old"), config)
    return path, model


def test_id_checkpoint_cannot_be_transplanted(tmp_path) -> None:
    # Not a limitation to route around: an ID table is indexed by the old
    # vocabulary, so a track it never saw has no representation at all.
    path, _ = _write_checkpoint(tmp_path, "id")
    with pytest.raises(ValueError, match="cannot be re-pointed"):
        transfer.load_for_catalog(
            str(path), _text(N_NEW, seed=1), _vocab(N_NEW, "new"), name="zero-shot"
        )


def test_transplant_carries_the_projection_exactly(tmp_path) -> None:
    path, original = _write_checkpoint(tmp_path, "text_frozen")
    result = transfer.load_for_catalog(str(path), _text(N_NEW, seed=1), _vocab(N_NEW, "new"))

    torch.testing.assert_close(
        result.model.items.projection.weight,
        original.items.projection.weight,
        msg="the learned text->model projection did not survive the transplant",
    )
    torch.testing.assert_close(
        result.model.blocks[0].attn.q_proj.weight,
        original.blocks[0].attn.q_proj.weight,
        msg="encoder weights did not survive the transplant",
    )


def test_transplant_installs_the_new_catalog(tmp_path) -> None:
    path, _ = _write_checkpoint(tmp_path, "text_frozen")
    new_text = _text(N_NEW, seed=1)
    result = transfer.load_for_catalog(str(path), new_text, _vocab(N_NEW, "new"))

    assert result.model.n_items == N_NEW, "model still sized for the old catalog"
    torch.testing.assert_close(
        result.model.items.text_vectors[FIRST_ITEM_ID:],
        new_text[FIRST_ITEM_ID:],
        msg="the old text matrix overwrote the new catalog",
    )

    with torch.no_grad():
        ids = torch.randint(FIRST_ITEM_ID, N_NEW, (3, 8))
        logits = result.head.full_logits(result.model.encode_last(ids))
    assert logits.shape == (3, N_NEW), "scores are not the width of the new catalog"


def test_transplanted_head_carries_no_stale_bias(tmp_path) -> None:
    # A per-item bias learned from someone else's listening is not a prior
    # about these items, and it is the wrong shape besides.
    path, _ = _write_checkpoint(tmp_path, "text_frozen")
    result = transfer.load_for_catalog(str(path), _text(N_NEW, seed=1), _vocab(N_NEW, "new"))
    assert result.head.bias is None, "the old catalog's item bias came along"


def test_transplant_rejects_text_from_a_different_encoder(tmp_path) -> None:
    path, _ = _write_checkpoint(tmp_path, "text_frozen")
    wrong = torch.randn(N_NEW, D_TEXT + 8)
    with pytest.raises(ValueError, match="projection consumes"):
        transfer.load_for_catalog(str(path), wrong, _vocab(N_NEW, "new"))


def test_transplant_rejects_vectors_misaligned_with_the_vocabulary(tmp_path) -> None:
    path, _ = _write_checkpoint(tmp_path, "text_frozen")
    with pytest.raises(ValueError, match="row i must be vocabulary id i"):
        transfer.load_for_catalog(str(path), _text(N_NEW + 3, seed=1), _vocab(N_NEW, "new"))


def test_freeze_encoder_leaves_only_the_item_table_trainable(tmp_path) -> None:
    path, _ = _write_checkpoint(tmp_path, "text_finetuned")
    result = transfer.load_for_catalog(
        str(path), _text(N_NEW, seed=1), _vocab(N_NEW, "new"), freeze_encoder=True
    )

    assert not result.model.blocks[0].attn.q_proj.weight.requires_grad, "encoder is still trainable"
    assert result.model.items.text_vectors.requires_grad, "nothing is left to adapt"
