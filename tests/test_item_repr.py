"""The three Phase 2 item representations, and the properties that separate them.

The ablation is only meaningful if the variants genuinely differ in the way the
plan claims: ID embeddings carry no information about an item they were never
trained on, text embeddings do. These tests pin that difference down rather
than assuming it.
"""

from __future__ import annotations

import pytest
import torch

from melochron.models.heads import TiedItemScorer
from melochron.models.item_repr import (
    IdEmbedding,
    ProjectedTextEmbedding,
    build_item_representation,
)
from melochron.models.sasrec import SASRec

N_ITEMS = 24
D_TEXT = 16
D_MODEL = 8


def _text() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    vectors = torch.randn(N_ITEMS, D_TEXT, generator=g)
    vectors[0].zero_()
    vectors[1].zero_()
    return vectors


def test_id_embedding_shape_and_pad_row() -> None:
    rep = IdEmbedding(N_ITEMS, D_MODEL)
    assert rep.item_vectors().shape == (N_ITEMS, D_MODEL)
    assert torch.count_nonzero(rep.item_vectors()[0]) == 0


def test_projected_text_shape_and_reserved_rows() -> None:
    rep = ProjectedTextEmbedding(_text(), D_MODEL)
    vectors = rep.item_vectors()

    assert vectors.shape == (N_ITEMS, D_MODEL)
    # The projection is bias-free, so zero rows in must stay zero rows out.
    assert torch.count_nonzero(vectors[0]) == 0, "PAD picked up a non-zero representation"
    assert torch.count_nonzero(vectors[1]) == 0, "OOV picked up a non-zero representation"


def test_frozen_text_matrix_takes_no_gradient() -> None:
    rep = ProjectedTextEmbedding(_text(), D_MODEL, freeze=True)
    rep.item_vectors().sum().backward()

    assert not isinstance(rep.text_vectors, torch.nn.Parameter)
    assert rep.projection.weight.grad is not None, "the projection must still train"


def test_finetuned_text_matrix_takes_a_gradient() -> None:
    rep = ProjectedTextEmbedding(_text(), D_MODEL, freeze=False)
    rep.item_vectors().sum().backward()

    assert isinstance(rep.text_vectors, torch.nn.Parameter)
    assert rep.text_vectors.grad is not None
    assert torch.count_nonzero(rep.text_vectors.grad) > 0


def test_frozen_text_vectors_ride_along_in_the_checkpoint() -> None:
    """Serving needs the same vectors the model was trained against."""
    rep = ProjectedTextEmbedding(_text(), D_MODEL, freeze=True)
    assert "text_vectors" in rep.state_dict()


def test_text_representation_is_determined_by_the_text() -> None:
    """The cold-start property: identical text implies an identical vector.

    This is what makes an item unseen in training rankable at all, and it is
    exactly what an ID table cannot do.
    """
    vectors = _text()
    vectors[7] = vectors[9]  # two items, same text
    rep = ProjectedTextEmbedding(vectors, D_MODEL)
    out = rep.item_vectors()

    torch.testing.assert_close(out[7], out[9])
    assert not torch.allclose(out[7], out[8]), "different text collapsed to the same vector"


def test_id_representation_is_not_determined_by_anything_transferable() -> None:
    """The control. Two ID rows are independent draws; nothing ties them."""
    rep = IdEmbedding(N_ITEMS, D_MODEL)
    out = rep.item_vectors()
    assert not torch.allclose(out[7], out[9])


def test_build_dispatches_all_three_variants() -> None:
    text = _text()
    id_rep = build_item_representation("id", N_ITEMS, D_MODEL)
    frozen = build_item_representation("text_frozen", N_ITEMS, D_MODEL, text_vectors=text)
    tuned = build_item_representation("text_finetuned", N_ITEMS, D_MODEL, text_vectors=text)

    assert isinstance(id_rep, IdEmbedding)
    assert frozen.frozen is True
    assert tuned.frozen is False


def test_build_rejects_an_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown item representation"):
        build_item_representation("word2vec", N_ITEMS, D_MODEL)


def test_build_rejects_text_vectors_misaligned_with_the_vocabulary() -> None:
    """Misalignment here would silently score every item as some other item."""
    with pytest.raises(ValueError, match="aligned by item id"):
        build_item_representation(
            "text_frozen", N_ITEMS, D_MODEL, text_vectors=torch.randn(N_ITEMS - 3, D_TEXT)
        )


def test_build_requires_text_vectors_for_text_variants() -> None:
    with pytest.raises(ValueError, match="needs text_vectors"):
        build_item_representation("text_frozen", N_ITEMS, D_MODEL)


@pytest.mark.parametrize("variant", ["id", "text_frozen", "text_finetuned"])
def test_encoder_runs_with_every_variant(variant: str) -> None:
    """One encoder, three representations, identical interface."""
    rep = build_item_representation(variant, N_ITEMS, D_MODEL, text_vectors=_text())
    model = SASRec(
        n_items=N_ITEMS, d_model=D_MODEL, n_heads=2, n_blocks=1, max_len=8, item_repr=rep
    ).eval()

    seq = torch.randint(2, N_ITEMS, (3, 8))
    with torch.no_grad():
        out = model(seq)
        logits = TiedItemScorer(model.items).full_logits(out[:, -1])

    assert out.shape == (3, 8, D_MODEL)
    assert logits.shape == (3, N_ITEMS)
    assert torch.isfinite(logits[:, 2:]).all()


def test_encoder_rejects_a_mismatched_representation() -> None:
    rep = IdEmbedding(N_ITEMS, D_MODEL)
    with pytest.raises(ValueError, match="but the model expects"):
        SASRec(n_items=N_ITEMS, d_model=D_MODEL * 2, item_repr=rep)


def test_masked_logits_stay_differentiable() -> None:
    """Reserved-slot masking must not break the fine-tuning ablation's backward."""
    rep = ProjectedTextEmbedding(_text(), D_MODEL, freeze=False)
    head = TiedItemScorer(rep)

    logits = head.full_logits(torch.randn(4, D_MODEL), mask_reserved=True)
    logits[:, 2:].sum().backward()

    assert rep.projection.weight.grad is not None
    assert torch.isfinite(rep.projection.weight.grad).all(), "masking leaked NaN into the gradient"
