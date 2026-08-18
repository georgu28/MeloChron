"""The item-vector cache and the order the projection is applied in.

Both of these are performance changes, and a performance change to a
representation is only safe if it is provably the same arithmetic. So these
tests are about equality first --- the fast path must agree with the slow one
it replaced --- and about invalidation second, because the one way a cache like
this turns into a silent wrong answer is by outliving the weights behind it.

The numbers that motivated them, measured through the serving endpoint on the
hybrid artifact: 538 ms p50 per recommendation, 271 ms with the table frozen at
load, 12 ms once the projection stopped being applied to the whole catalog to
serve a 200-play history.
"""

from __future__ import annotations

import torch

from melochron.models.item_repr import (
    HybridItemRepresentation,
    IdEmbedding,
    ProjectedTextEmbedding,
)

N_ITEMS = 32
D_TEXT = 16
D_MODEL = 8


def _text() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    vectors = torch.randn(N_ITEMS, D_TEXT, generator=g)
    vectors[0].zero_()
    vectors[1].zero_()
    return vectors


def _trained_hybrid() -> HybridItemRepresentation:
    """A hybrid whose residual is non-zero, so the two paths can disagree."""
    rep = HybridItemRepresentation(_text(), D_MODEL)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        rep.residual.weight.copy_(torch.randn(N_ITEMS, D_MODEL, generator=g) * 0.1)
        rep.residual.weight[rep.pad_id].zero_()
    return rep


def test_text_forward_matches_full_table() -> None:
    """Gathering before the projection is the same as gathering after it."""
    rep = ProjectedTextEmbedding(_text(), D_MODEL)
    ids = torch.tensor([[0, 1, 5, N_ITEMS - 1], [2, 2, 9, 17]])
    torch.testing.assert_close(rep(ids), rep.compute_item_vectors()[ids])


def test_hybrid_forward_matches_full_table() -> None:
    rep = _trained_hybrid()
    ids = torch.tensor([[0, 1, 5, N_ITEMS - 1], [2, 2, 9, 17]])
    torch.testing.assert_close(rep(ids), rep.compute_item_vectors()[ids])


def test_reserved_rows_stay_zero_through_the_fast_path() -> None:
    """PAD and OOV must not become scorable by taking a different route."""
    rep = ProjectedTextEmbedding(_text(), D_MODEL)
    out = rep(torch.tensor([0, 1]))
    assert torch.count_nonzero(out) == 0


def test_frozen_table_equals_computed_table() -> None:
    rep = _trained_hybrid().eval()
    computed = rep.compute_item_vectors().detach().clone()
    torch.testing.assert_close(rep.freeze_item_vectors(), computed)
    torch.testing.assert_close(rep.item_vectors(), computed)


def test_frozen_table_is_detached() -> None:
    """A cache must never be a path gradient flows along."""
    rep = _trained_hybrid().eval()
    assert rep.freeze_item_vectors().requires_grad is False
    assert rep.item_vectors().requires_grad is False


def test_training_mode_drops_the_frozen_table() -> None:
    """The invalidation that makes fine-tuning from an artifact safe."""
    rep = _trained_hybrid().eval()
    rep.freeze_item_vectors()
    assert rep._frozen_vectors is not None
    rep.train()
    assert rep._frozen_vectors is None


def test_weights_moving_after_a_thaw_are_visible_again() -> None:
    """freeze -> train -> mutate -> eval must not serve the stale table."""
    rep = _trained_hybrid().eval()
    stale = rep.freeze_item_vectors().clone()

    rep.train()
    with torch.no_grad():
        rep.residual.weight[5] += 1.0
    rep.eval()

    fresh = rep.item_vectors()
    assert not torch.allclose(fresh[5], stale[5])
    torch.testing.assert_close(fresh, rep.compute_item_vectors())


def test_device_or_dtype_change_drops_the_frozen_table() -> None:
    """_apply covers .to()/.float(); a cache built before it would be stale."""
    rep = _trained_hybrid().eval()
    rep.freeze_item_vectors()
    rep.float()
    assert rep._frozen_vectors is None


def test_id_embedding_does_not_copy_its_table() -> None:
    """Its table is already a parameter, so a cache would be a second copy."""
    rep = IdEmbedding(N_ITEMS, D_MODEL).eval()
    frozen = rep.freeze_item_vectors()
    assert frozen is rep.embedding.weight
    assert rep._frozen_vectors is None


def test_nested_text_module_is_reused_by_the_hybrid_table() -> None:
    """Hybrid composes the public accessor, so a frozen inner table is honoured.

    Same instance before and after: two separately constructed hybrids have
    independently initialized projections and would differ for that reason.
    """
    rep = _trained_hybrid().eval()
    before = rep.compute_item_vectors().detach().clone()
    rep.text.freeze_item_vectors()
    torch.testing.assert_close(rep.compute_item_vectors(), before)
