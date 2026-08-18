"""Tests for the hybrid item representation.

The load-bearing property is that an item which never receives gradient keeps a
residual of exactly zero, so it falls back to pure text. If that breaks, cold
items get random noise added to the only signal they have, and nothing
crashes: the cold-start numbers just quietly get worse.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melochron.data import sessions, synthetic, vocab
from melochron.models.heads import TiedItemScorer, sample_negatives
from melochron.models.item_repr import (
    HybridItemRepresentation,
    ProjectedTextEmbedding,
    build_item_representation,
)
from melochron.models.scorer import build_scorer
from melochron.train import checkpoint as ckpt


@pytest.fixture(scope="module")
def small():
    events, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=6, n_months=3, seed=13))
    positives = sessions.filter_positives(events)
    vc = vocab.build_vocab(positives, min_count=3)
    seqs = sessions.build_sequences(positives, vc)
    return vc, seqs


@pytest.fixture
def text_vectors(small):
    vc, _ = small
    torch.manual_seed(0)
    vectors = torch.randn(len(vc), 384)
    vectors[0].zero_()
    vectors[1].zero_()
    return vectors


def test_starts_as_pure_text_in_direction(text_vectors):
    """At init the hybrid must rank exactly as the text variant does.

    With normalization on it is not element-wise identical to the text variant,
    it is that variant projected onto the unit sphere. Direction is what decides
    ranking, so cosine similarity of 1.0 is the property worth asserting; exact
    equality would just be asserting that normalization is off.
    """
    torch.manual_seed(0)
    hybrid = HybridItemRepresentation(text_vectors, d_model=32)
    torch.manual_seed(0)
    text_only = ProjectedTextEmbedding(text_vectors, d_model=32)

    a = torch.nn.functional.normalize(hybrid.item_vectors()[2:], dim=-1)
    b = torch.nn.functional.normalize(text_only.item_vectors()[2:], dim=-1)
    torch.testing.assert_close((a * b).sum(-1), torch.ones(len(a)), atol=1e-5, rtol=1e-5)

    assert hybrid.residual.weight.abs().sum().item() == 0.0
    assert len(hybrid.pure_text_rows()) == hybrid.n_items


def test_unnormalized_starts_exactly_at_pure_text(text_vectors):
    """With normalization off the hybrid is element-wise the text variant."""
    torch.manual_seed(0)
    hybrid = HybridItemRepresentation(text_vectors, d_model=32, normalize=False)
    torch.manual_seed(0)
    text_only = ProjectedTextEmbedding(text_vectors, d_model=32)
    torch.testing.assert_close(hybrid.item_vectors(), text_only.item_vectors())


def test_normalization_equalizes_trained_and_cold_magnitudes(text_vectors):
    """The bug this exists to prevent, stated directly.

    An unnormalized residual grows during training while cold rows stay at
    text-only scale. Scoring is a dot product, so that magnitude gap alone
    buries cold items regardless of how good their direction is. The first
    hybrid run scored a flat 0.0000 on every cold slice for exactly this reason:
    trained items reached norm 1.52 against 0.56 for pure-text items.
    """
    hybrid = HybridItemRepresentation(text_vectors, d_model=32, normalize=True)
    with torch.no_grad():
        hybrid.residual.weight[5:20].normal_(std=3.0)  # simulate trained rows

    norms = hybrid.item_vectors().norm(dim=-1)
    trained, cold = norms[5:20], norms[20:]
    assert torch.allclose(trained.mean(), cold.mean(), atol=1e-4), (
        f"trained {trained.mean():.4f} vs cold {cold.mean():.4f}: magnitudes must match"
    )


def test_gradient_touches_only_the_rows_it_indexes(text_vectors):
    """The cold-start guarantee, stated as a test."""
    hybrid = HybridItemRepresentation(text_vectors, d_model=32)
    head = TiedItemScorer(hybrid, use_bias=False)
    optimizer = torch.optim.SGD(hybrid.parameters(), lr=1.0)

    touched = torch.tensor([5, 6, 7])
    hidden = torch.randn(len(touched), 32)

    loss = head.shared_negative_logits(hidden, touched, torch.tensor([8, 9]))[0].sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    moved = set((hybrid.residual.weight.abs().sum(dim=-1) != 0).nonzero().flatten().tolist())
    assert moved <= {5, 6, 7, 8, 9}, f"gradient leaked into {moved - {5, 6, 7, 8, 9}}"
    assert 5 in moved

    # Everything untouched is still exactly pure text.
    untouched = [i for i in range(10, hybrid.n_items)]
    assert hybrid.residual.weight[untouched].abs().sum().item() == 0.0


def test_popularity_negatives_never_draw_zero_count_items():
    """Why cold items stay pure text in a real run.

    Cold items are absent from the training period so they are never positives.
    They are also never negatives, because popularity sampling weights by
    training count and a count of zero has zero probability. That is the second
    half of the guarantee and it lives in sample_negatives, so it is worth
    pinning here rather than assuming.
    """
    counts = torch.zeros(50, dtype=torch.long)
    counts[10:20] = 100  # only these ten items exist in training

    drawn = sample_negatives(
        n_items=50, shape=(4000,), counts=counts, first_item_id=2, device="cpu"
    )
    assert set(drawn.tolist()) <= set(range(10, 20))


def test_factory_builds_hybrid(small, text_vectors):
    vc, _ = small
    repr_ = build_item_representation("hybrid", len(vc), 32, text_vectors=text_vectors)
    assert isinstance(repr_, HybridItemRepresentation)
    assert repr_.n_items == len(vc)
    assert repr_.d_model == 32


def test_factory_rejects_hybrid_without_text(small):
    vc, _ = small
    with pytest.raises(ValueError, match="needs text_vectors"):
        build_item_representation("hybrid", len(vc), 32)


def test_factory_rejects_misaligned_text(small, text_vectors):
    vc, _ = small
    with pytest.raises(ValueError, match="aligned by item id"):
        build_item_representation("hybrid", len(vc) + 5, 32, text_vectors=text_vectors)


def test_checkpoint_roundtrip(small, text_vectors, tmp_path):
    """A reloaded hybrid must score identically, including its nested buffer."""
    vc, seqs = small
    model, head, scorer = build_scorer(
        n_items=len(vc),
        device="cpu",
        variant="hybrid",
        text_vectors=text_vectors,
        d_model=32,
        n_blocks=1,
        max_len=20,
    )
    model.eval()
    head.eval()

    # Move the residual off zero so the round-trip actually exercises it.
    with torch.no_grad():
        model.items.residual.weight[5:9].normal_(std=0.1)

    histories = [s[:15] for s in seqs.items[:3] if len(s) >= 15]
    times = [t[:15] for t, s in zip(seqs.times, seqs.items) if len(s) >= 15][: len(histories)]
    assert histories

    before = scorer.score(histories, times)
    path = ckpt.save(
        tmp_path / "h.pt",
        model,
        head,
        vc,
        config={"variant": "hybrid", "d_model": 32, "n_blocks": 1, "max_len": 20},
    )
    after = ckpt.load(path, device="cpu").scorer.score(histories, times)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
