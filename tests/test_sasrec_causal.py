"""Causal-masking and padding invariants for the encoder.

The causal test is the non-negotiable one from the plan. It is written as a
*behavioural* test --- perturb a future input, assert earlier outputs are
bit-identical --- rather than an inspection of the mask tensor. Asserting the
mask looks right only proves the mask looks right; asserting no information
flows backwards proves the thing that actually matters, and would still catch a
regression if someone swapped the hand-rolled attention for a library kernel
with the opposite mask convention.
"""

from __future__ import annotations

import pytest
import torch

from melochron.models.heads import TiedItemScorer, sample_negatives
from melochron.models.sasrec import SASRec

N_ITEMS = 40
MAX_LEN = 16


@pytest.fixture
def model() -> SASRec:
    torch.manual_seed(0)
    # eval() matters: dropout would make two forward passes differ for reasons
    # that have nothing to do with masking.
    return SASRec(
        n_items=N_ITEMS, d_model=32, n_heads=4, n_blocks=2, max_len=MAX_LEN, dropout=0.1
    ).eval()


def _seq(batch: int = 2, length: int = MAX_LEN, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, N_ITEMS, (batch, length), generator=g)


def test_future_item_does_not_change_past_outputs(model: SASRec) -> None:
    seq = _seq()

    with torch.no_grad():
        base = model(seq)

    for t in range(1, MAX_LEN):
        perturbed = seq.clone()
        # Force a genuinely different item at position t.
        perturbed[:, t] = (perturbed[:, t] - 2 + 7) % (N_ITEMS - 2) + 2
        assert not torch.equal(perturbed[:, t], seq[:, t])

        with torch.no_grad():
            after = model(perturbed)

        torch.testing.assert_close(
            after[:, :t], base[:, :t], msg=f"perturbing position {t} leaked into earlier positions"
        )


def test_future_time_delta_does_not_change_past_outputs(model: SASRec) -> None:
    """The time channel has to respect causality too, not just the item channel."""
    seq = _seq()
    deltas = torch.full((2, MAX_LEN), 60, dtype=torch.long)

    with torch.no_grad():
        base = model(seq, deltas)

    for t in range(1, MAX_LEN):
        perturbed = deltas.clone()
        perturbed[:, t] = 60 * 60 * 24 * 200  # a months-long gap, a very different bucket

        with torch.no_grad():
            after = model(seq, perturbed)

        torch.testing.assert_close(
            after[:, :t], base[:, :t], msg=f"time delta at {t} leaked into earlier positions"
        )


def test_left_padding_produces_no_nan(model: SASRec) -> None:
    """A short, heavily left-padded sequence must not produce NaN.

    This is the failure the identity term in ``build_attention_mask`` exists to
    prevent: a leading query position whose only visible key is its own pad slot
    would otherwise softmax over a row of -inf.
    """
    seq = torch.zeros(3, MAX_LEN, dtype=torch.long)
    seq[0, -1:] = 5  # one real event, fifteen pads
    seq[1, -3:] = torch.tensor([5, 6, 7])
    seq[2, :] = 9  # no padding at all

    with torch.no_grad():
        out = model(seq)

    assert torch.isfinite(out).all(), "left-padded sequence produced NaN or inf"


def test_padded_positions_are_zeroed(model: SASRec) -> None:
    seq = torch.zeros(1, MAX_LEN, dtype=torch.long)
    seq[0, -4:] = torch.tensor([3, 4, 5, 6])

    with torch.no_grad():
        out = model(seq)

    assert torch.count_nonzero(out[0, :-4]) == 0, "pad positions carry a non-zero representation"
    assert torch.count_nonzero(out[0, -4:]) > 0, "real positions were zeroed"


def test_padding_length_does_not_change_the_answer(model: SASRec) -> None:
    """The same history must score the same however much padding precedes it.

    This is what the right-to-left position numbering buys, and it is the
    invariant that silently breaks if anyone switches to right-padding.
    """
    history = torch.tensor([[11, 12, 13, 14]])
    short = torch.nn.functional.pad(history, (2, 0), value=0)
    long = torch.nn.functional.pad(history, (MAX_LEN - 4, 0), value=0)

    with torch.no_grad():
        a = model.encode_last(short)
        b = model.encode_last(long)

    torch.testing.assert_close(a, b)


def test_encode_last_is_the_final_position(model: SASRec) -> None:
    seq = _seq()
    with torch.no_grad():
        torch.testing.assert_close(model.encode_last(seq), model(seq)[:, -1])


def test_sequence_longer_than_max_len_is_rejected(model: SASRec) -> None:
    with pytest.raises(ValueError, match="exceeds max_len"):
        model(_seq(length=MAX_LEN + 1))


def test_head_is_tied_to_the_item_representation(model: SASRec) -> None:
    head = TiedItemScorer(model.items)
    assert head.items is model.items

    with torch.no_grad():
        model.items.embedding.weight[5] += 1.0
    torch.testing.assert_close(head.items.item_vectors()[5], model.items.item_vectors()[5])


def test_full_logits_never_rank_pad_or_oov(model: SASRec) -> None:
    head = TiedItemScorer(model.items)
    with torch.no_grad():
        logits = head.full_logits(model.encode_last(_seq()))

    assert torch.isneginf(logits[:, 0]).all(), "PAD is rankable"
    assert torch.isneginf(logits[:, 1]).all(), "OOV is rankable"
    assert torch.isfinite(logits[:, 2:]).all(), "a real item was masked out"


def test_negative_sampling_avoids_reserved_ids() -> None:
    uniform = sample_negatives(N_ITEMS, (64, 8))
    assert uniform.min() >= 2 and uniform.max() < N_ITEMS

    counts = torch.zeros(N_ITEMS)
    counts[2:] = torch.arange(1, N_ITEMS - 1, dtype=torch.float)
    popular = sample_negatives(N_ITEMS, (64, 8), counts=counts)
    assert popular.min() >= 2 and popular.max() < N_ITEMS
    assert popular.shape == (64, 8)
