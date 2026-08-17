"""Tests for windowing, masking, the loss, and checkpoint round-trips.

The masking tests are the load-bearing ones. A supervision mask that is wrong
does not crash and does not obviously degrade anything: it just quietly trains
the model on positions that carry no label, and the effect shows up only as a
model that underperforms for no visible reason.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melochron.data import sessions, synthetic, vocab
from melochron.data.vocab import OOV_ID, PAD_ID
from melochron.models.scorer import build_scorer
from melochron.train import checkpoint as ckpt
from melochron.train.dataset import WindowDataset, collate, make_loader
from melochron.train.loop import TrainConfig, cosine_with_warmup
from melochron.train.losses import bpr_loss, get_loss, sampled_softmax_loss


@pytest.fixture(scope="module")
def small():
    events, _ = synthetic.generate(synthetic.SyntheticConfig(n_users=6, n_months=3, seed=11))
    positives = sessions.filter_positives(events)
    vc = vocab.build_vocab(positives, min_count=3)
    seqs = sessions.build_sequences(positives, vc)
    return vc, seqs


def test_collate_puts_most_recent_event_last():
    """Left padding is a contract: encode_last reads index -1."""
    rows = [
        (np.array([2, 3, 4]), np.array([3, 4, 5]), np.array([100, 200, 300])),
        (np.array([6]), np.array([7]), np.array([500])),
    ]
    batch = collate(rows, max_len=3)

    assert batch.item_ids.shape == (2, 3)
    assert batch.item_ids[0].tolist() == [2, 3, 4]
    assert batch.item_ids[1].tolist() == [PAD_ID, PAD_ID, 6]
    assert batch.item_ids[1, -1].item() == 6
    assert batch.targets[1, -1].item() == 7


def test_collate_pads_timestamps_with_first_real_value():
    """A zero pad would make the first gap read as decades, not as absent."""
    rows = [(np.array([6]), np.array([7]), np.array([1_600_000_000]))]
    batch = collate(rows, max_len=4)
    assert (batch.timestamps[0] == 1_600_000_000).all()


def test_mask_excludes_pad_and_oov_targets():
    """OOV is legal input but never a legal target.

    The evaluation head drives OOV to -inf, so supervising toward it spends
    gradient raising a logit that is discarded before ranking.
    """
    rows = [(np.array([2, OOV_ID, 4]), np.array([OOV_ID, 4, 5]), np.array([1, 2, 3]))]
    batch = collate(rows, max_len=3)

    assert batch.mask.tolist() == [[False, True, True]]
    assert batch.item_ids[0, 1].item() == OOV_ID  # kept as input


def test_window_dataset_covers_recent_history_first(small):
    _, seqs = small
    ds = WindowDataset(seqs, max_len=50, stride=50)
    assert len(ds) > 0

    inp, tgt, ts = ds[0]
    assert len(inp) == len(tgt) == len(ts)
    # Targets are the inputs shifted by one.
    assert inp[1:].tolist() == tgt[:-1].tolist()
    assert np.all(np.diff(ts) >= 0)


def test_disjoint_stride_targets_each_event_about_once(small):
    _, seqs = small
    max_len = 40
    ds = WindowDataset(seqs, max_len=max_len, stride=max_len)
    expected = sum(max(0, (len(s) - 2) // max_len + 1) for s in seqs.items)
    assert abs(len(ds) - expected) <= len(seqs)


@pytest.mark.parametrize("loss_name", ["sampled_softmax", "bpr"])
def test_loss_ignores_masked_rows(loss_name):
    """Perturbing a masked row must not move the loss at all."""
    loss_fn = get_loss(loss_name)
    torch.manual_seed(0)

    positive = torch.randn(6, 1)
    negative = torch.randn(6, 8)
    mask = torch.tensor([True, True, False, True, False, True])

    before = loss_fn(positive, negative, mask)

    positive = positive.clone()
    negative = negative.clone()
    positive[2] += 100.0
    negative[4] -= 100.0

    after = loss_fn(positive, negative, mask)
    assert torch.allclose(before, after)


def test_loss_rewards_ranking_the_positive_higher():
    good = sampled_softmax_loss(
        torch.full((4, 1), 5.0), torch.zeros(4, 8), torch.ones(4, dtype=torch.bool)
    )
    bad = sampled_softmax_loss(
        torch.zeros(4, 1), torch.full((4, 8), 5.0), torch.ones(4, dtype=torch.bool)
    )
    assert good < bad
    assert bpr_loss(
        torch.full((4, 1), 5.0), torch.zeros(4, 8), torch.ones(4, dtype=torch.bool)
    ) < bpr_loss(torch.zeros(4, 1), torch.full((4, 8), 5.0), torch.ones(4, dtype=torch.bool))


def test_all_masked_batch_is_finite():
    """A batch with no supervision must yield 0, not NaN."""
    loss = sampled_softmax_loss(
        torch.randn(3, 1), torch.randn(3, 4), torch.zeros(3, dtype=torch.bool)
    )
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_schedule_warms_up_then_decays():
    total, warmup = 100, 10
    assert cosine_with_warmup(0, total, warmup) < cosine_with_warmup(5, total, warmup)
    assert cosine_with_warmup(warmup - 1, total, warmup) == pytest.approx(1.0)
    assert cosine_with_warmup(total - 1, total, warmup) < 0.05
    assert cosine_with_warmup(total * 2, total, warmup) >= 0.0


def test_checkpoint_roundtrip_preserves_scores(small, tmp_path):
    """A reloaded artifact must score identically, or reported metrics are not
    the metrics the served model produces."""
    vc, seqs = small
    model, head, scorer = build_scorer(
        n_items=len(vc), device="cpu", variant="id", d_model=32, n_blocks=1, max_len=20
    )
    model.eval()
    head.eval()

    histories = [s[:15] for s in seqs.items[:3] if len(s) >= 15]
    times = [t[:15] for t, s in zip(seqs.times, seqs.items) if len(s) >= 15][: len(histories)]
    assert histories, "fixture produced no usable history"

    before = scorer.score(histories, times)

    cfg = TrainConfig(variant="id", d_model=32, n_blocks=1, max_len=20)
    path = ckpt.save(tmp_path / "m.pt", model, head, vc, config=vars(cfg) | {}, metrics={"x": 1})
    loaded = ckpt.load(path, device="cpu")

    after = loaded.scorer.score(histories, times)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
    assert loaded.vocab.id_to_key == vc.id_to_key
    assert loaded.metrics["x"] == 1


def test_checkpoint_rejects_wrong_format_version(small, tmp_path):
    vc, _ = small
    model, head, _ = build_scorer(n_items=len(vc), device="cpu", d_model=32, n_blocks=1)
    path = ckpt.save(tmp_path / "m.pt", model, head, vc, config={"variant": "id", "d_model": 32})

    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format version"):
        ckpt.load(path, device="cpu")


def test_loader_produces_usable_batches(small):
    _, seqs = small
    loader = make_loader(seqs, max_len=32, batch_size=4, seed=0)
    batch = next(iter(loader))

    assert batch.item_ids.shape == batch.targets.shape == batch.mask.shape
    assert batch.item_ids.shape[1] <= 32
    assert batch.mask.any(), "batch carries no supervision at all"
    assert (batch.targets[batch.mask] > OOV_ID).all()
