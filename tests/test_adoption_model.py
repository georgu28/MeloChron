"""Guards for the adoption model, its windows, and its training loop.

The one that matters above all is `test_window_excludes_the_encounter_and_after`.
The corpus is stored most-recent-first at source and the encoder is left-padded;
a single off-by-one in the window builder would feed the model the very event it
is trying to predict, and every metric would look wonderful and mean nothing. So
the window is checked event-by-event against a hand-written expectation, and the
model's history vector is checked to be invariant to anything at or after the
encounter.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from melochron.adoption.model import AdoptionModel
from melochron.adoption.train import (
    Corpus,
    Examples,
    TrainConfig,
    load_checkpoint,
    predict,
    save_checkpoint,
    train,
)
from melochron.adoption.windows import PAD_ID, build_windows

BASE_TS = 1_370_044_800


def one_user_corpus(n_events=20):
    """A single user whose k-th event is track k, one minute apart."""
    track_code = np.arange(n_events, dtype=np.int32)
    ts = (BASE_TS + np.arange(n_events) * 60).astype(np.int32)
    user_offsets = np.array([0, n_events], dtype=np.int64)
    return Corpus(track_code, ts, user_offsets)


class TestWindows:
    def test_window_excludes_the_encounter_and_after(self):
        """History for the encounter at position 10 is exactly tracks 0..9,
        never track 10 or later, right-aligned with track 9 last."""
        corpus = one_user_corpus(20)
        items, _ = build_windows(
            corpus.track_code,
            corpus.ts,
            corpus.user_offsets,
            users=np.array([0]),
            positions=np.array([10]),
            max_len=8,
        )
        # max_len 8, so the window holds positions 2..9 → track_code+1 = 3..10.
        assert items[0].tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
        # track 10 (the encounter, item id 11) and later never appear.
        assert 11 not in items[0].tolist()

    def test_short_history_is_left_padded(self):
        corpus = one_user_corpus(20)
        items, deltas = build_windows(
            corpus.track_code,
            corpus.ts,
            corpus.user_offsets,
            users=np.array([0]),
            positions=np.array([3]),
            max_len=8,
        )
        # Only positions 0,1,2 precede the encounter → 5 pads then items 1,2,3.
        assert items[0].tolist() == [PAD_ID] * 5 + [1, 2, 3]
        assert (deltas[0][:5] == 0).all()  # pad columns carry no gap

    def test_first_encounter_has_an_all_pad_window(self):
        corpus = one_user_corpus(20)
        items, _ = build_windows(
            corpus.track_code,
            corpus.ts,
            corpus.user_offsets,
            users=np.array([0]),
            positions=np.array([0]),
            max_len=8,
        )
        assert (items[0] == PAD_ID).all()

    def test_interior_gaps_are_the_real_seconds(self):
        corpus = one_user_corpus(20)
        _, deltas = build_windows(
            corpus.track_code,
            corpus.ts,
            corpus.user_offsets,
            users=np.array([0]),
            positions=np.array([10]),
            max_len=8,
        )
        # Events are 60s apart; every interior gap is 60.
        assert (deltas[0][1:] == 60).all()

    def test_positions_are_per_user(self):
        # Two users, 5 events each. User 1's position 2 must gather user 1's
        # events, not run into user 0's slice.
        track_code = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], dtype=np.int32)
        ts = (BASE_TS + np.arange(10) * 60).astype(np.int32)
        user_offsets = np.array([0, 5, 10], dtype=np.int64)

        items, _ = build_windows(
            track_code,
            ts,
            user_offsets,
            users=np.array([1]),
            positions=np.array([2]),
            max_len=4,
        )
        # user 1's positions 0,1 → track_code 10,11 → item ids 11,12.
        assert items[0].tolist() == [PAD_ID, PAD_ID, 11, 12]


class TestModelInvariance:
    def _corpus_and_examples(self, n_events=16):
        corpus = one_user_corpus(n_events)
        examples = Examples(
            users=np.array([0]),
            positions=np.array([8]),
            candidates=np.array([8], dtype=np.int32),
            labels=np.array([True]),
        )
        return corpus, examples

    def test_history_vector_ignores_events_at_or_after_the_encounter(self):
        """The behavioural no-look-ahead test, in the encoder's own style:
        perturb the encounter and everything after it, and the history vector
        the model builds must not move."""
        torch.manual_seed(0)
        corpus, examples = self._corpus_and_examples(16)
        model = AdoptionModel(n_items=32, d_model=16, n_heads=2, n_blocks=2, max_len=8, dropout=0.0)
        model.eval()
        device = torch.device("cpu")

        p_before = predict(model, corpus, examples, max_len=8, device=device)

        # Corrupt every event from the encounter onward. The window for position
        # 8 covers positions 0..7, so nothing here should change the score.
        corrupted = Corpus(corpus.track_code.copy(), corpus.ts.copy(), corpus.user_offsets)
        corrupted.track_code[8:] = 0
        corrupted.ts[8:] = corrupted.ts[7] + 999_999
        p_after = predict(model, corrupted, examples, max_len=8, device=device)

        assert p_before == pytest.approx(p_after, abs=1e-6)

    def test_priors_head_refuses_missing_priors(self):
        model = AdoptionModel(n_items=32, d_model=16, max_len=8, use_priors=True)
        item = torch.zeros(2, 8, dtype=torch.long)
        dt = torch.zeros(2, 8, dtype=torch.long)
        cand = torch.ones(2, dtype=torch.long)

        with pytest.raises(ValueError, match="got no priors"):
            model(item, dt, cand, None)


class TestTraining:
    def _synthetic(self, n_users=40, per_user=30, seed=0):
        """A corpus where adoption is learnable: a track recurs iff its code is
        even, so the candidate embedding alone carries the signal."""
        rng = np.random.default_rng(seed)
        track_code, ts, offsets = [], [], [0]
        users, positions, candidates, labels, enc_ts = [], [], [], [], []
        t = BASE_TS
        for u in range(n_users):
            seq = rng.integers(0, 20, size=per_user).astype(np.int32)
            for p, code in enumerate(seq):
                track_code.append(int(code))
                ts.append(t)
                t += 60
                if p >= 3:  # give every encounter some history
                    users.append(u)
                    positions.append(p)
                    candidates.append(int(code))
                    labels.append(bool(code % 2 == 0))
                    enc_ts.append(t)
            offsets.append(len(track_code))
        corpus = Corpus(
            np.array(track_code, dtype=np.int32),
            np.array(ts, dtype=np.int32),
            np.array(offsets, dtype=np.int64),
        )
        examples = Examples(
            np.array(users),
            np.array(positions),
            np.array(candidates, dtype=np.int32),
            np.array(labels),
        )
        return corpus, examples, np.array(enc_ts)

    def test_overfits_a_small_dataset(self):
        """The standard gradient-path proof: the loop can drive training BCE
        down and validation PR-AUC well above the base rate on a learnable
        signal. If it cannot, the wiring is wrong."""
        corpus, examples, enc_ts = self._synthetic()
        model = AdoptionModel(
            n_items=32, d_model=16, n_heads=2, n_blocks=1, max_len=16, dropout=0.0
        )
        config = TrainConfig(max_len=16, batch_size=64, epochs=15, patience=15, seed=0)

        result = train(model, corpus, examples, enc_ts, config, torch.device("cpu"))

        assert result["best_val_pr_auc"] > 0.75
        assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]

    def test_shuffled_labels_do_not_generalise(self):
        """The leakage floor, measured on held-out rows.

        Labels are shuffled, destroying any relationship to the features, and
        the model is trained on the first 70% and scored on the last 30% it
        never saw. It can memorise the training rows all it likes; on unseen
        rows with shuffled labels there is nothing honest to predict, so a
        PR-AUC meaningfully above the base rate would mean a feature is carrying
        the label through the pipeline — the leak this test exists to catch.
        Scoring on the training rows instead would measure memorisation, not
        leakage, and would fail on any model with enough capacity to overfit.
        """
        corpus, examples, enc_ts = self._synthetic(seed=1)
        rng = np.random.default_rng(2)
        labels = rng.permutation(examples.labels)

        n = len(examples)
        cut = int(n * 0.7)
        tr, te = np.arange(cut), np.arange(cut, n)
        train_ex = Examples(
            examples.users[tr], examples.positions[tr], examples.candidates[tr], labels[tr]
        )
        test_ex = Examples(
            examples.users[te], examples.positions[te], examples.candidates[te], labels[te]
        )

        model = AdoptionModel(
            n_items=32, d_model=16, n_heads=2, n_blocks=1, max_len=16, dropout=0.0
        )
        config = TrainConfig(max_len=16, batch_size=64, epochs=8, patience=8, seed=0)
        train(model, corpus, train_ex, enc_ts[tr], config, torch.device("cpu"))

        probs = predict(model, corpus, test_ex, 16, torch.device("cpu"))
        from melochron.adoption import metrics

        score = metrics.evaluate(test_ex.labels, probs)
        assert score.pr_auc < score.base_rate + 0.08

    def test_checkpoint_round_trips(self, tmp_path):
        corpus, examples, enc_ts = self._synthetic()
        model = AdoptionModel(
            n_items=32, d_model=16, n_heads=2, n_blocks=1, max_len=16, dropout=0.0
        )
        config = TrainConfig(max_len=16, batch_size=64, epochs=2, patience=2)
        train(model, corpus, examples, enc_ts, config, torch.device("cpu"))
        before = predict(model, corpus, examples, 16, torch.device("cpu"))

        save_checkpoint(tmp_path / "m.pt", model, config, {"note": "test"})
        reloaded, payload = load_checkpoint(tmp_path / "m.pt")
        after = predict(reloaded, corpus, examples, 16, torch.device("cpu"))

        assert before == pytest.approx(after, abs=1e-6)
        assert payload["metrics"]["note"] == "test"
