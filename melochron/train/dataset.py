"""Windowing and collation for next-item training.

Turns variable-length user histories into fixed-width left-padded batches. Two
choices here are worth stating because both are easy to get wrong silently.

**Every position is a training example, not just the last.** A window of ``L+1``
items yields ``L`` next-item predictions: input ``w[:-1]``, target ``w[1:]``.
Training only on the final position of each sequence would throw away almost
all the supervision in the data and, on a corpus this size, is the difference
between a model that converges and one that does not.

**Left padding, per the model contract.** ``SASRec.encode_last`` takes
``[:, -1]``, which is correct only if the most recent event is at index ``-1``.
Right padding would not crash: it would quietly score a pad slot as the
prediction position and produce plausible-looking garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from melochron.data.sessions import Sequences
from melochron.data.vocab import OOV_ID, PAD_ID


@dataclass
class Batch:
    """One training batch. All tensors are ``[B, L]`` except as noted."""

    item_ids: torch.Tensor  # input items, left-padded
    targets: torch.Tensor  # next item at each position, PAD where absent
    timestamps: torch.Tensor  # absolute unix seconds, aligned to item_ids
    mask: torch.Tensor  # bool, True where the target is a real item

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            item_ids=self.item_ids.to(device),
            targets=self.targets.to(device),
            timestamps=self.timestamps.to(device),
            mask=self.mask.to(device),
        )

    def __len__(self) -> int:
        return self.item_ids.shape[0]


class WindowDataset(Dataset):
    """Sliding windows over user histories.

    ``stride`` defaults to ``max_len``, giving disjoint windows so each event is
    a prediction target exactly once per epoch. A smaller stride increases
    supervision per epoch but duplicates targets, which biases the effective
    sample weighting toward users with long histories. Disjoint is the honest
    default; the knob exists because on a small corpus the extra windows are
    worth the redundancy.
    """

    def __init__(
        self,
        seqs: Sequences,
        max_len: int = 200,
        stride: int | None = None,
        min_targets: int = 2,
    ):
        self.seqs = seqs
        self.max_len = max_len
        self.stride = stride or max_len
        self.index: list[tuple[int, int]] = []

        for u, items in enumerate(seqs.items):
            n = len(items)
            if n < min_targets + 1:
                continue
            # `end` is the index of the last target in the window. Walk
            # backwards from the most recent event so that the newest data is
            # always aligned to a window boundary: the tail of a user's history
            # is the part the test period actually follows on from.
            end = n - 1
            while end >= min_targets:
                self.index.append((u, end))
                end -= self.stride

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, end = self.index[i]
        items, times = self.seqs.items[u], self.seqs.times[u]

        # L+1 items produce L (input, target) pairs.
        start = max(0, end - self.max_len)
        window = items[start : end + 1]
        window_ts = times[start : end + 1]
        return window[:-1], window[1:], window_ts[:-1]


def collate(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], max_len: int) -> Batch:
    """Left-pad a list of windows into a rectangular batch."""
    b = len(rows)
    width = min(max_len, max(len(r[0]) for r in rows))

    item_ids = np.full((b, width), PAD_ID, dtype=np.int64)
    targets = np.full((b, width), PAD_ID, dtype=np.int64)
    timestamps = np.zeros((b, width), dtype=np.int64)

    for i, (inp, tgt, ts) in enumerate(rows):
        inp, tgt, ts = inp[-width:], tgt[-width:], ts[-width:]
        n = len(inp)
        item_ids[i, width - n :] = inp
        targets[i, width - n :] = tgt
        timestamps[i, width - n :] = ts
        # Pad the timestamp region with the window's own first timestamp, not
        # zero. A zero would make the first real event's gap read as ~55 years
        # and land it in the largest time bucket instead of being masked out.
        timestamps[i, : width - n] = ts[0] if n else 0

    return Batch(
        item_ids=torch.from_numpy(item_ids),
        targets=torch.from_numpy(targets),
        timestamps=torch.from_numpy(timestamps),
        # Supervise only on targets that are real, rankable items.
        #
        # PAD is obvious. OOV is the subtle one, and the asymmetry is
        # deliberate: OOV is kept in the *input*, because "you played something
        # outside the catalog" is genuine information about the sequence, but
        # excluded as a *target*, because the evaluation head drives OOV to
        # -inf and it can never be a legal recommendation. Training the model
        # to score a token it is forbidden to predict spends gradient on
        # pushing up a logit that is thrown away, and on a corpus where the
        # tail is large that is a substantial fraction of the batch.
        mask=torch.from_numpy((targets != PAD_ID) & (targets != OOV_ID)),
    )


def make_loader(
    seqs: Sequences,
    max_len: int = 200,
    batch_size: int = 128,
    stride: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = WindowDataset(seqs, max_len=max_len, stride=stride)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda rows: collate(rows, max_len),
        # Windows are numpy slices of arrays already in memory; a worker pool
        # would cost more in pickling than it saves.
        num_workers=0,
        drop_last=False,
    )
