"""Adapts :class:`~melochron.models.sasrec.SASRec` to the evaluation harness.

`eval/protocol.py` defines one `Scorer` seam that every model and every
baseline is ranked through, which is what makes the results table an
apples-to-apples comparison by construction. This module is the transformer's
implementation of that seam, and it is deliberately the only place where the
two lanes' conventions are reconciled:

- the harness passes **absolute** unix timestamps; the encoder wants **gaps in
  seconds**, so the conversion happens here
- the harness passes **variable-length** histories as a list of arrays; the
  encoder wants a **left-padded** rectangular batch
- the harness indexes targets by **raw vocab id**, so the returned score matrix
  must be the full vocabulary width, PAD and OOV columns included

Nothing here changes the protocol and nothing here changes the model. Putting
the impedance mismatch in one adapter is what keeps both sides clean.
"""

from __future__ import annotations

import numpy as np
import torch

from melochron.models.heads import TiedItemScorer
from melochron.models.item_repr import build_item_representation
from melochron.models.sasrec import SASRec
from melochron.models.time_encoding import deltas_from_timestamps


def left_pad(
    sequences: list[np.ndarray], max_len: int, pad_value: int = 0
) -> tuple[torch.Tensor, int]:
    """Stack variable-length sequences into a left-padded ``[B, L]`` tensor.

    Left, not right, so that column ``-1`` is the most recent event for every
    row regardless of history length. ``L`` is the longest history in the batch
    capped at ``max_len``, so short batches do not pay for the full window.
    Histories longer than ``max_len`` keep their **most recent** ``max_len``
    events; truncating the other end would throw away the recent context that
    carries nearly all of the signal.
    """
    length = max(1, min(max_len, max((len(s) for s in sequences), default=1)))
    out = np.full((len(sequences), length), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = np.asarray(seq, dtype=np.int64)[-length:]
        if len(seq):
            out[i, length - len(seq) :] = seq
    return torch.from_numpy(out), length


class SASRecScorer:
    """Scores the full catalog for a batch of histories.

    Satisfies the ``Scorer`` protocol in :mod:`melochron.eval.protocol`.
    """

    def __init__(
        self,
        model: SASRec,
        head: TiedItemScorer,
        device: torch.device | str = "cpu",
        name: str = "sasrec",
        use_time: bool = True,
        mask_reserved: bool = True,
        batch_size: int = 128,
    ):
        self.model = model.to(device).eval()
        self.head = head.to(device).eval()
        self.device = torch.device(device)
        self.name = name
        self.use_time = use_time
        self.mask_reserved = mask_reserved
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, histories: list[np.ndarray], times: list[np.ndarray]) -> np.ndarray:
        """``(len(histories), vocab_size)`` scores, one row per instance."""
        if len(histories) != len(times):
            raise ValueError(f"got {len(histories)} histories but {len(times)} time arrays")
        if not histories:
            return np.empty((0, self.model.n_items), dtype=np.float32)

        out = np.empty((len(histories), self.model.n_items), dtype=np.float32)

        # Computed once for the whole call. For a projected text representation
        # this is an [n_items, d_text] x [d_text, d_model] product that would
        # otherwise repeat on every batch of a full-catalog evaluation.
        item_vectors = self.head.items.item_vectors()

        for start in range(0, len(histories), self.batch_size):
            stop = min(start + self.batch_size, len(histories))
            item_ids, length = left_pad(
                histories[start:stop], self.model.max_len, self.model.pad_id
            )
            item_ids = item_ids.to(self.device)

            deltas = None
            if self.use_time and self.model.time_encoding is not None:
                # Pad timestamps with the sequence's own first timestamp rather
                # than zero: a zero would make the first real event's gap look
                # like fifty years, landing it in the largest bucket instead of
                # the "no predecessor" one.
                ts = np.zeros((stop - start, length), dtype=np.int64)
                for i, t in enumerate(times[start:stop]):
                    t = np.asarray(t, dtype=np.int64)[-length:]
                    if len(t):
                        ts[i, length - len(t) :] = t
                        ts[i, : length - len(t)] = t[0]
                deltas = deltas_from_timestamps(torch.from_numpy(ts).to(self.device))

            hidden = self.model.encode_last(item_ids, deltas)
            logits = self.head.full_logits(
                hidden, mask_reserved=self.mask_reserved, item_vectors=item_vectors
            )
            out[start:stop] = logits.float().cpu().numpy()

        return out


def build_scorer(
    n_items: int,
    device: torch.device | str = "cpu",
    name: str = "sasrec",
    variant: str = "id",
    text_vectors: torch.Tensor | None = None,
    d_model: int = 128,
    **model_kwargs,
) -> tuple[SASRec, TiedItemScorer, SASRecScorer]:
    """Construct an untrained model, its tied head, and the eval adapter.

    ``variant`` selects the Phase 2 ablation row --- ``"id"``,
    ``"text_frozen"``, or ``"text_finetuned"`` --- so all three are built
    through one path and cannot drift apart.

    Returned together because the head holds a live reference to the model's
    item representation; building them separately is the easy way to
    accidentally untie the weights.
    """
    items = build_item_representation(
        variant, n_items=n_items, d_model=d_model, text_vectors=text_vectors
    )
    model = SASRec(n_items=n_items, d_model=d_model, item_repr=items, **model_kwargs)
    head = TiedItemScorer(model.items, pad_id=model.pad_id)
    return model, head, SASRecScorer(model, head, device=device, name=name)
