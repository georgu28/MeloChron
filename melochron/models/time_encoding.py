"""Time-interval encoding for sequential recommendation.

Position tells the model *what order* events happened in. It does not tell it
that two tracks were played back to back in one sitting versus eight months
apart, and for listening histories that difference carries most of the signal:
consecutive plays inside a session are strongly predictive, plays either side
of a long gap are barely related at all.

So inter-event gaps are bucketed on a log scale and embedded alongside
position. Log scale because the informative structure is multiplicative --- the
difference between 5 seconds and 5 minutes matters enormously, the difference
between 5 months and 10 months almost not at all. A linear encoding would spend
most of its resolution where none is needed.

This is time-interval-aware attention in the spirit of TiSASRec (Li, Wang,
McAuley, WSDM 2020), implemented by hand. TiSASRec injects relative intervals
into the attention computation itself; this is the cheaper additive variant,
which keeps the attention kernel standard and is ablatable against
position-only by construction (pass ``time_deltas=None``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_MIN = 60
_HOUR = 60 * _MIN
_DAY = 24 * _HOUR
_WEEK = 7 * _DAY
_MONTH = 30 * _DAY
_YEAR = 365 * _DAY

#: Upper edges, in seconds, of the inter-event gap buckets. Dense through the
#: session-scale range (seconds to an hour) where consecutive plays actually
#: discriminate, sparse out at the months-to-years end where they do not.
BOUNDARIES: tuple[int, ...] = (
    1,
    5,
    10,
    30,
    1 * _MIN,
    2 * _MIN,
    5 * _MIN,
    10 * _MIN,
    30 * _MIN,
    1 * _HOUR,
    2 * _HOUR,
    6 * _HOUR,
    12 * _HOUR,
    1 * _DAY,
    2 * _DAY,
    1 * _WEEK,
    2 * _WEEK,
    1 * _MONTH,
    3 * _MONTH,
    6 * _MONTH,
    1 * _YEAR,
)

#: Bucket id reserved for padded positions, which have no meaningful gap.
PAD_BUCKET = 0

#: Bucket for the first real event in a sequence, which has no predecessor and
#: therefore no gap. It needs its own id rather than sharing the zero-second
#: bucket: "this is where the history starts" and "these two tracks played
#: back to back" are opposite situations, and collapsing them would blur
#: exactly the session-boundary signal the time channel exists to provide.
NO_PREDECESSOR_BUCKET = 1

#: Real gap buckets occupy ``2 .. len(BOUNDARIES) + 2`` inclusive.
N_BUCKETS = len(BOUNDARIES) + 3


def first_real_position(mask: Tensor) -> Tensor:
    """``[B, L]`` boolean, True at each row's earliest unpadded position.

    Derived from the padding mask rather than taken on trust from the caller,
    which is what makes :func:`bucketize` safe against a badly-padded timestamp
    array: whatever garbage the leading delta holds, it is overridden.
    """
    shifted = torch.zeros_like(mask)
    shifted[:, 1:] = mask[:, :-1]
    return mask & ~shifted


def bucketize(deltas: Tensor, mask: Tensor | None = None) -> Tensor:
    """Map inter-event gaps in seconds to bucket ids.

    ``deltas`` is ``[B, L]`` of gaps between consecutive events. ``mask`` is
    ``[B, L]``, True at real positions; padded positions are forced to
    :data:`PAD_BUCKET` so they cannot alias onto a real bucket and teach the
    time embedding something about padding.

    The first real position of every row is forced to
    :data:`NO_PREDECESSOR_BUCKET` regardless of the delta supplied for it. That
    is deliberate defensiveness at the seam: a left-padded timestamp array
    padded with zeros would otherwise hand that position a gap of ~1.7e9
    seconds --- fifty-five years --- and silently file the start of every user's
    history in the largest bucket, which is the same signal as a genuine
    multi-year absence.

    Negative gaps are clamped rather than raised on: a handful of out-of-order
    timestamps is normal in scrobble data (clients backfill), and killing a
    training run over one is the wrong trade.
    """
    deltas = deltas.clamp(min=0)
    edges = torch.as_tensor(BOUNDARIES, device=deltas.device, dtype=deltas.dtype)

    # right=True makes this "how many edges does the gap meet or exceed", so a
    # gap of exactly one hour lands in the bucket above the 1-hour edge, not on it.
    buckets = torch.bucketize(deltas, edges, right=True) + 2

    if mask is None:
        mask = torch.ones_like(deltas, dtype=torch.bool)
    buckets = buckets.masked_fill(first_real_position(mask), NO_PREDECESSOR_BUCKET)
    return buckets.masked_fill(~mask, PAD_BUCKET)


def deltas_from_timestamps(ts: Tensor) -> Tensor:
    """Convert absolute unix timestamps ``[B, L]`` to gaps from the previous event.

    Provided because the evaluation harness hands out absolute timestamps while
    the model contract is deltas; this is the one honest place to convert.

    The value produced for each row's first position is **meaningless** --- it
    has no predecessor to subtract. Do not try to make it meaningful by padding
    the timestamps cleverly; :func:`bucketize` overrides that position from the
    mask instead, which is robust to however the caller padded.
    """
    deltas = torch.zeros_like(ts)
    deltas[:, 1:] = ts[:, 1:] - ts[:, :-1]
    return deltas.clamp(min=0)


class TimeDeltaEncoding(nn.Module):
    """Embeds log-scale inter-event gaps into the model's hidden width."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(N_BUCKETS, d_model, padding_idx=PAD_BUCKET)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD_BUCKET].zero_()

    def forward(self, deltas: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.dropout(self.embedding(bucketize(deltas, mask)))
