"""SASRec-style sequential encoder, built from scratch.

A causal transformer over a user's listening sequence: at every position it
produces a representation that has seen that event and everything before it,
never anything after. Position ``t``'s output is the model's state for
predicting event ``t + 1``.

The attention is written out by hand --- explicit projections, explicit scaled
dot product, explicit masking --- rather than delegated to
``nn.MultiheadAttention`` or ``F.scaled_dot_product_attention``. That is the
point of the exercise, and it is also what makes the causal-masking test
meaningful: a test that a library kernel masks correctly tests the library.

Pre-norm residual blocks, following Xiong et al. (2020) rather than the
post-norm of the original SASRec paper. Pre-norm trains without a warmup
schedule, which matters on a single laptop GPU where a diverged run is an hour
that cannot be spent twice.

Sequences are **left-padded**: index ``-1`` is always the most recent event, so
``encode(...)[:, -1]`` is the next-item prediction state regardless of how much
history a user has.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from melochron.models.item_repr import IdEmbedding, ItemRepresentation
from melochron.models.time_encoding import TimeDeltaEncoding

#: Must match ``melochron.data.vocab.PAD_ID``. Injected as a constructor
#: argument rather than imported, so the model package stays free of the data
#: package's pandas dependency.
DEFAULT_PAD_ID = 0


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, computed explicitly."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, x: Tensor, attn_mask: Tensor, need_weights: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``x`` is ``[B, L, D]``; ``attn_mask`` is ``[B, 1, L, L]``, True where
        a query position is allowed to see a key position.

        With ``need_weights``, also returns the ``[B, n_heads, L, L]`` attention
        distribution --- the same tensor the output is computed from, not a
        recomputation. Phase 6 reads it; training never asks for it, so the
        default keeps the single-tensor return the rest of the model expects.
        """
        b, length, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~attn_mask, float("-inf"))

        weights = self.attn_dropout(torch.softmax(scores, dim=-1))

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, length, -1)
        out = self.resid_dropout(self.out_proj(out))
        return (out, weights) if need_weights else out


class FeedForward(nn.Module):
    """Position-wise feed-forward network, 4x expansion."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(
        self, x: Tensor, attn_mask: Tensor, need_weights: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if need_weights:
            delta, weights = self.attn(self.norm_attn(x), attn_mask, need_weights=True)
            x = x + delta
            return x + self.ffn(self.norm_ffn(x)), weights

        x = x + self.attn(self.norm_attn(x), attn_mask)
        return x + self.ffn(self.norm_ffn(x))


class SASRec(nn.Module):
    """Causal transformer encoder over item sequences.

    ``n_items`` is the full embedding-table size *including* the reserved PAD
    and OOV slots, i.e. ``len(vocab)``.
    """

    def __init__(
        self,
        n_items: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
        use_time: bool = True,
        pad_id: int = DEFAULT_PAD_ID,
        item_repr: ItemRepresentation | None = None,
    ):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.use_time = use_time

        self.position_embedding = nn.Embedding(max_len, d_model)
        self.time_encoding = TimeDeltaEncoding(d_model, dropout) if use_time else None

        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_blocks))
        self.norm_out = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

        # Built after _init_weights so a supplied representation --- in
        # particular a frozen text matrix --- is not re-randomized by it.
        if item_repr is None:
            item_repr = IdEmbedding(n_items, d_model, pad_id=pad_id)
        elif item_repr.n_items != n_items or item_repr.d_model != d_model:
            raise ValueError(
                f"item_repr is [{item_repr.n_items}, {item_repr.d_model}] but the model "
                f"expects [{n_items}, {d_model}]"
            )
        self.items = item_repr

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def build_attention_mask(self, item_ids: Tensor) -> Tensor:
        """``[B, 1, L, L]``, True where a query may attend to a key.

        Combines the causal constraint with padding: a position may see itself
        and earlier positions, and may never see a pad slot.
        """
        b, length = item_ids.shape
        device = item_ids.device

        causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        key_valid = (item_ids != self.pad_id).view(b, 1, 1, length)
        allowed = causal.view(1, 1, length, length) & key_valid

        # A left-padded sequence has leading query positions whose only visible
        # key is their own pad slot, which the line above just masked out. Every
        # key masked means a softmax over all -inf, which is NaN, and one NaN
        # propagates through the residual stream into every later position. So
        # let every position see itself. Pad positions' outputs are garbage
        # either way and are zeroed on the way out.
        eye = torch.eye(length, dtype=torch.bool, device=device).view(1, 1, length, length)
        return allowed | eye

    def _encode(
        self, item_ids: Tensor, time_deltas: Tensor | None, need_weights: bool
    ) -> tuple[Tensor, list[Tensor]]:
        """The single implementation behind both public encode paths.

        ``forward`` and ``forward_with_attention`` differ only in what they
        return. Keeping one body means the attention a visualization reports is
        provably the attention that produced the hidden states --- two copies of
        this preamble would be free to drift apart without failing any test.
        """
        if item_ids.dim() != 2:
            raise ValueError(f"expected item_ids of shape [B, L], got {tuple(item_ids.shape)}")
        _, length = item_ids.shape
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")

        mask = item_ids != self.pad_id

        x = self.items(item_ids) * math.sqrt(self.d_model)

        # Positions are numbered from the right so that "most recent" is a fixed
        # position id across sequences of differing history length. Numbering
        # from the left would make the same recent event look different to the
        # model depending on how much padding preceded it.
        positions = torch.arange(length - 1, -1, -1, device=item_ids.device)
        x = x + self.position_embedding(positions).unsqueeze(0)

        if self.time_encoding is not None and time_deltas is not None:
            x = x + self.time_encoding(time_deltas, mask)

        x = self.input_dropout(x)
        x = x * mask.unsqueeze(-1)

        attn_mask = self.build_attention_mask(item_ids)
        weights: list[Tensor] = []
        for block in self.blocks:
            if need_weights:
                x, block_weights = block(x, attn_mask, need_weights=True)
                weights.append(block_weights)
            else:
                x = block(x, attn_mask)

        return self.norm_out(x) * mask.unsqueeze(-1), weights

    def forward(self, item_ids: Tensor, time_deltas: Tensor | None = None) -> Tensor:
        """Encode a batch of sequences.

        ``item_ids`` is ``[B, L]`` of vocab ids, left-padded with ``pad_id``.
        ``time_deltas`` is ``[B, L]`` of gaps in seconds from the previous
        event; pass ``None`` for the position-only ablation.

        Returns ``[B, L, d_model]``, zeroed at padded positions.
        """
        hidden, _ = self._encode(item_ids, time_deltas, need_weights=False)
        return hidden

    def forward_with_attention(
        self, item_ids: Tensor, time_deltas: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        """``forward``, plus one ``[B, n_heads, L, L]`` attention tensor per block.

        The hidden states are bit-identical to what ``forward`` returns for the
        same input, because both go through ``_encode``.

        Two properties of the weights are easy to misread and are the caller's
        problem to handle. Rows are a distribution over *keys*, so they sum to
        one along the last axis. And every query position attends to itself with
        weight 1.0 when it is a pad slot --- ``build_attention_mask`` ORs in the
        identity to keep the softmax from producing NaN --- so pad rows carry no
        information and must be dropped, not averaged in.
        """
        return self._encode(item_ids, time_deltas, need_weights=True)

    def encode_last(self, item_ids: Tensor, time_deltas: Tensor | None = None) -> Tensor:
        """The ``[B, d_model]`` state used to predict the next item.

        Correct only because sequences are left-padded, which is why that is a
        stated contract and not an implementation detail.
        """
        return self.forward(item_ids, time_deltas)[:, -1]
