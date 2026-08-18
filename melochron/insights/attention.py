"""Attention extraction: which past plays the model looked at.

For each history this reads the attention row at the **last** query position ---
the state that produces the next-item prediction --- and reports where its mass
landed, labelled with the track it points at, how many plays ago that was, and
how long ago in wall-clock time.

Per head, not just averaged. Heads in a two-head model reliably specialize, and
the usual split is one head pinned to the immediately-previous play and one
spread across the session. Averaging them produces a smooth curve that shows
neither behaviour, and the specialization is the more interesting finding.

Two numbers accompany every trace because they connect attention back to the
result that defines this project. ``recency_mass`` is the share sitting on the
single most recent play: on a corpus with a 0.87 repeat rate, a model that puts
nearly all of it there is behaving like a cache, and that is worth being able
to see rather than infer. ``repeat_item_mass`` is the share on positions holding
tracks the listener plays more than once in this window, separating habitual
material from one-off plays.

The masking is not optional. ``SASRec.build_attention_mask`` ORs an identity
matrix into the causal mask so that left-padded rows do not softmax over an
all-``-inf`` row and produce NaN. The cost is that every pad query position
attends to itself with weight 1.0. Those rows are not attention, they are a
numerical guard, and anything that averages them in is reporting padding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from melochron.data.vocab import Vocab
from melochron.models.sasrec import SASRec
from melochron.models.scorer import prepare_batch

DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 32


@dataclass
class AttendedItem:
    """One history position and the attention it received."""

    recency: int
    item_id: int
    label: str
    gap_s: int
    weight: float
    per_head: list[float] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "recency": self.recency,
            "item_id": self.item_id,
            "label": self.label,
            "gap_s": self.gap_s,
            "weight": round(self.weight, 5),
            "per_head": [round(w, 5) for w in self.per_head],
        }


@dataclass
class AttentionTrace:
    user_id: str
    block: int
    history_len: int
    recency_mass: float
    repeat_item_mass: float
    #: Mass held by the returned top positions, and that mass over what the same
    #: number of positions would hold if attention were flat. Reported because
    #: ``recency_mass`` alone cannot distinguish "diffuse" from "concentrated
    #: somewhere other than the last play", and those are opposite findings.
    top_mass: float = 0.0
    concentration: float = 0.0
    #: Attention-weighted mean recency per head. A head sitting near 0 is the
    #: "what did you just play" channel.
    expected_recency_per_head: list[float] = field(default_factory=list)
    top: list[AttendedItem] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "user_id": self.user_id,
            "block": self.block,
            "history_len": self.history_len,
            "recency_mass": round(self.recency_mass, 5),
            "repeat_item_mass": round(self.repeat_item_mass, 5),
            "top_mass": round(self.top_mass, 5),
            "concentration": round(self.concentration, 2),
            "expected_recency_per_head": [round(v, 3) for v in self.expected_recency_per_head],
            "top": [item.as_row() for item in self.top],
        }


def _label(vocab: Vocab | None, item_id: int) -> str:
    if vocab is None or not vocab.display or item_id >= len(vocab.display):
        return str(item_id)
    artist, track = vocab.display[item_id]
    return f"{artist} - {track}" if artist or track else str(item_id)


@torch.no_grad()
def trace(
    model: SASRec,
    histories: list[np.ndarray],
    times: list[np.ndarray],
    user_ids: list[str] | None = None,
    vocab: Vocab | None = None,
    blocks: list[int] | None = None,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str = "cpu",
    use_time: bool = True,
) -> list[AttentionTrace]:
    """Attention traces for each history, one per requested block.

    ``model`` must be in eval mode --- attention dropout would otherwise perturb
    the very weights being reported. ``checkpoint.load`` already guarantees it.
    """
    if len(histories) != len(times):
        raise ValueError(f"got {len(histories)} histories but {len(times)} time arrays")
    if model.training:
        raise ValueError("model is in training mode; attention dropout would corrupt the weights")
    if not histories:
        return []

    if user_ids is None:
        user_ids = [str(i) for i in range(len(histories))]

    out: list[AttentionTrace] = []
    use_time = use_time and model.time_encoding is not None

    for start in range(0, len(histories), batch_size):
        stop = min(start + batch_size, len(histories))
        item_ids, deltas = prepare_batch(
            histories[start:stop],
            times[start:stop],
            max_len=model.max_len,
            pad_id=model.pad_id,
            use_time=use_time,
            device=device,
        )

        _, weights = model.forward_with_attention(item_ids, deltas)
        wanted = range(len(weights)) if blocks is None else blocks

        valid = (item_ids != model.pad_id).float()
        ids_np = item_ids.cpu().numpy()
        valid_np = valid.cpu().numpy().astype(bool)
        length = ids_np.shape[1]

        for block in wanted:
            # The last query position is the one that predicts the next item.
            last = weights[block][:, :, -1, :]
            # Pad keys are already masked to zero everywhere except the identity
            # guard on the diagonal; multiplying is belt-and-braces, and keeps
            # the renormalization below honest for degenerate all-pad rows.
            last = last * valid.unsqueeze(1)
            last = last / last.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            per_head = last.cpu().numpy()
            mean_w = per_head.mean(axis=1)

            for row in range(per_head.shape[0]):
                keep = valid_np[row]
                if not keep.any():
                    continue

                positions = np.flatnonzero(keep)
                # Recency 0 is the most recent event, which left padding puts in
                # the final column.
                recency = (length - 1) - positions
                row_items = ids_np[row][positions]
                row_weights = mean_w[row][positions]

                unique, counts = np.unique(row_items, return_counts=True)
                repeated = set(unique[counts > 1].tolist())
                repeat_mass = float(
                    row_weights[np.isin(row_items, list(repeated))].sum() if repeated else 0.0
                )

                history_times = np.asarray(times[start + row], dtype=np.int64)
                latest_ts = int(history_times[-1]) if len(history_times) else 0
                truncated = history_times[-len(positions) :] if len(history_times) else None

                expected = [
                    float((per_head[row][h][positions] * recency).sum())
                    for h in range(per_head.shape[1])
                ]

                order = np.argsort(-row_weights)[:top_k]
                top: list[AttendedItem] = []
                for idx in order:
                    item_id = int(row_items[idx])
                    if truncated is not None and idx < len(truncated):
                        gap = latest_ts - int(truncated[idx])
                    else:
                        gap = 0
                    top.append(
                        AttendedItem(
                            recency=int(recency[idx]),
                            item_id=item_id,
                            label=_label(vocab, item_id),
                            gap_s=int(gap),
                            weight=float(row_weights[idx]),
                            per_head=[
                                float(per_head[row][h][positions][idx])
                                for h in range(per_head.shape[1])
                            ],
                        )
                    )

                history_len = int(keep.sum())
                top_mass = float(sum(item.weight for item in top))
                flat = len(top) / max(history_len, 1)
                out.append(
                    AttentionTrace(
                        user_id=user_ids[start + row],
                        block=int(block),
                        history_len=history_len,
                        recency_mass=float(row_weights[recency == 0].sum()),
                        repeat_item_mass=repeat_mass,
                        top_mass=top_mass,
                        concentration=top_mass / flat if flat > 0 else 0.0,
                        expected_recency_per_head=expected,
                        top=top,
                    )
                )

    return out


def summarize(traces: list[AttentionTrace]) -> dict:
    """Corpus-level view of where attention goes."""
    if not traces:
        return {"traces": 0}
    by_block: dict[int, list[AttentionTrace]] = {}
    for t in traces:
        by_block.setdefault(t.block, []).append(t)

    return {
        "traces": len(traces),
        "blocks": sorted(by_block),
        "mean_recency_mass": {
            str(block): round(float(np.mean([t.recency_mass for t in rows])), 4)
            for block, rows in sorted(by_block.items())
        },
        "mean_repeat_item_mass": {
            str(block): round(float(np.mean([t.repeat_item_mass for t in rows])), 4)
            for block, rows in sorted(by_block.items())
        },
        "mean_top_mass": {
            str(block): round(float(np.mean([t.top_mass for t in rows])), 4)
            for block, rows in sorted(by_block.items())
        },
        "mean_concentration": {
            str(block): round(float(np.mean([t.concentration for t in rows])), 2)
            for block, rows in sorted(by_block.items())
        },
        "mean_expected_recency_per_head": {
            str(block): [
                round(float(v), 3)
                for v in np.mean([t.expected_recency_per_head for t in rows], axis=0)
            ]
            for block, rows in sorted(by_block.items())
        },
    }
