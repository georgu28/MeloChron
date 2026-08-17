"""The training loop.

Model selection uses a validation cut taken *inside* the training period, so
the test period is never consulted while choosing a checkpoint. Early stopping
on test data is one of the quietest ways to leak, and it is invisible in the
final numbers: the model looks like it generalizes because it was selected for
looking like it generalizes.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from melochron.data.sessions import Sequences
from melochron.data.vocab import FIRST_ITEM_ID, Vocab
from melochron.eval import protocol
from melochron.models.heads import TiedItemScorer, sample_negatives
from melochron.models.sasrec import SASRec
from melochron.models.scorer import SASRecScorer
from melochron.models.time_encoding import deltas_from_timestamps
from melochron.train import checkpoint as ckpt
from melochron.train.dataset import Batch, make_loader
from melochron.train.losses import get_loss


@dataclass
class TrainConfig:
    variant: str = "id"  # id | text_frozen | text_finetuned
    d_model: int = 128
    n_heads: int = 2
    n_blocks: int = 2
    max_len: int = 200
    dropout: float = 0.2
    use_time: bool = True

    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 5.0
    n_negatives: int = 512
    loss: str = "sampled_softmax"
    popularity_negatives: bool = True
    #: One negative set per batch instead of one per position. Required at any
    #: realistic batch x seq_len on a 6 GB card; see shared_negative_logits.
    shared_negatives: bool = True

    stride: int | None = None
    patience: int = 3
    seed: int = 0
    eval_batch_size: int = 128
    max_val_instances: int = 5000


@dataclass
class TrainState:
    best_metric: float = -1.0
    best_epoch: int = -1
    history: list[dict] = field(default_factory=list)


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base LR."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / (total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class Trainer:
    def __init__(
        self,
        model: SASRec,
        head: TiedItemScorer,
        scorer: SASRecScorer,
        vocab: Vocab,
        config: TrainConfig,
        device: torch.device | str = "cuda",
        train_counts: np.ndarray | None = None,
    ):
        self.model = model.to(device)
        self.head = head.to(device)
        self.scorer = scorer
        self.vocab = vocab
        self.cfg = config
        self.device = torch.device(device)
        self.loss_fn = get_loss(config.loss)
        self.state = TrainState()

        # Negatives drawn by popularity, not uniformly. Uniform negatives from
        # a long-tailed catalog are almost always obscure, so the model is only
        # ever asked to beat items it will never be ranked against at eval.
        self.counts = (
            torch.as_tensor(train_counts, dtype=torch.float, device=self.device)
            if (config.popularity_negatives and train_counts is not None)
            else None
        )

        params = list(model.parameters()) + [p for p in head.parameters() if p.requires_grad]
        # De-duplicate: the head holds a live reference to the model's item
        # table, so its parameters are already in the model's list. Handing the
        # same tensor to AdamW twice applies weight decay twice.
        seen, unique = set(), []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        self.optimizer = torch.optim.AdamW(
            unique, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98)
        )
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

    def _forward_loss(self, batch: Batch) -> Tensor:
        deltas = deltas_from_timestamps(batch.timestamps) if self.cfg.use_time else None
        hidden = self.model(batch.item_ids, deltas)  # [B, L, D]

        # Keep only positions that carry supervision. On short histories most
        # of a left-padded batch is padding, and scoring 512 negatives against
        # padded positions is pure waste.
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = batch.targets.reshape(-1)
        keep = batch.mask.reshape(-1).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            return hidden.sum() * 0.0

        sel_hidden = flat_hidden[keep]
        sel_targets = flat_targets[keep]

        if self.cfg.shared_negatives:
            # One negative set for the whole batch: [K] rather than [B, K].
            # Per-row negatives need a [B, K, D] embedding gather, and B here is
            # batch x seq_len because every position is supervised. At
            # 20,000 x 512 x 128 that is ~5.2 GB for a single intermediate,
            # which OOMs a 6 GB card. Sharing needs ~41 MB.
            negatives = sample_negatives(
                n_items=len(self.vocab),
                shape=(self.cfg.n_negatives,),
                device=self.device,
                counts=self.counts,
                first_item_id=FIRST_ITEM_ID,
                generator=self.generator,
            )
            positive, negative = self.head.shared_negative_logits(
                sel_hidden, sel_targets, negatives
            )
        else:
            negatives = sample_negatives(
                n_items=len(self.vocab),
                shape=(sel_targets.shape[0], self.cfg.n_negatives),
                device=self.device,
                counts=self.counts,
                first_item_id=FIRST_ITEM_ID,
                generator=self.generator,
            )
            positive, negative = self.head.sampled_logits(sel_hidden, sel_targets, negatives)

        ones = torch.ones(sel_targets.shape[0], dtype=torch.bool, device=self.device)
        return self.loss_fn(positive, negative, ones)

    def train_epoch(self, loader, scheduler_state: dict) -> float:
        self.model.train()
        self.head.train()

        total, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)

            lr_scale = cosine_with_warmup(
                scheduler_state["step"], scheduler_state["total"], scheduler_state["warmup"]
            )
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.lr * lr_scale

            loss = self._forward_loss(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]], self.cfg.grad_clip
            )
            self.optimizer.step()

            total += float(loss.detach())
            n_batches += 1
            scheduler_state["step"] += 1

        return total / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, instances: protocol.EvalInstances) -> dict:
        self.model.eval()
        self.head.eval()
        result = protocol.evaluate(self.scorer, instances, batch_size=self.cfg.eval_batch_size)
        return result.overall

    def fit(
        self,
        train_seqs: Sequences,
        val_instances: protocol.EvalInstances,
        out_dir: str | Path,
        select_on: str = "NDCG@10",
    ) -> TrainState:
        out_dir = Path(out_dir)
        loader = make_loader(
            train_seqs,
            max_len=self.cfg.max_len,
            batch_size=self.cfg.batch_size,
            stride=self.cfg.stride,
            seed=self.cfg.seed,
        )

        steps_per_epoch = max(len(loader), 1)
        scheduler_state = {
            "step": 0,
            "total": steps_per_epoch * self.cfg.epochs,
            "warmup": int(steps_per_epoch * self.cfg.epochs * self.cfg.warmup_frac),
        }
        print(f"training windows {len(loader.dataset):,} | steps/epoch {steps_per_epoch:,}")

        since_improved = 0
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            loss = self.train_epoch(loader, scheduler_state)
            metrics = self.validate(val_instances)
            score = metrics.get(select_on, float("nan"))

            row = {
                "epoch": epoch,
                "loss": round(loss, 5),
                "seconds": round(time.time() - t0, 1),
                **{k: round(v, 5) for k, v in metrics.items()},
            }
            self.state.history.append(row)
            print(
                f"epoch {epoch:>3}  loss {loss:.4f}  val {select_on} {score:.4f}  "
                f"({time.time() - t0:.0f}s)"
            )

            if score > self.state.best_metric:
                self.state.best_metric = score
                self.state.best_epoch = epoch
                since_improved = 0
                ckpt.save(
                    out_dir / "best.pt",
                    self.model,
                    self.head,
                    self.vocab,
                    config=asdict(self.cfg),
                    metrics={"val": metrics, "epoch": epoch, "select_on": select_on},
                )
            else:
                since_improved += 1
                if since_improved >= self.cfg.patience:
                    print(
                        f"early stop: no {select_on} improvement in {self.cfg.patience} epochs "
                        f"(best {self.state.best_metric:.4f} @ epoch {self.state.best_epoch})"
                    )
                    break

        return self.state
