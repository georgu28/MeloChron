"""Training and scoring the adoption model.

Binary classification, so the loop is ordinary: BCE, Adam, a temporal
validation slice inside train for early stopping, and scoring on the one fixed
cohort every baseline used. No negative sampling and no ranking harness — the
label is already there on every encounter.

Windows are built per batch from the resident corpus arrays (`windows.py`), not
materialised up front. Priors, when the head consumes them, are refit here on
the exact same train rows via `baselines.fit_priors`, so the model and the
`user × item` baseline see identical numbers and the comparison is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from melochron.adoption import metrics
from melochron.adoption.model import AdoptionModel
from melochron.adoption.windows import build_windows

CHECKPOINT_FORMAT = 1


@dataclass
class Corpus:
    """The three resident arrays every window is gathered from."""

    track_code: np.ndarray
    ts: np.ndarray
    user_offsets: np.ndarray


@dataclass
class Examples:
    """A set of encounters to train or score on."""

    users: np.ndarray  # user code per row
    positions: np.ndarray  # within-user position of the encounter
    candidates: np.ndarray  # track_code of the encountered track
    labels: np.ndarray  # bool adoption label
    priors: np.ndarray | None = None  # [N, 2] user-rate, item-rate; None unless used

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def subset(self, idx: np.ndarray) -> Examples:
        return Examples(
            users=self.users[idx],
            positions=self.positions[idx],
            candidates=self.candidates[idx],
            labels=self.labels[idx],
            priors=None if self.priors is None else self.priors[idx],
        )


@dataclass
class TrainConfig:
    max_len: int = 200
    batch_size: int = 512
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 5.0
    patience: int = 3
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _batch_tensors(
    corpus: Corpus,
    examples: Examples,
    rows: np.ndarray,
    max_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    items, deltas = build_windows(
        corpus.track_code,
        corpus.ts,
        corpus.user_offsets,
        examples.users[rows],
        examples.positions[rows],
        max_len,
    )
    item_ids = torch.from_numpy(items).to(device)
    time_deltas = torch.from_numpy(deltas).to(device)
    candidate_ids = torch.from_numpy(examples.candidates[rows].astype(np.int64) + 1).to(device)
    labels = torch.from_numpy(examples.labels[rows].astype(np.float32)).to(device)
    priors = None
    if examples.priors is not None:
        priors = torch.from_numpy(examples.priors[rows].astype(np.float32)).to(device)
    return item_ids, time_deltas, candidate_ids, priors, labels


def _lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


@torch.no_grad()
def predict(
    model: AdoptionModel,
    corpus: Corpus,
    examples: Examples,
    max_len: int,
    device: torch.device,
    batch_size: int = 2048,
    forward=None,
) -> np.ndarray:
    """Adoption probabilities for every row of ``examples``, in row order.

    ``forward`` overrides what is called for the forward pass -- a
    ``torch.compile`` wrapper sharing ``model``'s parameters -- while ``model``
    stays the object we set to eval and read weights from. They share tensors,
    so the compiled path scores exactly the weights the eager one holds.
    """
    model.eval()
    run = forward if forward is not None else model
    use_amp = device.type == "cuda"
    out = np.empty(len(examples), dtype=np.float32)
    for start in range(0, len(examples), batch_size):
        rows = np.arange(start, min(start + batch_size, len(examples)))
        item_ids, time_deltas, candidate_ids, priors, _ = _batch_tensors(
            corpus, examples, rows, max_len, device
        )
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = run(item_ids, time_deltas, candidate_ids, priors)
        out[rows] = torch.sigmoid(logits).float().cpu().numpy()
    return out


def _validation_split(examples: Examples, encounter_ts: np.ndarray, frac: float, rows: np.ndarray):
    """Temporal slice inside the training rows, so early stopping never looks
    forward — the same rule the baselines' shrinkage tuning follows."""
    cut = np.quantile(encounter_ts[rows], 1.0 - frac)
    is_val = encounter_ts[rows] >= cut
    return rows[~is_val], rows[is_val]


def compiled_forward(model: AdoptionModel, enable: bool):
    """A ``torch.compile`` wrapper, or the model itself when disabled.

    Compilation is the real throughput lever for this small, many-small-kernels
    model, but it needs Triton, which is absent on Windows -- so it is opt-in and
    validated on Linux/WSL. ``dynamic=True`` because the last batch of an epoch
    and the eval batch size differ, and recompiling per shape would erase the
    win it is meant to deliver.
    """
    if not enable:
        return model
    return torch.compile(model, dynamic=True)


def train(
    model: AdoptionModel,
    corpus: Corpus,
    examples: Examples,
    encounter_ts: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    validation_frac: float = 0.1,
    compile: bool = False,
    progress: bool = True,
) -> dict:
    """Fit the model, early-stopping on a temporal validation slice.

    Returns the training history and loads the best weights back into ``model``
    before returning, so the caller scores the checkpoint that validated best
    rather than the last epoch.
    """
    rng = np.random.default_rng(config.seed)
    all_rows = np.arange(len(examples))
    fit_rows, val_rows = _validation_split(examples, encounter_ts, validation_frac, all_rows)
    val = examples.subset(val_rows)
    run = compiled_forward(model, compile)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    steps_per_epoch = int(np.ceil(fit_rows.shape[0] / config.batch_size))
    total_steps = steps_per_epoch * config.epochs
    warmup = int(total_steps * config.warmup_frac)

    # Mixed precision on CUDA. The RTX 4050 gains only ~1.4x here because the
    # hand-written attention does not map cleanly onto tensor cores, but it is
    # free and the scaler guards against fp16 underflow in the gradients. The
    # real throughput lever is max_len, which the config carries.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_metric = -np.inf
    best_state = None
    since_best = 0
    step = 0

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(fit_rows)
        epoch_loss = 0.0
        seen = 0
        # One bar per epoch. `loss.item()` below already syncs each step, so the
        # postfix is free; `mininterval` throttles redraws so a nohup log gets a
        # readable trickle rather than thousands of lines. Batches/s and ETA come
        # from tqdm itself. Disabled by `progress=False` for quiet test runs.
        starts = range(0, order.shape[0], config.batch_size)
        bar = tqdm(
            starts,
            total=steps_per_epoch,
            desc=f"epoch {epoch:2d}",
            unit="batch",
            mininterval=2.0,
            leave=False,
            disable=not progress,
        )
        for start in bar:
            rows = order[start : start + config.batch_size]
            item_ids, time_deltas, candidate_ids, priors, labels = _batch_tensors(
                corpus, examples, rows, config.max_len, device
            )
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = run(item_ids, time_deltas, candidate_ids, priors)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            for group in optimizer.param_groups:
                group["lr"] = config.lr * _lr_scale(step, total_steps, warmup)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * rows.shape[0]
            seen += rows.shape[0]
            step += 1
            bar.set_postfix(loss=f"{epoch_loss / max(seen, 1):.4f}", refresh=False)
        bar.close()

        probs = predict(model, corpus, val, config.max_len, device, forward=run)
        val_score = metrics.evaluate(val.labels, probs, val.users, "val")
        pr_auc = val_score.pr_auc
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss / max(fit_rows.shape[0], 1), 5),
                "val_pr_auc": round(pr_auc, 5),
                "val_base_rate": round(val_score.base_rate, 5),
            }
        )
        print(
            f"  epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  "
            f"val PR-AUC {pr_auc:.4f} (base {val_score.base_rate:.4f})",
            flush=True,
        )

        if pr_auc > best_metric:
            best_metric = pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= config.patience:
                print(f"  early stop at epoch {epoch} (best {best_metric:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_pr_auc": round(best_metric, 5),
        "best_epoch": int(np.argmax([h["val_pr_auc"] for h in history])),
    }


def save_checkpoint(path: Path, model: AdoptionModel, config: TrainConfig, metrics_: dict) -> None:
    """Primitive-only artifact, safe to load with ``weights_only=True``.

    `train/checkpoint.py` is not reused because it is welded to the ranking
    scorer; this follows the same principle — nothing but tensors and plain
    Python — so the deployed loader never has to unpickle arbitrary objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "model_config": model.config,
            "train_config": {k: v for k, v in vars(config).items() if k != "extra"},
            "metrics": metrics_,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[AdoptionModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = dict(payload["model_config"])
    kwargs = {}
    # Text/hybrid variants need a `text_vectors` tensor to *construct* the module,
    # but the config never carries the 155 MB matrix. Its shape lives in the saved
    # buffer, so a zero placeholder of that shape rebuilds the architecture and
    # `load_state_dict` immediately fills it with the real (already pad/oov-zeroed)
    # values — the checkpoint is self-contained, no genre file needed to reload.
    if config.get("item_variant", "id") != "id":
        sd = payload["state_dict"]
        key = next(k for k in sd if k.endswith("text_vectors"))
        kwargs["text_vectors"] = torch.zeros_like(sd[key])
    model = AdoptionModel(**config, **kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
