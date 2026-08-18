"""Versioned model artifacts.

Phase 5 loads exactly what this writes, so the checkpoint carries everything
needed to reconstruct a working scorer without consulting the training code:
weights, the vocabulary, the model config, and the metrics at save time.

Everything stored is a tensor or a plain Python primitive, so ``torch.load``
runs with ``weights_only=True``. That matters because a checkpoint is a pickle,
and a pickle is arbitrary code execution on load. The deployed service will one
day load an artifact from object storage, and "it is our own file" is not a
property the loader can verify. Keeping the format primitive-only means the
safe loader is sufficient, rather than being a setting someone has to remember
to turn off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from melochron.data.vocab import Vocab
from melochron.models.heads import TiedItemScorer
from melochron.models.sasrec import SASRec
from melochron.models.scorer import SASRecScorer, build_scorer

FORMAT_VERSION = 1


@dataclass
class Artifact:
    model: SASRec
    head: TiedItemScorer
    scorer: SASRecScorer
    vocab: Vocab
    config: dict
    metrics: dict = field(default_factory=dict)


def _vocab_to_payload(vocab: Vocab) -> dict:
    return {
        "id_to_key": list(vocab.id_to_key),
        "counts": torch.as_tensor(np.asarray(vocab.counts), dtype=torch.long),
        "display": [list(pair) for pair in vocab.display] if vocab.display else [],
    }


def _vocab_from_payload(payload: dict) -> Vocab:
    id_to_key = list(payload["id_to_key"])
    return Vocab(
        key_to_id={k: i for i, k in enumerate(id_to_key)},
        id_to_key=id_to_key,
        # .cpu() before .numpy(): torch.load's map_location moves every tensor
        # in the payload, including this one, so on a GPU load it arrives on
        # the device and numpy() raises.
        counts=payload["counts"].cpu().numpy(),
        display=[tuple(p) for p in payload.get("display", [])],
    )


def save(
    path: str | Path,
    model: SASRec,
    head: TiedItemScorer,
    vocab: Vocab,
    config: dict,
    metrics: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model": model.state_dict(),
            # The head's only own parameter is its bias; the item table is a
            # live reference to the model's, so saving both would store the
            # embedding twice and, worse, allow them to diverge on load.
            "head": {k: v for k, v in head.state_dict().items() if not k.startswith("items.")},
            "vocab": _vocab_to_payload(vocab),
            "config": dict(config),
            "metrics": dict(metrics or {}),
        },
        path,
    )
    return path


def load(path: str | Path, device: torch.device | str = "cpu", name: str = "sasrec") -> Artifact:
    """Reconstruct a scorer from an artifact, ready to evaluate or serve."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version} but this build expects {FORMAT_VERSION}"
        )

    config = dict(payload["config"])
    vocab = _vocab_from_payload(payload["vocab"])

    # The text matrix rides inside ProjectedTextEmbedding as a buffer, so it
    # arrives with the state dict; build_scorer only needs its shape to
    # construct the right module, and load_state_dict fills in the values.
    text_vectors = None
    if config.get("variant", "id") != "id":
        # The buffer sits at items.text_vectors for the text variants and at
        # items.text.text_vectors for the hybrid, which nests the text module.
        # Searching by suffix keeps this working if the nesting changes again.
        keys = [k for k in payload["model"] if k.endswith("text_vectors")]
        if not keys:
            raise ValueError(
                f"config says variant={config['variant']!r} but the checkpoint contains no "
                "*text_vectors buffer; the artifact and its config disagree"
            )
        text_vectors = torch.zeros_like(payload["model"][keys[0]])

    model, head, scorer = build_scorer(
        n_items=len(vocab),
        device=device,
        name=name,
        variant=config.get("variant", "id"),
        text_vectors=text_vectors,
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 2),
        n_blocks=config.get("n_blocks", 2),
        max_len=config.get("max_len", 200),
        dropout=config.get("dropout", 0.2),
        use_time=config.get("use_time", True),
    )

    # Non-strict, then verified. A checkpoint written before a new optional
    # parameter existed is still loadable, but silently accepting *any* missing
    # weight would let a genuinely incompatible artifact through and show up
    # only as bad metrics. So the gap is checked against a known-safe list.
    OPTIONAL_SINCE_V1 = {"items.log_scale"}
    incompatible = model.load_state_dict(payload["model"], strict=False)
    unexpected_missing = set(incompatible.missing_keys) - OPTIONAL_SINCE_V1
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"checkpoint does not match this model: missing {sorted(unexpected_missing)}, "
            f"unexpected {sorted(incompatible.unexpected_keys)}"
        )
    head.load_state_dict(payload["head"], strict=False)
    model.eval()
    head.eval()

    return Artifact(
        model=model,
        head=head,
        scorer=scorer,
        vocab=vocab,
        config=config,
        metrics=dict(payload.get("metrics", {})),
    )
