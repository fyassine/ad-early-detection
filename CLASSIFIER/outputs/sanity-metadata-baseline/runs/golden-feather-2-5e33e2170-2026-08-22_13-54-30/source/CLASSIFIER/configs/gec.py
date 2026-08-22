"""
Dataclass configs and batch contract for GEC training.

`GECTrainConfig` collects training-loop hyperparameters. `GECBatch`
documents the attributes a batch must expose (it's a contract — at
runtime PyG ``Data`` / ``Batch`` objects are passed; this dataclass
is used purely for type hints and documentation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .encoder import EncoderInit  # noqa: F401  (re-exported for config authors)


@dataclass
class GECTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    use_scheduler: bool = True
    seed: int = 42
    wandb_project: str = "ad-early-detection"
    wandb_enabled: bool = True
    threshold_mode: str = "youden"
    fixed_threshold: float = 0.5
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6
    # Encoder arm for the reconstruction-value ablation (see configs/encoder.py).
    # GEC pre-encodes each visit once, offline, before training the MLP — there is
    # no training-time forward pass through the encoder — so only the two arms
    # that don't require encoder gradients are meaningful here:
    # "pretrained_frozen" (default via None, today's behaviour) or "none". The
    # trainable arms ("pretrained_finetuned" / "random") raise in GECAdapter;
    # use the GELSTM adapter for those. See DOCS/reconstruction-value-ablation.md.
    encoder_init: str | None = None


@dataclass
class GECBatch:
    """Attribute contract for batches consumed by GEC training.

    Real call sites pass PyG ``Batch`` objects; this dataclass exists
    purely to document the expected fields.
    """

    x: Any
    edge_index: Any
    batch: Any
    is_converter: Any
    patient_age: Any
    patient_sex: Any
