"""
Dataclass configs and batch contract for GEC training.

`GECTrainConfig` collects training-loop hyperparameters. `GECBatch`
documents the attributes a batch must expose (it's a contract — at
runtime PyG ``Data`` / ``Batch`` objects are passed; this dataclass
is used purely for type hints and documentation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GECTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    use_scheduler: bool = True
    seed: int = 42
    wandb_project: str = "gec-classification"
    wandb_enabled: bool = False
    threshold_mode: str = "youden"
    fixed_threshold: float = 0.5


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
