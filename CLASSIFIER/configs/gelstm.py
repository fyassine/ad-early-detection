"""
Dataclass configs for GELSTM training and evaluation.

`GELSTMTrainConfig` collects training-loop hyperparameters; `EvalConfig`
groups the (formerly loose) kwargs threaded through ``evaluate`` and
``encode_batch_sequences`` so they can be logged as a single bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GELSTMTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 16
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    use_scheduler: bool = True
    seed: int = 42
    threshold_mode: str = "youden"
    fixed_threshold: float = 0.5
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6


@dataclass
class EvalConfig:
    use_time_delta: bool = True
    zero_time_delta: bool = False
    graph_pool: str = "mean"
    dim_filter: Optional[Any] = None
    shuffle_order: bool = False
    shuffle_rng: Optional[Any] = field(default=None, repr=False)
    threshold_mode: str = "youden"
    fixed_threshold: float = 0.5
