"""Dataclass configs for TFGN training and evaluation.

`TFGNTrainConfig` collects training-loop hyperparameters; `TFGNEvalConfig`
groups the kwargs threaded through evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .encoder import EncoderInit  # noqa: F401  (re-exported for config authors)


@dataclass
class TFGNTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    use_scheduler: bool = True
    seed: int = 42
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6

    # Model architecture fields
    n_rois: int = 200
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.3
    use_time_delta: bool = True
    gvae_hidden: int = 128
    gvae_latent: int = 64
    gvae_heads: int = 2
    gvae_dropout: float = 0.3
    adjacency_k: int = 8

    # TFGN ladder knobs
    node_lstm_init: str = "random"  # "random" | "pretrained_frozen" | "pretrained_finetuned"
    node_lstm_ckpt_path: Optional[str] = None
    use_gate: bool = True
    lambda_sparse: float = 0.1
    lambda_drift: float = 0.01
    gate_rho: float = 0.15
    recon_target: str = "delta_a_topk"  # "none" | "delta_a_topk" | "delta_a_mse" | "a_last"
    lambda_recon: float = 1.0
    beta_kl: float = 1.0
    free_bits: float = 0.5
    beta_warmup_epochs: float = 5.0
    change_mask_kappa: float = 0.10
    fusion: str = "concat_residual"  # "concat_residual" | "z_only"
    readout: str = "attention"  # "mean" | "attention"
    dual_score: bool = True
    lambda_cent: float = 0.1
    tau: float = 0.05
    cohort_conditioning: str = "none"  # "none" | "adversarial"
    cohort_adv_lambda: float = 1.0  # gradient-reversal strength once warmed up
    cohort_adv_warmup_epochs: float = 5.0  # linear ramp 0 -> cohort_adv_lambda, mirrors beta_warmup_epochs
    encoder_init: Optional[str] = None  # GVAE encoder arm: "pretrained_frozen" | "pretrained_finetuned" | "random" | "none"
    gvae_ckpt_path: Optional[str] = None


@dataclass
class TFGNEvalConfig:
    use_time_delta: bool = True
    zero_time_delta: bool = False
    graph_pool: str = "mean"
    dim_filter: Optional[Any] = None
    shuffle_order: bool = False
    shuffle_rng: Optional[Any] = field(default=None, repr=False)
    threshold_mode: str = "youden"
    fixed_threshold: float = 0.5
    encoder_grad: bool = False
