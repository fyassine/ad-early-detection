"""
Dataclass configs for GELSTM training and evaluation.

`GELSTMTrainConfig` collects training-loop hyperparameters; `EvalConfig`
groups the (formerly loose) kwargs threaded through ``evaluate`` and
``encode_batch_sequences`` so they can be logged as a single bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .encoder import EncoderInit  # noqa: F401  (re-exported for config authors)


@dataclass
class GELSTMTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    rnn_type: str = "lstm"
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
    # Classifier-head normalisation: "none" (default, back-compat) or "layernorm".
    # LayerNorm sharpens the head's logits to widen the (otherwise narrow) RNN
    # probability spread. See common/VISIT_COUNT_CONFOUND.md.
    classifier_norm: str = "none"
    # Encoder arm for the reconstruction-value ablation — one of
    # ENCODER_INIT_ARMS ("pretrained_frozen" | "pretrained_finetuned" |
    # "random" | "none"). ``None`` means "not set": the arm is then derived from
    # the legacy ``freeze_encoder`` flag, so every pre-ablation config keeps its
    # exact current behaviour. See configs/encoder.py and
    # DOCS/reconstruction-value-ablation.md.
    encoder_init: Optional[str] = None


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
    # Let gradients flow through the graph encoder while embedding each visit.
    # ``False`` (default) reproduces the historical behaviour: visits are encoded
    # under ``torch.no_grad()`` + forced eval mode, so the encoder is a pure
    # feature extractor no matter what ``requires_grad`` says. Set ``True`` only
    # for the encoder-trainable ablation arms (see configs/encoder.py) — without
    # it a randomly-initialised encoder would never actually train.
    encoder_grad: bool = False
