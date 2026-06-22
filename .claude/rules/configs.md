---
paths:
  - "CLASSIFIER/configs/**/*.py"
  - "CLASSIFIER/model/**/*.py"
  - "CLASSIFIER/notebooks/**/*.ipynb"
---

# Configs — dataclass hyperparameters

## Principles

- New hyperparameters go in the relevant `@dataclass` in `CLASSIFIER/configs/`. Never thread them through as loose function arguments.
- Always provide a default so existing call sites don't break.
- For mutable defaults (lists, dicts, factory objects), use `field(default_factory=...)` — never bare `[]` or `{}`.
- Configs are serialized into checkpoints via `dataclasses.asdict(cfg)`. Keep fields JSON-friendly (no lambdas, no live RNG objects). If a field must hold a non-serializable value, exclude it manually when saving (see `_eval_cfg_to_dict` in `model/GELSTM/train.py`).

## Existing configs

- `CLASSIFIER.configs.GELSTMTrainConfig` — epochs, lr, batch_size, grad_clip, early_stopping_patience, use_scheduler, seed, threshold_mode, fixed_threshold.
- `CLASSIFIER.configs.EvalConfig` — use_time_delta, zero_time_delta, graph_pool, dim_filter, shuffle_order, shuffle_rng, threshold_mode, fixed_threshold. Groups the kwargs threaded through `evaluate` and `encode_batch_sequences`.
- `CLASSIFIER.configs.GECTrainConfig` — same hyperparameter pattern plus `wandb_project`, `wandb_enabled`.
- `CLASSIFIER.configs.GECBatch` — **documentation only**. Documents the attributes a PyG `Batch` must expose (`x, edge_index, batch, is_converter, patient_age, patient_sex`). Never instantiate `GECBatch` at runtime; real call sites pass PyG `Batch` objects. The dataclass exists purely as a type-hint contract.

## Extending a config

```python
@dataclass
class GELSTMTrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    # New field — add here, with a default, never as a loose kwarg in train_model:
    weight_decay: float = 0.0
```

Then the training function reads it from `cfg.weight_decay` rather than accepting it as a separate argument.

## Do not

- Do not add new function arguments for hyperparameters — extend the dataclass.
- Do not use `dataclass(frozen=True)` for training configs unless you also handle the fact that downstream code may need to override fields per fold.
- Do not instantiate `GECBatch` — it's a docs contract for PyG `Batch`.
