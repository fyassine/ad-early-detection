---
paths:
  - "CLASSIFIER/model/**/*.py"
  - "CLASSIFIER/notebooks/**/*.ipynb"
---

# Checkpoints

## Schema (new runs save full state)

```python
torch.save({
    "model_state_dict":      ...,
    "optimizer_state_dict":  ...,
    "scheduler_state_dict":  ... or None,
    "epoch":                 int,
    "val_auc":               float,
    "best_threshold":        float,
    "rng_state":             rng.bit_generator.state if rng is not None else None,
    "torch_rng_state":       torch.get_rng_state(),
    "config":                asdict(cfg),
    "eval_config":           dict (GELSTM only — see _eval_cfg_to_dict),
}, save_path)
```

## Back-compat loading

Old checkpoints in `CLASSIFIER/checkpoints/` are bare `state_dict` files (no wrapping dict). Load with:

```python
ckpt = torch.load(path, map_location=device)
state = ckpt.get("model_state_dict", ckpt)   # works for both schemas
model.load_state_dict(state)
```

This is the only pattern that's safe for both new and legacy artifacts.

## Save paths

- New runs: `CLASSIFIER/outputs/<experiment_id>/best.pth`. The `outputs/` tree is gitignored except for `.gitkeep`.
- `CLASSIFIER/checkpoints/` is **legacy read-only**. Do not write here.

The `<experiment_id>` should match an entry in `CLASSIFIER/experiments.yaml`.

## Do not

- Do not save only `model.state_dict()` for a new run — full-state is mandatory for reproducibility.
- Do not write to `CLASSIFIER/checkpoints/`.
- Do not load checkpoints with `weights_only=False` unless you control the source.
