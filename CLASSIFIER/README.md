# CLASSIFIER

Research package for AD early-detection classifiers (GAAE, GEC, GELSTM).

## Layout

```
CLASSIFIER/
  common/
    seeding.py        # set_seed, make_rng, make_torch_generator, seed_worker
    splits.py         # make_splits — central train/val/test partitioning
    utils.py          # generic utils (set_seed re-exported for back-compat)
    sanity.py         # run_full_audit (split-hygiene check)
    dataset.py, validation.py
  configs/
    gelstm.py         # GELSTMTrainConfig, EvalConfig
    gec.py            # GECTrainConfig, GECBatch (attribute contract)
  model/
    GAAE/ GEC/ GELSTM/ utils/   # model families + their train/eval modules
  notebooks/          # experiments — prefix-tagged (see below)
  outputs/            # all new run artifacts (gitignored, .gitkeep tracked)
  checkpoints/        # legacy artifacts (back-compat only — do not write here)
  dev/                # one-off migration tools (e.g. patch_v2_notebooks.py)
  tests/              # pytest suite
  experiments.yaml    # registry of major runs
```

## Naming convention

Every notebook in `notebooks/` MUST start with exactly one of these prefixes:

| Prefix          | Meaning                                                      |
| --------------- | ------------------------------------------------------------ |
| `BASELINE_`     | Baseline only (first visit, no follow-up)                    |
| `LONGITUDINAL_` | Multi-visit / full-trajectory experiments                    |
| `STATIC_`       | Per-scan / cross-sectional (each scan an independent sample) |
| `SANITY_`       | Sanity checks and ablations                                  |

Mirror this in run names and config tags.

## Reproducibility

All training notebooks start with the seeding cell:

```python
from CLASSIFIER.common.seeding import (
    set_seed, make_rng, make_torch_generator, seed_worker,
)
SEED = 42
set_seed(SEED)
rng = make_rng(SEED)
torch_gen = make_torch_generator(SEED)
```

**RNG injection contract** — pass an explicit RNG into anything that shuffles:

- `make_batches(items, batch_size, shuffle=True, rng=rng)`
- `DataLoader(..., generator=torch_gen, worker_init_fn=seed_worker)`
  (or use `model.GEC.train.build_loader(...)` which wires both).

Calling `make_batches` with `shuffle=True` and no `rng` emits a
`DeprecationWarning` and falls back to global `np.random` — acceptable only
for transitional use.

### Note on legacy seeds

The new `make_rng` uses `np.random.default_rng` (PCG64), which produces a
different shuffle sequence than legacy `np.random.permutation` for the same
seed. Reproducibility is preserved going forward but historical seed values
will not reproduce pre-cleanup runs byte-for-byte.

## Configs

Training hyperparameters are dataclasses (see `configs/`):

```python
from CLASSIFIER.configs import GELSTMTrainConfig, EvalConfig, GECTrainConfig

cfg = GELSTMTrainConfig(epochs=100, lr=1e-3, batch_size=16, seed=SEED)
eval_cfg = EvalConfig(use_time_delta=True, graph_pool="mean", threshold_mode="youden")
```

## Splits

All notebooks should call `make_splits` from `common.splits` for any
train/val/test partitioning so partitions are consistent and reproducible:

```python
from CLASSIFIER.common import make_splits
idx = make_splits(subject_ids, labels, seed=SEED, val_frac=0.15, test_frac=0.15)
# idx['train'], idx['val'], idx['test']
```

## Training entrypoints

### GELSTM
- `model.GELSTM.train.train_model(model, train_batches, val_batches, cfg, eval_cfg, device, *, rng, save_path)`
- Lower-level: `train_epoch`, `evaluate`, `make_batches` (back-compat with legacy positional kwargs)

### GEC
- `model.GEC.train.train_classifier(model, train_loader, val_loader, optimizer, device, pos_weight, cfg, *, wandb_run=None, rng=None, model_save_path=None)`
- `evaluate_classifier(..., threshold=val_best_threshold)` — explicit threshold required to prevent leakage.
- `build_loader(dataset, batch_size, shuffle, seed, ...)` — DataLoader with seeded generator + worker init.

## Checkpoint schema

New runs save a full-state checkpoint:

```python
{
    "model_state_dict":      ...,
    "optimizer_state_dict":  ...,
    "scheduler_state_dict":  ... or None,
    "epoch":                 int,
    "val_auc":               float,
    "best_threshold":        float,
    "rng_state":             ... or None,
    "torch_rng_state":       torch.ByteTensor,
    "config":                asdict(GELSTMTrainConfig | GECTrainConfig),
    "eval_config":           dict (GELSTM only),
}
```

Loaders use the legacy pattern `ckpt.get("model_state_dict", ckpt)` so they
accept both new full-state checkpoints and old weights-only files in
`checkpoints/`.

## Outputs vs checkpoints/

- **`outputs/`** — write all new run artifacts here (`outputs/<experiment_id>/...`). Gitignored.
- **`checkpoints/`** — kept for back-compat with pre-cleanup artifacts. Do not write new runs here.

## Notebook index

| Notebook                                                          | Task              |
| ----------------------------------------------------------------- | ----------------- |
| `BASELINE_MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb`             | Model comparison  |
| `LONGITUDINAL_GELSTM_DELCODE_WHOLE_BRAIN.ipynb`                   | Trajectory        |
| `LONGITUDINAL_GELSTM_FDR_FILTERED_DELCODE_WHOLE_BRAIN.ipynb`      | Trajectory + FDR  |
| `LONGITUDINAL_GELSTM_FIRST_N_DELCODE_WHOLE_BRAIN.ipynb`           | Early detection   |
| `LONGITUDINAL_GEC_MLP_DELCODE.ipynb`                              | Trajectory (MLP)  |
| `STATIC_GAAE_LOGREG_DELCODE_WHOLE_BRAIN.ipynb`                    | Static per-scan   |
| `SANITY_SPLIT_HYGIENE_DELCODE.ipynb`                              | Split audit       |
| `SANITY_BASELINE_METADATA_TIME.ipynb`                             | Metadata floor    |
| `SANITY_LONGITUDINAL_GELSTM.ipynb`                                | LSTM ablations    |

## How to run

```bash
# Tests
pytest CLASSIFIER/tests/

# Apply notebook patcher (idempotent — injects seeding/sanity/framing cells)
python CLASSIFIER/dev/patch_v2_notebooks.py
```

## W&B logging (opt-in)

`train_classifier` no longer calls `wandb.init` internally. Pass a run:

```python
import wandb
run = wandb.init(project=cfg.wandb_project, config=asdict(cfg)) if cfg.wandb_enabled else None
train_classifier(..., cfg=cfg, wandb_run=run)
```

## Import convention

Prefer fully-qualified imports from notebooks so behavior is invariant to
working directory:

```python
from CLASSIFIER.common.seeding import set_seed, make_rng
from CLASSIFIER.configs import GELSTMTrainConfig, EvalConfig
```
