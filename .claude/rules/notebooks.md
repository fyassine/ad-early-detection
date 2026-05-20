---
paths:
  - "CLASSIFIER/notebooks/**/*.ipynb"
---

# Notebooks

## Prefix taxonomy (mandatory)

Every notebook filename in `CLASSIFIER/notebooks/` MUST start with exactly one of:

| Prefix          | Meaning                                          |
| --------------- | ------------------------------------------------ |
| `BASELINE_`     | First-visit only, no follow-up                   |
| `LONGITUDINAL_` | Multi-visit / full-trajectory                    |
| `STATIC_`       | Per-scan / cross-sectional                       |
| `SANITY_`       | Sanity checks and ablations                      |

Never create a notebook without one of these prefixes.

## Required structure

1. First code cell — seeding (see `seeding.md`).
2. Path setup cell — `sys.path` injection for `CLASSIFIER/`.
3. Split-hygiene audit — `run_full_audit(...)` from `CLASSIFIER.common.sanity`. Already injected by `dev/patch_v2_notebooks.py`.
4. Data loading + splits via `make_splits` from `common.splits` (never inline `train_test_split` or `KFold`).
5. Model instantiation + training via `model/<family>/train.py` entrypoints.
6. Save artifacts under `CLASSIFIER/outputs/<experiment_id>/`.

## Single source of truth for splits

```python
from CLASSIFIER.common import make_splits
idx = make_splits(subject_ids, labels, seed=SEED, val_frac=0.15, test_frac=0.15)
```

For cross-validation loops, use `CLASSIFIER.common.validation.run_kfold_cv`. Do not write a new `StratifiedGroupKFold` loop in a notebook.

## Sanity audit

```python
from CLASSIFIER.common.sanity import run_full_audit
run_full_audit({
    "train": str(METADATA_DIR / "splits_gaae" / "train.csv"),
    "val":   str(METADATA_DIR / "splits_gaae" / "val.csv"),
    "test":  str(METADATA_DIR / "splits_gaae" / "test.csv"),
})
```

Hard-fails if any subject crosses splits. Run at the head of every training notebook.

## Imports

Use fully-qualified imports so notebooks work regardless of working directory:

```python
from CLASSIFIER.common.seeding import set_seed, make_rng
from CLASSIFIER.configs import GELSTMTrainConfig, EvalConfig
from CLASSIFIER.model.GELSTM.train import train_model
```

## Do not

- Do not create a notebook without a prefix.
- Do not inline `train_test_split` or `KFold` — use `common.splits` / `common.validation`.
- Do not skip the sanity audit.
- Do not put training loops inside notebook cells — call `train_model` / `train_classifier`.
