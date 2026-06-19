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
| `COMPARISON_`   | Cross-model or cross-region aggregation of saved predictions (no training) |

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
from DATA.src.splitting.load_splits import split_csv_paths
run_full_audit(split_csv_paths("downstream"))
```

Hard-fails if any subject crosses splits. Run at the head of every training notebook.

`split_csv_paths` takes one of two keys:

- `"pretrain"` — all-cohort split (healthy/ad/mci/converter). Used **only** by the GAAE-pretraining notebook.
- `"downstream"` — mci/converter-only split. Used by every other notebook (GEC, GELSTM, Long-GEC-MLP, LogReg, FDR analysis, sanity/baseline comparators, PROGNOSER).

## Imports

Use fully-qualified imports so notebooks work regardless of working directory:

```python
from CLASSIFIER.common.seeding import set_seed, make_rng
from CLASSIFIER.configs import GELSTMTrainConfig, EvalConfig
from CLASSIFIER.model.GELSTM.train import train_model
```

## Interactive prompts (mandatory in every notebook that uses these features)

### 1 — GAAE checkpoint selection (every notebook that loads a GAAE encoder)

Print the indexed list from `CHECKPOINT_SEARCH_DIRS`, prompt `int(input("Select GAAE checkpoint index: "))`. Sets `GAAE_RUN_NAME`, `GAAE_CKPT_PATH`, `GAAE_RUN_DIR`. Cell headed by `## Select GAAE Checkpoint`.

### 2 — Train vs load existing checkpoint (every training notebook)

After the config cell: scan `OUTPUT_DIR` for existing `run_summary.json` files, list them, prompt `[y=train (default), n=load existing]`. Sets `USE_EXISTING_CHECKPOINT` (bool) and `EXISTING_RUN_DIR`. If `True`, skip the training loop and go straight to test evaluation.

### 3 — Threshold selection (every notebook that derives a classification threshold)

Show both options with their OOF sens/spec/F1. **Best-F1 is option 1 / the default (Enter)**; Youden is option 2. Prompt: `[1=Best-F1 (default), 2=Youden]`. Sets `ACTIVE_THRESHOLD` and `THRESHOLD_METHOD` (`"oof_f1"` or `"oof_youden"`).

## Do not

- Do not create a notebook without a prefix.
- Do not inline `train_test_split` or `KFold` — use `common.splits` / `common.validation`.
- Do not skip the sanity audit.
- Do not put training loops inside notebook cells — call `train_model` / `train_classifier`.
- Do not omit the GAAE checkpoint selection prompt in any notebook that loads a GAAE encoder.
- Do not default the threshold selection to Youden — Best-F1 is the default (option 1 / Enter).
