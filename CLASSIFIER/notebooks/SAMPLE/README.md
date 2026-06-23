# SAMPLE — model-agnostic workflow templates

Canonical, non-running **templates** that document a pipeline once, in full, so every
model-specific notebook conforms to the same shape instead of drifting independently.

| Template | Pipeline | Conforming notebooks |
| --- | --- | --- |
| [SAMPLE_LONGITUDINAL_TEMPLATE.ipynb](SAMPLE_LONGITUDINAL_TEMPLATE.ipynb) | MCI→converter longitudinal classification | GELSTM, GEC-MLP, … |
| [SAMPLE_STATIC_TEMPLATE.ipynb](SAMPLE_STATIC_TEMPLATE.ipynb) | Whole-brain reconstruction-autoencoder pretraining | GAAE, VGAE |

## Longitudinal template

### Instructions this was built from

1. **Move notebook code into scripts** under the model/common packages so notebooks
   stop being a place where subtle mistakes hide.
2. **Create a model-agnostic SAMPLE notebook** showing every step seed → test, plus the
   post-test cells (ROC, early-detection curve, conversion trajectories). Doesn't need to
   run — it shows the steps. (Dashboard-deploy cell excluded.)
3. **Lift the repeated blocks into `common/`** — explicitly the CV loop, and also the other
   parts that repeat across the GEC/GELSTM notebooks.
4. **Uppercase all cross-cell notebook globals** to match the constant style.

### What we did

**Template** — [SAMPLE_LONGITUDINAL_TEMPLATE.ipynb](SAMPLE_LONGITUDINAL_TEMPLATE.ipynb).
All model-specific work is funneled through six `raise NotImplementedError` **hooks**
(`build_model`, `prepare_data`, `train_fold`, `eval_split`, `truncate_to_n_visits`,
`per_visit_probs`) plus the `common.crossval.Bundle` container. Every other cell is SHARED.
Cross-cell globals are UPPERCASE (`CV_BUNDLE`, `BEST_MODEL_STATE`, `MODEL_CONFIG`, `BEST_FOLD`,
`OOF_PROBS`, `TEST_METRICS`, `TRAJ_DF`, …); `run_name`/`run_dir` merged into the existing
`RUN_NAME`/`RUN_DIR` papermill params. `rng`/`torch_gen` and loop-locals stay lowercase.

**New shared modules** (`CLASSIFIER/common/`, each with tests in `CLASSIFIER/tests/`):

| Module | Public API |
| --- | --- |
| `crossval.py` | `Bundle`, `CVResult`, `run_kfold_cv` (hook-driven, W&B-free via `log_fn`), `summarize_cv` |
| `thresholds.py` | `youden_threshold`, `best_f1_threshold`, `oof_threshold_metrics`, `select_oof_threshold` (OOF-only; Best-F1 default) |
| `plots.py` | `plot_oof_test_roc`, `plot_conversion_trajectories` |
| `early_detection.py` | `early_detection_table`, `trajectory_frame` |
| `run_artifacts.py` | `save_run`, `record_test_metrics` (wrap `provenance.*`) |

All re-exported from `common/__init__.py`.

### Status

- New tests: 24 passing. Full `CLASSIFIER/tests/` suite: 138 passing.
- Template: valid notebook JSON, all 32 cells compile, `common.*` imports resolve.

### Next step (not yet done)

Implement the six hooks per family to produce the concrete GELSTM / GEC notebooks, then
migrate the real notebooks onto these helpers.

## Static template

[SAMPLE_STATIC_TEMPLATE.ipynb](SAMPLE_STATIC_TEMPLATE.ipynb) documents the whole-brain
reconstruction-autoencoder **pretraining** pipeline: seed → config → checkpoint picker →
`GraphDatasetInMemoryFiltered` × 3 → model → train-or-load → loss curve → cohort
reconstruction-error analysis → robustness evaluation.

It was written by reading the two notebooks it generalizes —
[STATIC_GAAE_DELCODE_WHOLE_BRAIN.ipynb](../STATIC/STATIC_GAAE_DELCODE_WHOLE_BRAIN.ipynb) and
[STATIC_VGAE_DELCODE_WHOLE_BRAIN.ipynb](../STATIC/STATIC_VGAE_DELCODE_WHOLE_BRAIN.ipynb) —
which had diverged: GAAE has a cohort-error + robustness-evaluation section that VGAE's
notebook lacks entirely, and they used two different W&B/checkpoint conventions (raw
`wandb.login()/finish()` vs. `common.tracking` + `update_latest_checkpoint`). Both existing
notebooks are left untouched; this template is additive only.

### What we did

All model-specific work is funneled through four `raise NotImplementedError` **hooks**:

| Hook | Responsibility |
| --- | --- |
| `build_model(in_features, cfg, device)` | instantiate the model, return `(model, model_config)` |
| `run_training(model, optimizer, train_loader, val_loader, cfg, device, wandb_run)` | train one model to convergence, return `(best_state_dict, history)` |
| `compute_sample_error(sample, model, device, cfg)` | per-sample reconstruction error — `float` or `dict` with `"total_error"` |
| `latest_checkpoint_tag(cfg)` | tag string passed to `update_latest_checkpoint` (e.g. `"GAAE"`, `f"VGAE_{conv_type.upper()}"`) |

It standardizes on the `common.tracking` + `update_latest_checkpoint` convention (matching
the longitudinal template) rather than GAAE's older raw `wandb.login()/finish()`.

**New shared module:**

| Module | Public API |
| --- | --- |
| `reconstruction_eval.py` | `compute_errors_for_dataset(dataset, split_name, error_fn, cohort_map, ...)` — a generic, hook-driven replacement for `model/GAAE/evaluation.py::compute_errors_for_dataset`, which is hardwired to GAAE's two-component loss |

`compute_one_vs_rest_thresholds` / `is_cohort_positive` / `plot_cohort_errors` /
`plot_robustness_sweep` (from `model/GAAE/evaluation.py`) only touch DataFrames/dicts — no
GAAE coupling despite living there — so the template reuses them as-is rather than
duplicating or moving them.

### Status

- New tests: lint clean (`ruff check`), full `CLASSIFIER/tests/` suite still passing (205).
- Template: valid notebook JSON (25 cells), `common.*` / `model.GAAE.evaluation` imports
  resolve.

### Next step (not yet done)

Implement the four hooks for GAAE and VGAE to produce a single concrete
`STATIC_COMMON_DELCODE.ipynb` (mirroring how `LONGITUDINAL_COMMON_DELCODE.ipynb` already
picks an adapter), selectable purely via the `MODEL` param from `run_experiment.py` +
`experiments/*.yaml` — replacing the need for one bespoke notebook file per encoder
architecture.
