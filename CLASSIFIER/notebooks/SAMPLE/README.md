# SAMPLE — model-agnostic longitudinal workflow

A canonical, non-running **template** for the longitudinal MCI→converter pipeline.
It is the reference every model-specific notebook (GELSTM, GEC-MLP, …) conforms to,
and the spec for the shared logic now living under `CLASSIFIER/common/`.

## Instructions this was built from

1. **Move notebook code into scripts** under the model/common packages so notebooks
   stop being a place where subtle mistakes hide.
2. **Create a model-agnostic SAMPLE notebook** showing every step seed → test, plus the
   post-test cells (ROC, early-detection curve, conversion trajectories). Doesn't need to
   run — it shows the steps. (Dashboard-deploy cell excluded.)
3. **Lift the repeated blocks into `common/`** — explicitly the CV loop, and also the other
   parts that repeat across the GEC/GELSTM notebooks.
4. **Uppercase all cross-cell notebook globals** to match the constant style.

## What we did

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

## Status

- New tests: 24 passing. Full `CLASSIFIER/tests/` suite: 138 passing.
- Template: valid notebook JSON, all 32 cells compile, `common.*` imports resolve.

## Next step (not yet done)

Implement the six hooks per family to produce the concrete GELSTM / GEC notebooks, then
migrate the real notebooks onto these helpers.
