# PROGNOSER Experiment Runner

Reproducible, background-capable, tracked execution of the survival experiments
registered in [`experiments.yaml`](experiments.yaml), via
[`run_experiment.py`](run_experiment.py) (papermill). Mirrors the
[CLASSIFIER runner](../CLASSIFIER/EXPERIMENTS.md) for the survival pipeline.

Every registry entry runs the **same** notebook
[`notebooks/PROGNOSER_RUNNER.ipynb`](notebooks/PROGNOSER_RUNNER.ipynb) with a
different injected `EXPERIMENT` dict — survival experiments are combo × method ×
strategy sweeps over one notebook, not one-notebook-per-experiment.

## What you can do

Run from the `PROGNOSER/` directory:

```bash
python run_experiment.py --id km-baseline                       # one experiment
python run_experiment.py --id <id> --background                  # detach, returns immediately
python run_experiment.py --all                                    # sequential queue, continue-on-error
python run_experiment.py --method cox_clinical                    # filter by survival method, then queue
python run_experiment.py --status                                  # table of every run
python run_experiment.py --collect                                  # rebuild outputs/RESULTS.csv
python run_experiment.py --dry-run --id <id>                       # preview merged EXPERIMENT, no execution
python run_experiment.py --id <id> --no-wandb                      # force WANDB_MODE=disabled
python run_experiment.py --id <id> --require-clean                  # hard-fail on a dirty git tree
```

## Prerequisite: build the embedding cache

Embedding-based methods (`cox_embedding`, `cox_combined`, `rsf`, `deepsurv`,
`lstm_surv`) read a precomputed GAAE embedding cache. Build it once per
combo × strategy:

```bash
python -m PROGNOSER.src.build_subject_embeddings --combo dmn_hippo --strategy last
```

The runner's preflight checks the cache exists and **fails loud with the exact
command to run** before spending any GPU time. `km` and clinical-only Cox methods
need no cache.

## Registry schema

Each entry: `id`, `method`, `network_combo`, `seed`, `notebook`, plus optional
`experiment:` (override any `EXPERIMENT` key), `wandb: false`, `output_dir`,
`notes`. `network_combo` drives `data_version` + `file_suffix` from the canonical
`COMBO_TABLE` in [`common/experiment_utils.py`](common/experiment_utils.py) —
the single source of truth shared with the embedding-build CLI.

Merge order for the resolved `EXPERIMENT`:
`DEFAULT_EXPERIMENT < combo-derived (data_version/file_suffix) < entry.experiment`.

## What was built

- [`common/experiment_utils.py`](common/experiment_utils.py) — registry
  load/validate (fail-loud: valid method/strategy, embedding methods require a
  strategy, unique ids), `build_experiment` / `build_parameter_dict`, and
  `collect_results` (flattens the nested `metrics[split][c_index|ibs|auc]` block
  into `metric.test_c_index`, `metric.test_auc_24`, … columns).
- [`run_experiment.py`](run_experiment.py) — the CLI: sequential queue, `nohup`
  backgrounding, per-run `run.log` / `status.json`, `RESULTS.csv` / `RESULTS.jsonl`,
  `latest` symlink, shared repo-root `.env` loader, embedding-cache preflight.
- **Reused from CLASSIFIER** (imported, not copied): `CLASSIFIER.common.tracking`
  (W&B `init_run`/`log_metrics`/`finish_run`, on by default, offline fallback,
  never blocks headless) and `CLASSIFIER.common.provenance` (git/env capture,
  run-summary helpers).

## Run layout

```
PROGNOSER/outputs/
  <experiment-id>/
    runs/<run_name>/             # e.g. ethereal-planet-5-349c3823d-2026-06-19_19-31-24
      PROGNOSER_RUNNER_run.ipynb   # papermill-executed notebook
      run.log
      status.json                  # {state, started_at, finished_at, pid, exit_code, error}
      run_summary.json              # git + env + EXPERIMENT + metrics
      resolved_config.json          # final merged EXPERIMENT dict
      source/                       # snapshot of the code that produced this run
        manifest.json               #   (PROGNOSER/**, CLASSIFIER/common/**, splitting/ …)
      git_commit.txt                # commit / branch / dirty at run time
    latest -> runs/<run_name>/
  RESULTS.csv
  RESULTS.jsonl
```

Each run maps 1:1 to a W&B run whose display name matches the directory's random
name (e.g. `ethereal-planet-5`). The git commit lives in the run config, the
timestamp in the local `run_dir`. In project
`ad-early-detection-prognosis` (separate from the classifier project), grouped by
experiment id, `job_type=<method>`, tagged `[survival, <method>, <combo>, seed=<n>]`.

The terminal shows a live `⏱ elapsed` counter while a run executes, then a green
`✓ DONE (MM:SS)` on success or a red `✗ FAILED (MM:SS)` with the failing cell's
traceback (color gated on a TTY; `run.log` stays plain). `source/` holds a
text-only snapshot of the exact code that produced the run.

The legacy `notebooks/checkpoints_prognoser_<combo>/` run dirs still work — the
`CROSS_NETWORK_COMPARISON.ipynb` leaderboard reads both old and new locations.

## Caveats

- The trainable path can't be runtime-verified without DELCODE data + an
  embedding cache + GPU (lstm_surv/deepsurv). Notebook edits are validated
  statically (`nbformat` + per-cell `compile()`). `km-baseline` is the cheapest
  end-to-end smoke test (no embeddings/GPU).
- Runner deps are in [`requirements-runner.txt`](requirements-runner.txt),
  installed on top of the root `.venv` (shared with CLASSIFIER's runner).
