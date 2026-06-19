# Experiment Runner

Reproducible, background-capable, tracked execution of the notebooks registered in
[`experiments.yaml`](experiments.yaml), via [`run_experiment.py`](run_experiment.py)
(papermill).

## What you can do

Run from the `CLASSIFIER/` directory:

```bash
python run_experiment.py --id gelstm-trajectory-whole-brain    # one experiment
python run_experiment.py --id <id> --background                # detach, returns immediately
python run_experiment.py --all                                  # sequential queue, continue-on-error
python run_experiment.py --mode longitudinal                    # filter then queue
python run_experiment.py --status                                # table of every run
python run_experiment.py --collect                                # rebuild outputs/RESULTS.csv
python run_experiment.py --dry-run --id <id>                     # preview merged params, no execution
python run_experiment.py --id <id> --no-wandb                    # force WANDB_MODE=disabled
python run_experiment.py --id <id> --require-clean                # hard-fail on a dirty git tree
```

## What was built

### Engine (unit-tested, `pytest CLASSIFIER/tests/` passes)

- [`common/tracking.py`](common/tracking.py) — single W&B entry point for every model
  (`init_run` / `log_metrics` / `finish_run`). Online by default, auto-falls back to
  `offline` if no credentials/network, returns a no-op stub under `WANDB_MODE=disabled`.
  Never blocks a headless run.
- [`common/experiment_utils.py`](common/experiment_utils.py) — registry load/validate,
  hyperparameter merge (`dataclass defaults < JSON config_path < YAML hyperparams <
  YAML eval_config`), results-ledger aggregation (`collect_results`, `read_statuses`).
- [`run_experiment.py`](run_experiment.py) — the CLI: sequential queue, `nohup`
  backgrounding, per-run `run.log` / `status.json`, `RESULTS.csv` / `RESULTS.jsonl`,
  `latest` symlink, `.env` loader.
- [`common/checkpoints.py`](common/checkpoints.py) — `select_gaae_checkpoint(...,
  checkpoint_path=None)` non-interactive bypass so a known checkpoint path skips the
  `input()` prompt.

### Run layout

```
CLASSIFIER/outputs/
  <experiment-id>/
    runs/
      <timestamp>/
        <notebook>_run.ipynb   # papermill-executed notebook
        run.log                # stdout/stderr
        status.json            # {state, started_at, finished_at, pid, exit_code, error}
        run_summary.json        # git + env + params + metrics
        resolved_config.json    # final merged hyperparameters for this run
    latest -> runs/<timestamp>/
  RESULTS.csv
  RESULTS.jsonl
```

Each run maps 1:1 to a W&B run named `<id>-<short_git>-<timestamp>`, in project
`ad-early-detection`, grouped by experiment id, `job_type=<model>`.

### W&B, on by default for every model

GAAE's previously-hardcoded `wandb.init` was removed in favor of an injected
`wandb_run=` (mirrors GEC/GELSTM). Per-epoch learning curves are logged from the
GELSTM/GEC/FIRST_N training loops. The API key lives in the gitignored repo-root
`.env` (`WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT`) — confirmed via
`git check-ignore .env`. **Never hardcode the key in code, notebooks, or YAML.**

### Notebooks

- [`dev/patch_runner_params.py`](dev/patch_runner_params.py) injected a
  `parameters`-tagged cell into all 12 registry notebooks (idempotent).
- [`dev/wire_runner_notebooks.py`](dev/wire_runner_notebooks.py) wired the 7
  notebooks with prompts/training to consume those parameters (idempotent,
  fail-loud, every edited cell `compile()`-checked): checkpoint guard, threshold
  guard, config-merge, run-dir wiring, W&B init/per-epoch logging/finish, uniform
  metrics block. No unguarded `input()` remains; interactive Jupyter use is
  unchanged when params are left `None`.

### Not yet wired for metrics/W&B

These 4 no-prompt aggregation/sanity notebooks run headless and already appear in
`--status`, but have no `run_summary`/W&B-summary write of their own (only the
parameters cell from `patch_runner_params.py`):

- `notebooks/SANITY/SANITY_SPLIT_HYGIENE_DELCODE.ipynb`
- `notebooks/SANITY/SANITY_BASELINE_METADATA_TIME.ipynb`
- `notebooks/BASELINE/BASELINE_MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb`
- `notebooks/COMPARISON/COMPARISON_CROSS_REGION_CLASSIFIER.ipynb`
- `notebooks/COMPARISON/COMPARISON_CROSS_REGION_SURVIVAL.ipynb`

## Caveats

- The runner itself was tested end-to-end with a synthetic notebook and the unit
  tests; the wired training notebooks were verified statically (compile-checked,
  nbformat-valid, idempotent) — not run against real DELCODE data/GPU/checkpoint.
  First real smoke test to run: `python run_experiment.py --id logreg-static`
  (fastest trainable entry, no GPU/checkpoint dependency).
- Runner deps are in [`requirements-runner.txt`](requirements-runner.txt)
  (`papermill`, `ipykernel`, `wandb`, `PyYAML`), installed on top of the root
  `.venv`.
