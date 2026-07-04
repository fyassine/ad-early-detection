# ABI

Abnormality Index (ABI) experiment runner — adapts the per-ROI deviation index
from Stoecklein et al. (2020) to Schaefer-200 whole-brain correlation matrices,
as a lightweight, non-trained baseline for converter-vs-MCI classification.

Two notebooks are wired into the registry:

| Notebook | `id` example | What it does |
|---|---|---|
| `ABI_BASELINE.ipynb` | `abi-baseline-delcode` | Cross-sectional ABI per scan; 5-fold stratified CV picks the Youden threshold; evaluated on a held-out test split. |
| `ABI_LONGITUDINAL_DELCODE_WHOLE_BRAIN.ipynb` | `abi-longitudinal-delcode-whole-brain` | Subject-level ABI across all visits (`abi_baseline`/`abi_last`/`abi_delta`); 5-fold subject-level (`StratifiedGroupKFold`) CV picks the best feature + threshold. |

`comp_corr_v1.py` and `dci_scripts/*.py` are **not** part of this runner — they
are standalone HPC batch scripts with hardcoded cluster paths, a different
execution environment entirely.

## Quick Start

```bash
cd ABI
python run_experiment.py --id abi-baseline-delcode --dry-run   # preview merged params
python run_experiment.py --id abi-baseline-delcode              # real run
python run_experiment.py --all                                  # run both, sequentially
python run_experiment.py --status                               # table of all runs
python run_experiment.py --collect                              # rebuild outputs/RESULTS.csv
```

## Registry (`experiments.yaml`)

Required fields: `id`, `notebook`, `seed`. `kind` (`baseline` | `longitudinal`)
is derived from the notebook filename — there's exactly one notebook per kind,
so a separate `kind:` field could only ever disagree with `notebook:` for no
benefit.

Optional fields: `config_path` (JSON file under `configs/` overriding the
notebook's default scalars/paths), `config` (inline override mapping, wins
over `config_path`), `wandb` (`true` to opt in — see below), `output_dir`,
`notes`.

Config merge order: `DEFAULT_CONFIG[kind]` (in `common/experiment_utils.py`)
`< config_path JSON < inline config: block`. `RANDOM_STATE` is always taken
from the registry's `seed` field — it cannot drift from it.

## W&B: off by default

Unlike `CLASSIFIER/` and `PROGNOSER/`, ABI experiments default to
`wandb: false`. These notebooks are CV-only (no training loop, runs in
seconds), so W&B's main value — watching live training — doesn't apply. Set
`wandb: true` on a registry entry to opt in; runs land in the
`ad-early-detection-abi` project. `SHARED.tracking.init_run` is a true no-op
when disabled, so wiring it in costs nothing either way.

## Outputs

Each run writes to `outputs/<id>/runs/<display_name>-<git>-<timestamp>/`:
`resolved_config.json`, `run_summary.json` (flat `metrics` block — `cv_auc_mean`,
`best_threshold`, `test_auc`, `test_sensitivity`, `test_specificity`, `test_f1`,
...), `status.json`, `run.log`, `source/` (code snapshot), and the executed
notebook. `outputs/<id>/latest` symlinks to the most recent run.
`run_experiment.py --collect` flattens every `run_summary.json` into
`outputs/RESULTS.csv` / `RESULTS.jsonl`.

## Layout

```
ABI/
├── common/
│   └── experiment_utils.py   # registry loader, config merge, results ledger
├── configs/
│   ├── baseline_delcode_whole_brain.json
│   └── longitudinal_delcode_whole_brain.json
├── notebooks/
│   ├── ABI_BASELINE.ipynb
│   └── ABI_LONGITUDINAL_DELCODE_WHOLE_BRAIN.ipynb
├── tests/
│   └── test_experiment_utils.py
├── experiments.yaml
└── run_experiment.py
```
