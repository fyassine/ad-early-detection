Research codebase for Alzheimer's disease early detection using longitudinal brain graphs (DELCODE cohort).

## Outline

**Active — write new code here:**

| Directory | Purpose |
|---|---|
| [`CLASSIFIER/`](CLASSIFIER/README.md) | Graph classifiers (GAAE, GEC, GELSTM, VGAE, GEP). Reproducibility framework (seeding, configs, splits, checkpoints) lives here. |
| [`PROGNOSER/`](PROGNOSER/README.md) | Survival analysis (Cox, RSF, LSTM-Surv, Kaplan-Meier). Consumes GAAE embeddings to predict time-to-conversion. |
| [`DASHBOARD/`](DASHBOARD/README.md) | FastAPI backend + Vite frontend for cohort browsing and GELSTM inference. |
| `DATA/src/processing/` | Preprocessing pipeline — atlas/subcortex extraction, run-all orchestration. |

**Legacy — read-only:**

| Directory | Purpose |
|---|---|
| `ABI/` | Artifact-based indices, notebooks only. |
| `DCI/` | Disconnectivity index, notebooks only. |

**Other:**

| Directory | Purpose |
|---|---|
| `DOCS/` | Design notes, pipeline write-ups, codebase analysis docs. |

Do not pattern-match against legacy code when writing new code.

## Environment

Single project-root `.venv` (Python 3.10.12) used by `CLASSIFIER/`, `PROGNOSER/`, and DASHBOARD's model-inference path:

```
torch 2.10.0+cu128
torch_geometric 2.7.0
numpy 1.26.4
pandas 2.2.0
scikit-learn 1.7.2
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # if present, else see subpackage requirements below

# Subpackage extras (install on top of the root venv — do not duplicate torch/PyG)
pip install -r PROGNOSER/requirements.txt           # lifelines, scikit-survival, joblib
pip install -r DASHBOARD/requirements.txt            # fastapi, uvicorn, pandas, scipy, networkx, umap-learn
pip install -r CLASSIFIER/requirements-explain.txt   # captum, nilearn (EXPLAIN notebook only)
```

Do not pin different torch/PyG versions in subpackage requirements, and don't assume CUDA is available in test code (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## Tests

```bash
pytest CLASSIFIER/tests/ PROGNOSER/tests/ DATA/src/splitting/tests/
```

## CLASSIFIER

Graph autoencoders + classifiers for converter vs. non-converter prediction.

```
CLASSIFIER/
  common/      seeding.py, splits.py, sanity.py, dataset.py
  configs/     @dataclass hyperparameter bundles (GELSTMTrainConfig, GECTrainConfig, EvalConfig, ...)
  model/       GAAE/ GEC/ GELSTM/ VGAE/ GEP/ utils/ — model families + train/eval modules
  notebooks/   orchestration only — prefix-tagged (see below)
  outputs/     all new run artifacts (gitignored)
  checkpoints/ legacy artifacts (back-compat only — do not write new runs here)
  tests/       pytest suite
  experiments.yaml   registry of major runs
```

Every notebook in `notebooks/` starts with exactly one prefix:

| Prefix | Meaning |
|---|---|
| `BASELINE_` | Baseline only (first visit, no follow-up) |
| `LONGITUDINAL_` | Multi-visit / full-trajectory experiments |
| `STATIC_` | Per-scan / cross-sectional |
| `SANITY_` | Sanity checks and ablations |
| `COMPARISON_` | Cross-model or cross-region aggregation of saved predictions (no training) |
| `EXPLAIN_` | Explainability / diagnostics on a reloaded model (no training) |

**Reproducibility contract** — every training notebook opens with:

```python
from CLASSIFIER.common.seeding import set_seed, make_rng, make_torch_generator, seed_worker
SEED = 42
set_seed(SEED)
rng = make_rng(SEED)
torch_gen = make_torch_generator(SEED)
```

Pass the RNG explicitly into anything that shuffles (`make_batches(..., rng=rng)`, `DataLoader(..., generator=torch_gen, worker_init_fn=seed_worker)`). All train/val/test partitioning goes through `common.splits.make_splits` so splits are consistent and reproducible across notebooks.

**Checkpoints** are full-state (model + optimizer + scheduler + RNG state + config), loaded with `ckpt.get("model_state_dict", ckpt)` for back-compat with old weights-only files.

Run an experiment via the registry rather than ad hoc notebook execution:

```bash
python run_experiment.py --id <experiment_id>     # CLASSIFIER/experiments.yaml entries
python run_experiment.py --mode explain           # all EXPLAIN_ entries
```

W&B logging is on by default, routed through `common/tracking.py` (`wandb_run=` injected, not a hardcoded `wandb.init`).

Full reference: [CLASSIFIER/README.md](CLASSIFIER/README.md) (notebook index, training entrypoints, checkpoint schema, import convention).

## PROGNOSER

Time-to-conversion (MCI → AD) survival analysis. Complements the binary CLASSIFIER by predicting *when* a subject converts, using the same longitudinal logic for converters and non-converters (restricted to each subject's at-risk window — no look-ahead bias).

```
PROGNOSER/
  common/   survival_table.py, longitudinal.py, embeddings.py, metrics.py, io.py
  model/    SurvivalModel ABC + kaplan_meier, cox, cox_time_varying, rsf, lstm_surv, deepsurv
  src/      build_subject_embeddings.py — CLI to compute + cache GAAE embeddings per strategy
  notebooks/
    KAPLAN_MEIER_BASELINE.ipynb
    PROGNOSER_RUNNER.ipynb          # parameterized: combo × method × strategy
    CROSS_NETWORK_COMPARISON.ipynb  # leaderboard heatmap
```

Methods: `km`, `cox_clinical`, `cox_clinical_longitudinal`, `cox_embedding`, `cox_combined`, `cox_time_varying`, `rsf`, `deepsurv`, `lstm_surv`. Embedding strategies: `baseline`, `last`, `mean`, `slope`, `all_aggs`, `sequence`.

```bash
pip install -r PROGNOSER/requirements.txt
python -m PROGNOSER.src.build_subject_embeddings --all --strategy all_aggs
jupyter notebook PROGNOSER/notebooks/CROSS_NETWORK_COMPARISON.ipynb
```

Full reference: [PROGNOSER/README.md](PROGNOSER/README.md), [DOCS/prognosis_pipeline.md](DOCS/prognosis_pipeline.md).

## DASHBOARD

FastAPI backend + Vite frontend for cohort browsing and GELSTM inference, run over SSH tunnel (not `npm run dev` — that's port 5173, not what the tunnel forwards).

```bash
cd DASHBOARD/frontend && npm install && npm run build   # one-time / after JS/CSS edits

# on the remote
DASHBOARD/restart.sh --bg     # starts FastAPI on :8050 in the background

# on your laptop
ssh -L 8050:localhost:8050 wunderlich@138.245.113.6
```

Then open `http://localhost:8050`. Use the project-root `.venv` for the server (not `DASHBOARD/.venv`, which lacks torch/PyG). `restart.sh --rebuild` after frontend changes, `--clean-gelstm` after GELSTM code/checkpoint changes, `--full` for both.

Full reference: [DASHBOARD/README.md](DASHBOARD/README.md) (API reference, GELSTM ensemble deployment, cache layout, troubleshooting).

## Architecture rules

- `CLASSIFIER/model/**` — pure logic. No I/O, no `wandb.init`, no path construction, no hardcoded hyperparameters.
- `CLASSIFIER/configs/**` — `@dataclass` hyperparameter bundles.
- `CLASSIFIER/notebooks/**` — orchestration only; training loops live in `model/`.
- `CLASSIFIER/common/**` — shared utilities with no model-specific imports.
- Errors fail loudly: prefer raising over silent fallbacks (e.g. no implicit `threshold=0.5` default — see `.claude/rules/errors.md`).
