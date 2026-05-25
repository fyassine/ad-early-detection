# Architecture

## Active vs legacy directories

**Active — write new code here:**

- `CLASSIFIER/` — graph classifiers (GAAE, GEC, GELSTM). The reproducibility framework lives here.
- `PROGNOSER/` — survival analysis (Cox, RSF, LSTM-Surv, Kaplan-Meier). Consumes GAAE embeddings.
- `DASHBOARD/` — FastAPI backend + Vite frontend for cohort browsing and GELSTM inference. Has its own `requirements.txt` for the API layer but reuses the project-root `.venv` for torch / torch_geometric / nilearn.
- `DATA/src/processing/` — preprocessing pipeline. Atlas/subcortex extraction, run-all orchestration.

**Legacy — read-only:**

- `__CLASSIFIER__/` — prior iteration; keep only for back-compat artifact loading.
- `ABI/` — artifact-based indices, notebooks only.
- `DCI/` — disconnectivity index, notebooks only.

Do not pattern-match against legacy code when writing new code.

## Layered architecture (enforce in all new code)

- `CLASSIFIER/model/**` — pure logic. No I/O, no `wandb.init`, no path construction, no hardcoded hyperparameters. If you need a hyperparameter, add a field to the relevant dataclass in `configs/`.
- `CLASSIFIER/configs/**` — `@dataclass` hyperparameter bundles with typed defaults.
- `CLASSIFIER/notebooks/**` — orchestration only: load data, instantiate config, call into `model/`, log/save results. Training loops do not live in notebooks.
- `CLASSIFIER/common/**` — shared utilities with no model-specific imports. Seeding, splits, sanity audits, validation hooks.

## GAAE intentionally un-refactored

`CLASSIFIER/model/GAAE/train.py` still has hardcoded `wandb.init` at lines 15–22 and uses loose kwargs rather than a dataclass config. This is deliberate (the GAAE encoder is a pretrained feature extractor consumed by GEC and GELSTM — its training loop runs once, not per experiment). Do not copy this pattern into new training code. Use `model/GEC/train.py` and `model/GELSTM/train.py` as the reference.
