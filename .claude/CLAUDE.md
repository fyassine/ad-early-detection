# CLAUDE.md — AD Early Detection

Research codebase for Alzheimer's disease early detection using longitudinal brain graphs (DELCODE cohort).

## Active vs legacy directories

- **Active** (write new code here): `CLASSIFIER/` (graph classifiers), `PROGNOSER/` (survival analysis, consumes GAAE embeddings), `ABI/` (Abnormality Index experiment runner, 2 wired notebooks), `DASHBOARD/` (FastAPI+Vite app), `DATA/` (preprocessing pipeline, scripts, configs)
- **Legacy / read-only**: everything inside `__CLASSIFIER__/`, `DCI/`; within `ABI/`, `comp_corr_v1.py` and `dci_scripts/` (hardcoded HPC paths)

## Code search scope

Search and index ALL files in the repository **including `DATA/`** and every other folder,
except the following which are excluded for performance/privacy — never read these:

- `.venv/` — Python virtual environment (vendored packages, not project code)
- `.git/` — git internals
- `.env` — secrets / credentials file
- `**/wandb/` — ML experiment tracker artifacts
- `**/checkpoints/` — large model checkpoint blobs (`.pth`)
- `**/__pycache__/` — compiled bytecode
- `**/*.nii.gz`, `**/*.npz`, `**/*.pkl` — large binary data arrays
- `**/*.csv`, `**/*.xlsx`, `**/*.xls` — raw data tables (too large to reason over)
- `DATA/src/processing/subcortex/` — vendored third-party toolbox (non-project code)

Everything else — Python scripts, notebooks, JSON configs, shell scripts, markdown docs,
yaml files, etc. — is project code and **must** be included in searches.

## Rule modules (loaded automatically)

@.claude/rules/architecture.md
@.claude/rules/environment.md
@.claude/rules/errors.md
@.claude/rules/seeding.md
@.claude/rules/configs.md
@.claude/rules/evaluation.md
@.claude/rules/checkpoints.md
@.claude/rules/notebooks.md
@.claude/rules/ci.md

## Reference docs (load on demand — do not embed)

- `CLASSIFIER/README.md` — full reproducibility contract, checkpoint schema, notebook index
- `CLASSIFIER/experiments/` — run registry directory (split by domain)
- `PROGNOSER/README.md` — survival pipeline
- `ABI/README.md` — Abnormality Index experiment runner, registry schema
- `DASHBOARD/README.md` — app setup, venv contract

## Tests and commands

```bash
python scripts/run_checks.py
```

Run this once all steps of an implementation are finished, before handing code off as
done — not after each individual step of a multi-step plan (see [ci.md](rules/ci.md)
for why). It runs lint + tests (must pass) plus type/format/complexity/security checks
(ratcheted — your change must not introduce findings beyond the existing backlog).
