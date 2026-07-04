# CLAUDE.md — AD Early Detection

Research codebase for Alzheimer's disease early detection using longitudinal brain graphs (DELCODE cohort).

## Active vs legacy directories

- **Active** (write new code here): `CLASSIFIER/` (graph classifiers), `PROGNOSER/` (survival analysis, consumes GAAE embeddings), `ABI/` (Abnormality Index experiment runner, 2 wired notebooks), `DASHBOARD/` (FastAPI+Vite app), `DATA/DELCODE/src/processing/` (preprocessing pipeline)
- **Legacy / read-only**: everything inside `__CLASSIFIER__/`, `DCI/`; within `ABI/`, `comp_corr_v1.py` and `dci_scripts/` (hardcoded HPC paths)

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
