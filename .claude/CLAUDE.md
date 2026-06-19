# CLAUDE.md — AD Early Detection

Research codebase for Alzheimer's disease early detection using longitudinal brain graphs (DELCODE cohort).

## Active vs legacy directories

- **Active** (write new code here): `CLASSIFIER/` (graph classifiers), `PROGNOSER/` (survival analysis, consumes GAAE embeddings), `DASHBOARD/` (FastAPI+Vite app), `DATA/src/processing/` (preprocessing pipeline)
- **Legacy / read-only**: everything inside `__CLASSIFIER__/`, `ABI/`, `DCI/`

## Rule modules (loaded automatically)

@.claude/rules/architecture.md
@.claude/rules/environment.md
@.claude/rules/errors.md
@.claude/rules/seeding.md
@.claude/rules/configs.md
@.claude/rules/evaluation.md
@.claude/rules/checkpoints.md
@.claude/rules/notebooks.md

## Reference docs (load on demand — do not embed)

- `CLASSIFIER/README.md` — full reproducibility contract, checkpoint schema, notebook index
- `CLASSIFIER/experiments.yaml` — run registry (9 entries)
- `PROGNOSER/README.md` — survival pipeline
- `DASHBOARD/README.md` — app setup, venv contract

## Tests and commands

```bash
pytest CLASSIFIER/tests/ PROGNOSER/tests/ DATA/src/splitting/tests/
```
