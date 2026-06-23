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
@.claude/rules/ci.md

## Reference docs (load on demand — do not embed)

- `CLASSIFIER/README.md` — full reproducibility contract, checkpoint schema, notebook index
- `CLASSIFIER/experiments/` — run registry directory (split by domain)
- `PROGNOSER/README.md` — survival pipeline
- `DASHBOARD/README.md` — app setup, venv contract

## Tests and commands

```bash
python scripts/run_checks.py
```

Run this once all steps of an implementation are finished, before handing code off as
done — not after each individual step of a multi-step plan (see [ci.md](rules/ci.md)
for why). It runs lint + tests (must pass) plus type/format/complexity/security checks
(ratcheted — your change must not introduce findings beyond the existing backlog).
