# Copilot Instructions — AD Early Detection (latest APIs as of May 2026)

## Code style
- Don't write comments unless the WHY is non-obvious (hidden constraint, subtle invariant, workaround)
- Use latest Python 3.10 / PyTorch 2.10 / PyG 2.7 syntax
- Do not generate `.md` files unless explicitly requested
- Don't explain, just implement

## Active vs legacy directories
- **Active** (write new code here): `CLASSIFIER/`, `PROGNOSER/`, `DASHBOARD/`, `DATA/src/processing/`
- **Legacy / read-only**: `everything inside __CLASSIFIER__/`, `ABI/`, `DCI/`

## Rule modules (load when relevant to the task)

- [Architecture](../.claude/rules/architecture.md) — layered code organization (`model/` pure logic, `configs/` dataclasses, `notebooks/` orchestration only); GAAE intentionally un-refactored
- [Environment](../.claude/rules/environment.md) — Python 3.10, torch 2.10, PyG 2.7, pinned versions for each subpackage
- [Errors](../.claude/rules/errors.md) — fail-loud policy, no silent fallbacks
- [Seeding](../.claude/rules/seeding.md) — RNG injection contract (every non-deterministic function takes `rng=`)
- [Configs](../.claude/rules/configs.md) — `@dataclass` hyperparameters, `field(default_factory=...)` for mutables, `GECBatch` is docs-only
- [Evaluation](../.claude/rules/evaluation.md) — best-F1 threshold (default) on val only, never re-threshold on test; use `threshold_mode="f1"` (preferred over `"youden"` for imbalanced data)
- [Checkpoints](../.claude/rules/checkpoints.md) — full-state schema, `outputs/` vs legacy `checkpoints/`
- [Notebooks](../.claude/rules/notebooks.md) — `BASELINE_` / `LONGITUDINAL_` / `STATIC_` / `SANITY_` prefix, splits via `common.splits`, sanity audit at head; three mandatory interactive prompts: (1) GAAE checkpoint index selection, (2) train vs load existing checkpoint, (3) threshold mode with Best-F1 as default (Youden as option 2)
- [CI](../.claude/rules/ci.md) — `ruff check .` + `pytest` block before code is done; mypy/format/complexity/bandit/pip-audit report only; never introduce new errors in any check, but don't fix pre-existing findings outside your change

## Reference docs

- [CLASSIFIER README](../CLASSIFIER/README.md) — reproducibility contract, checkpoint schema, notebook index
- [Experiments registry](../CLASSIFIER/experiments.yaml)
- [PROGNOSER README](../PROGNOSER/README.md)
- [DASHBOARD README](../DASHBOARD/README.md)
