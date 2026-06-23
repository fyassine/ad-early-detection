# CI — quality gates

CI (`.github/workflows/test.yml`) runs on every push to `main` and on every PR. Before
handing off code as done, run the blocking checks locally — don't wait for CI to report
the failure.

## Blocking (must pass before you say a task is done)

- `ruff check .` — lint (pyflakes, import sort, pycodestyle subset, bugbear)
- `pytest CLASSIFIER/tests/ PROGNOSER/tests/ DATA/src/splitting/tests/ -q` — full suite

## Non-blocking (run, report, don't let them gate the task)

These surface real signal but the codebase has a pre-existing backlog in each — don't
try to fix unrelated pre-existing findings as part of an unrelated change. Only act on
new findings your change introduces.

- `ruff format --check .` — formatting (145 pre-existing files unformatted as of 2026-06-23)
- `ruff check --select C90 .` — McCabe complexity (39 pre-existing violations)
- `mypy CLASSIFIER PROGNOSER DATA/src` — type checking, `ignore_missing_imports = true` (not `strict`)
- `bandit -r CLASSIFIER PROGNOSER DASHBOARD DATA/src -c .bandit` — security static analysis
- `pip-audit` — dependency CVE scan

Coverage (`pytest-cov`, configured via `[tool.pytest.ini_options].addopts` in
`pyproject.toml`) reports `--cov-report=term-missing` on every test run but enforces no
minimum threshold. Don't add `--cov-fail-under` without the user asking — see
[[feedback_ci_gate_scope]] for why these stay non-blocking.

## Local tooling

All CI tooling versions are pinned in `requirements-dev.txt` at the repo root:

```bash
pip install -r requirements-dev.txt
```

`pytest-randomly` and `pytest-xdist` are in that file and activate automatically once
installed — no flags needed locally; CI passes `-n auto --dist loadscope` explicitly for
parallelism. If a test only passes in one ordering, that's a real bug (hidden state
dependency) — fix the test, don't disable randomization.

## Pre-commit

`.pre-commit-config.yaml` strips notebook outputs (`nbstripout`) and runs `ruff --fix`.
It does not run mypy/bandit/format — those stay CI-only and non-blocking (see above).
