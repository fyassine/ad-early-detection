# CI — quality gates

CI (`.github/workflows/test.yml`) runs on every push to `main` and on every PR. Before
handing off code as done, run the blocking checks locally — don't wait for CI to report
the failure.

## No new errors

Across every check below — blocking or non-blocking — a change should not introduce
errors that weren't already there. Pre-existing failures elsewhere in the codebase are
not yours to fix as a side effect, but don't add to the count.

## Local check script — run this before saying a task is done

```bash
python scripts/run_checks.py
```

Run this **once, after all steps of a multi-step implementation plan are finished** —
not after each individual step. Intermediate steps can leave the tree in a transiently
broken state (a function defined before its caller is wired up, a test added before the
code it covers); checking after each step produces false regressions and wastes a full
lint+test+mypy+bandit pass per step. Run it once the whole plan is implemented, right
before handing the work off as done.

Runs every check below in one go and mirrors `.github/workflows/test.yml` exactly. The
blocking checks must have zero issues. The non-blocking checks are **ratcheted** against
`CHECKS.json` (repo root, gitignored, not a number to hardcode in this doc): a
pre-existing backlog of findings is fine, but the script fails if your change introduces
*any* finding — blocking or non-blocking — that wasn't already in that file, and tells
you exactly which one. On a clean pass it rewrites `CHECKS.json` with the current state
(counts can shrink if you fixed something, that's fine). On failure it leaves
`CHECKS.json` untouched so the same regression is reported again until it's fixed.

**Never hand-edit `CHECKS.json` or otherwise force a failure to disappear** — fix the
regression and rerun the script.

## Blocking (must pass before you say a task is done)

- `ruff check .` — lint (pyflakes, import sort, pycodestyle subset, bugbear)
- `pytest CLASSIFIER/tests/ PROGNOSER/tests/ DATA/src/splitting/tests/ -q` — full suite

## Non-blocking (ratcheted via the script — see above; don't fix unrelated backlog)

- `ruff format --check .` — formatting
- `ruff check --select C90 .` — McCabe complexity
- `mypy CLASSIFIER PROGNOSER DASHBOARD DATA/src` — type checking, `ignore_missing_imports = true` (not `strict`)
- `bandit -r CLASSIFIER PROGNOSER DASHBOARD DATA/src -c .bandit` — security static analysis
- `pip-audit` — dependency CVE scan (skipped, not failed, if offline)

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
