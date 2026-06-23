"""
Run the full local check suite and ratchet non-blocking findings.

Mirrors .github/workflows/test.yml exactly:

  Blocking (must have zero issues):
    - ruff check .
    - pytest CLASSIFIER/tests/ PROGNOSER/tests/ DATA/src/splitting/tests/

  Non-blocking, ratcheted against CHECKS.json (a pre-existing backlog is fine,
  a *new* finding is not):
    - ruff format --check .
    - ruff check --select C90 .
    - mypy CLASSIFIER PROGNOSER DASHBOARD DATA/src
    - bandit -r CLASSIFIER PROGNOSER DASHBOARD DATA/src -c .bandit
    - pip-audit (skipped, not failed, if it can't reach the network)

CHECKS.json (repo root, gitignored) stores the fingerprinted findings from the
last clean run. A run that introduces no new findings overwrites it (counts
may shrink as pre-existing issues get fixed — that's fine). A run that
regresses leaves it untouched, so the same regression is reported again on
retry instead of being silently absorbed into the baseline.

Never hand-edit CHECKS.json to make a failure disappear — fix the regression
and rerun.

Run with (project-root .venv active):  python scripts/run_checks.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS_FILE = ROOT / "CHECKS.json"

BLOCKING_CHECKS = ["ruff_check", "pytest"]
RATCHETED_CHECKS = ["ruff_format", "ruff_complexity", "mypy", "bandit", "pip_audit"]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    return proc.returncode, proc.stdout + proc.stderr


def run_json(cmd: list[str], timeout: float | None = None) -> dict | list | None:
    """Like run(), but keeps stdout clean of stderr noise (progress bars, log
    lines) so it can be parsed as JSON. Returns None if parsing fails."""
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False, timeout=timeout
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def check_ruff_lint() -> tuple[bool, str]:
    code, output = run(["ruff", "check", "."])
    return code == 0, output


def check_pytest() -> tuple[bool, str]:
    code, output = run(
        [
            "pytest",
            "CLASSIFIER/tests/",
            "PROGNOSER/tests/",
            "DATA/src/splitting/tests/",
            "-q",
            "-n",
            "auto",
            "--dist",
            "loadscope",
        ]
    )
    return code == 0, output


def check_ruff_format() -> set[str]:
    _, output = run(["ruff", "format", "--check", "."])
    fingerprints = set()
    for line in output.splitlines():
        if line.startswith("Would reformat: "):
            fingerprints.add(line.removeprefix("Would reformat: ").strip())
    return fingerprints


def check_ruff_complexity() -> set[str]:
    issues = run_json(["ruff", "check", "--select", "C90", "--output-format=json", "."])
    if issues is None:
        return set()
    return {
        f"{issue['filename']}:{issue['location']['row']}:{issue['code']}"
        for issue in issues
    }


_MYPY_LINE_RE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: .*\[(?P<code>[\w-]+)\]$")


def check_mypy() -> set[str]:
    _, output = run(["mypy", "CLASSIFIER", "PROGNOSER", "DASHBOARD", "DATA/src"])
    fingerprints = set()
    for line in output.splitlines():
        match = _MYPY_LINE_RE.match(line)
        if match:
            fingerprints.add(f"{match['file']}:{match['line']}:{match['code']}")
    return fingerprints


def check_bandit() -> set[str]:
    report = run_json(
        [
            "bandit",
            "-r",
            "CLASSIFIER",
            "PROGNOSER",
            "DASHBOARD",
            "DATA/src",
            "-c",
            ".bandit",
            "-f",
            "json",
            "-q",
        ]
    )
    if report is None:
        return set()
    return {
        f"{result['filename']}:{result['line_number']}:{result['test_id']}"
        for result in report.get("results", [])
    }


def check_pip_audit() -> tuple[set[str], bool]:
    """Returns (fingerprints, skipped)."""
    try:
        report = run_json(["pip-audit", "--format", "json"], timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return set(), True
    if report is None:
        return set(), True
    fingerprints = set()
    for dep in report.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            fingerprints.add(f"{dep['name']}:{dep['version']}:{vuln['id']}")
    return fingerprints, False


def load_baseline() -> dict:
    if not CHECKS_FILE.exists():
        return {}
    try:
        return json.loads(CHECKS_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def main() -> int:
    baseline_exists = CHECKS_FILE.exists()
    baseline = load_baseline()
    regressions: dict[str, list[str]] = {}
    new_state: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("Running blocking checks...")
    lint_ok, lint_output = check_ruff_lint()
    new_state["ruff_check"] = {"status": "pass" if lint_ok else "fail"}
    print(f"[BLOCKING]  ruff check .            {'PASS' if lint_ok else 'FAIL'}")
    if not lint_ok:
        print(lint_output)

    test_ok, test_output = check_pytest()
    new_state["pytest"] = {"status": "pass" if test_ok else "fail"}
    print(f"[BLOCKING]  pytest                  {'PASS' if test_ok else 'FAIL'}")
    if not test_ok:
        print(test_output)

    print("\nRunning ratcheted checks...")

    ratchet_results: dict[str, set[str]] = {
        "ruff_format": check_ruff_format(),
        "ruff_complexity": check_ruff_complexity(),
        "mypy": check_mypy(),
        "bandit": check_bandit(),
    }
    pip_audit_findings, pip_audit_skipped = check_pip_audit()
    ratchet_results["pip_audit"] = pip_audit_findings

    labels = {
        "ruff_format": "ruff format --check",
        "ruff_complexity": "ruff check --select C90",
        "mypy": "mypy",
        "bandit": "bandit",
        "pip_audit": "pip-audit",
    }

    for name in RATCHETED_CHECKS:
        current = ratchet_results[name]
        if name == "pip_audit" and pip_audit_skipped:
            new_state[name] = baseline.get(name, {"findings": []})
            print(f"[ratchet]   {labels[name]:<24} skipped (no network)")
            continue

        previous = set(baseline.get(name, {}).get("findings", []))
        new_findings = (current - previous) if baseline_exists else set()
        new_state[name] = {"findings": sorted(current)}
        if new_findings:
            regressions[name] = sorted(new_findings)
            print(
                f"[ratchet]   {labels[name]:<24} {len(previous)} pre-existing, "
                f"{len(new_findings)} NEW"
            )
        elif not baseline_exists:
            print(f"[ratchet]   {labels[name]:<24} {len(current)} (baseline established)")
        else:
            print(f"[ratchet]   {labels[name]:<24} {len(current)} pre-existing, 0 new")

    blocking_failed = not (lint_ok and test_ok)

    if regressions:
        print("\nNew findings introduced:")
        for name, findings in regressions.items():
            print(f"  {labels[name]}:")
            for finding in findings:
                print(f"    - {finding}")

    if blocking_failed or regressions:
        n_new = sum(len(v) for v in regressions.values())
        reasons = []
        if blocking_failed:
            reasons.append("blocking check(s) failed")
        if regressions:
            reasons.append(f"{n_new} new non-blocking issue(s)")
        print(f"\nRESULT: FAIL — {', '.join(reasons)}. CHECKS.json left unchanged.")
        return 1

    CHECKS_FILE.write_text(json.dumps(new_state, indent=2, sort_keys=True) + "\n")
    print("\nRESULT: PASS — no new issues introduced. CHECKS.json updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
