#!/usr/bin/env python3
"""
Experiment runner: execute notebooks from the experiments.yaml registry.

Run from the CLASSIFIER/ directory. Each experiment is executed with papermill,
its parameters injected from the registry, and its artifacts written to a
timestamped run directory under ``outputs/<id>/runs/<timestamp>/``.

Examples
--------
    python run_experiment.py --id gelstm-trajectory-whole-brain
    python run_experiment.py --id sanity-split-hygiene --background
    python run_experiment.py --all
    python run_experiment.py --mode longitudinal
    python run_experiment.py --id <id> --dry-run
    python run_experiment.py --status
    python run_experiment.py --collect

W&B is on by default (see common/tracking.py). Use --no-wandb to disable, or set
WANDB_MODE=offline in the environment. Credentials are read from the repo-root
.env (loaded automatically) or ~/.netrc.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Allow `python run_experiment.py` from CLASSIFIER/ to import the CLASSIFIER package.
_CLASSIFIER_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _CLASSIFIER_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CLASSIFIER.common.experiment_utils import (  # noqa: E402
    build_config,
    build_parameter_dict,
    collect_results,
    load_experiment,
    load_registry,
    read_statuses,
)
from CLASSIFIER.common.provenance import capture_git_provenance  # noqa: E402

_REGISTRY = _CLASSIFIER_ROOT / "experiments.yaml"
_OUTPUTS = _CLASSIFIER_ROOT / "outputs"


# --------------------------------------------------------------------------- #
# Environment / .env
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path = _REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Status file helpers
# --------------------------------------------------------------------------- #
def _write_status(run_dir: Path, **fields) -> None:
    status_path = run_dir / "status.json"
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            status = {}
    status.update(fields)
    status_path.write_text(json.dumps(status, indent=2))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #
def _preflight(exp: dict, require_clean: bool) -> dict:
    """Validate everything cheap before spending GPU time. Returns git info."""
    notebook = _CLASSIFIER_ROOT / exp["notebook"]
    if not notebook.is_file():
        raise FileNotFoundError(f"Experiment {exp['id']!r}: notebook {notebook} not found.")

    ckpt = exp.get("checkpoint_path")
    if ckpt and not (_CLASSIFIER_ROOT / ckpt).exists():
        raise FileNotFoundError(
            f"Experiment {exp['id']!r}: checkpoint_path {ckpt} does not exist "
            f"(resolved to {_CLASSIFIER_ROOT / ckpt})."
        )

    git = capture_git_provenance()
    if git.get("dirty"):
        msg = f"[preflight] git tree is dirty (uncommitted changes) at commit {git.get('short_commit')}."
        if require_clean:
            raise RuntimeError(msg + " Re-run without --require-clean to proceed anyway.")
        print("WARNING:", msg, "Results will record dirty=True.", file=sys.stderr)
    return git


def run_one(exp: dict, *, no_wandb: bool, require_clean: bool) -> bool:
    """Execute one experiment notebook. Returns True on success."""
    print(f"\n=== Running experiment: {exp['id']} ({exp['model']} / {exp['mode']}) ===")
    git = _preflight(exp, require_clean)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_git = git.get("short_commit") or "nogit"
    run_name = f"{exp['id']}-{short_git}-{timestamp}"
    run_dir = _OUTPUTS / exp["id"] / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved hyperparameter config alongside the run.
    resolved_config = build_config(exp, _CLASSIFIER_ROOT)
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2))

    params = build_parameter_dict(exp, _CLASSIFIER_ROOT)
    params["RUN_DIR"] = str(run_dir)
    params["RUN_NAME"] = run_name

    input_nb = _CLASSIFIER_ROOT / exp["notebook"]
    output_nb = run_dir / f"{input_nb.stem}_run.ipynb"
    log_path = run_dir / "run.log"

    _write_status(
        run_dir,
        experiment_id=exp["id"],
        run_name=run_name,
        state="running",
        pid=os.getpid(),
        started_at=_now(),
        git_commit=git.get("short_commit"),
        git_dirty=git.get("dirty"),
        notebook=str(input_nb.relative_to(_CLASSIFIER_ROOT)),
    )

    env_for_run = dict(os.environ)
    if no_wandb:
        env_for_run["WANDB_MODE"] = "disabled"
    env_for_run.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "ad-early-detection"))
    os.environ.update(env_for_run)  # papermill runs in-process kernel; propagate env

    try:
        import papermill as pm
    except ImportError as exc:
        _write_status(run_dir, state="failed", finished_at=_now(),
                      error=f"papermill not installed: {exc}")
        raise

    print(f"  notebook : {input_nb.relative_to(_CLASSIFIER_ROOT)}")
    print(f"  run_dir  : {run_dir.relative_to(_CLASSIFIER_ROOT)}")
    print(f"  log      : {log_path.relative_to(_CLASSIFIER_ROOT)}")
    try:
        with open(log_path, "w") as logf:
            pm.execute_notebook(
                str(input_nb),
                str(output_nb),
                parameters=params,
                cwd=str(_CLASSIFIER_ROOT),
                kernel_name="python3",
                progress_bar=False,
                stdout_file=logf,
                stderr_file=logf,
            )
    except Exception:
        _write_status(run_dir, state="failed", finished_at=_now(),
                      error=traceback.format_exc(limit=3))
        collect_results(_OUTPUTS)
        print(f"  FAILED — see {log_path}", file=sys.stderr)
        return False

    _update_latest_symlink(exp["id"], run_dir)
    _write_status(run_dir, state="done", finished_at=_now(), exit_code=0)
    rows = collect_results(_OUTPUTS)
    row = next((r for r in rows if r.get("run_dir", "").endswith(timestamp)), {})
    metric_summary = {k[len("metric."):]: v for k, v in row.items() if k.startswith("metric.")}
    print(f"  DONE — metrics: {metric_summary}" if metric_summary else "  DONE")
    return True


def _update_latest_symlink(exp_id: str, run_dir: Path) -> None:
    latest = _OUTPUTS / exp_id / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(Path("runs") / run_dir.name)
    except OSError:
        # Filesystems without symlink support (some Windows mounts): write a pointer file.
        (latest.parent / "latest.txt").write_text(str(run_dir.name))


# --------------------------------------------------------------------------- #
# Background launch
# --------------------------------------------------------------------------- #
def launch_background(argv: list[str]) -> None:
    """Re-exec this script detached, without the --background flag."""
    child_args = [a for a in argv if a != "--background"]
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    launch_log = _OUTPUTS / "launch.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args]
    with open(launch_log, "a") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_CLASSIFIER_ROOT),
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"Launched in background (pid {proc.pid}).")
    print(f"  launcher log : {launch_log}")
    print("  track with   : python run_experiment.py --status")


# --------------------------------------------------------------------------- #
# Status / collect commands
# --------------------------------------------------------------------------- #
def cmd_status() -> None:
    statuses = read_statuses(_OUTPUTS)
    if not statuses:
        print("No runs found under outputs/.")
        return
    header = f"{'STATE':<8} {'EXPERIMENT':<34} {'STARTED':<20} {'GIT':<10} RUN"
    print(header)
    print("-" * len(header))
    for s in statuses:
        print(
            f"{str(s.get('state', '?')):<8} "
            f"{str(s.get('experiment_id', '?')):<34} "
            f"{str(s.get('started_at', '?')):<20} "
            f"{str(s.get('git_commit', '?')):<10} "
            f"{s.get('run_name', '?')}"
        )


def cmd_collect() -> None:
    rows = collect_results(_OUTPUTS)
    print(f"Collected {len(rows)} run(s) into {_OUTPUTS / 'RESULTS.csv'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--id", help="Run the single experiment with this id.")
    sel.add_argument("--all", action="store_true", help="Run every experiment sequentially.")
    sel.add_argument("--mode", help="Run every experiment with this mode (e.g. longitudinal).")
    p.add_argument("--dry-run", action="store_true", help="Print merged parameters and exit.")
    p.add_argument("--background", action="store_true", help="Detach and run in the background.")
    p.add_argument("--status", action="store_true", help="Print a table of all runs and exit.")
    p.add_argument("--collect", action="store_true", help="Rebuild RESULTS.csv from run summaries and exit.")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this invocation.")
    p.add_argument("--require-clean", action="store_true", help="Refuse to run if the git tree is dirty.")
    return p.parse_args(argv)


def resolve_targets(args: argparse.Namespace) -> list[dict]:
    if args.id:
        return [load_experiment(_REGISTRY, args.id)]
    registry = load_registry(_REGISTRY)
    if args.mode:
        targets = [e for e in registry if e.get("mode") == args.mode]
        if not targets:
            raise ValueError(f"No experiments with mode={args.mode!r}.")
        return targets
    return registry  # --all


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    load_dotenv()
    args = parse_args(argv)

    if args.status:
        cmd_status()
        return 0
    if args.collect:
        cmd_collect()
        return 0

    if not (args.id or args.all or args.mode):
        print("Nothing to do: pass --id, --all, --mode, --status, or --collect.", file=sys.stderr)
        return 2

    targets = resolve_targets(args)

    if args.dry_run:
        for exp in targets:
            params = build_parameter_dict(exp, _CLASSIFIER_ROOT)
            print(f"\n# {exp['id']}")
            print(json.dumps(params, indent=2, default=str))
        return 0

    if args.background:
        launch_background(argv)
        return 0

    failures = []
    for exp in targets:
        try:
            ok = run_one(exp, no_wandb=args.no_wandb, require_clean=args.require_clean)
        except Exception as exc:  # preflight / papermill-import errors
            ok = False
            print(f"  ERROR: {exc}", file=sys.stderr)
        if not ok:
            failures.append(exp["id"])

    if len(targets) > 1:
        print(f"\n=== Summary: {len(targets) - len(failures)}/{len(targets)} succeeded ===")
        if failures:
            print("Failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
