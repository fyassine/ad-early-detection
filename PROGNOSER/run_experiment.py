#!/usr/bin/env python3
"""
PROGNOSER experiment runner: execute survival experiments from experiments.yaml.

Mirror of ``CLASSIFIER/run_experiment.py`` for the survival pipeline. Every
registry entry is executed with papermill against the single parameterized
notebook ``notebooks/PROGNOSER_RUNNER.ipynb``, with its ``EXPERIMENT`` dict
injected, and its artifacts written to ``outputs/<id>/runs/<display_name>-<git>-<timestamp>/``.

Run from the PROGNOSER/ directory.

Examples
--------
    python run_experiment.py --id km-baseline
    python run_experiment.py --id cox-combined-dmn-hippo-last --background
    python run_experiment.py --all
    python run_experiment.py --method cox_clinical
    python run_experiment.py --id <id> --dry-run
    python run_experiment.py --status
    python run_experiment.py --collect

W&B is on by default (see CLASSIFIER/common/tracking.py). Survival runs log to the
``ad-early-detection-prognosis`` project. Use --no-wandb to disable. Credentials are
read from the repo-root .env (loaded automatically) or ~/.netrc.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Allow `python run_experiment.py` from PROGNOSER/ to import the project packages.
_PROGNOSER_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PROGNOSER_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CLASSIFIER.common.provenance import capture_git_provenance, snapshot_source_dirs  # noqa: E402
from CLASSIFIER.common.run_naming import generate_run_name  # noqa: E402
from CLASSIFIER.common.runner_io import (  # noqa: E402
    Heartbeat,
    color,
    format_elapsed,
    format_metric_summary,
)
from PROGNOSER.common.experiment_utils import (  # noqa: E402
    build_experiment,
    build_parameter_dict,
    collect_results,
    load_experiment,
    load_registry,
    read_statuses,
)

_REGISTRY = _PROGNOSER_ROOT / "experiments.yaml"
_OUTPUTS = _PROGNOSER_ROOT / "outputs"
_EMBEDDINGS_CACHE = _PROGNOSER_ROOT / "notebooks" / "_embeddings_cache_"
_DEFAULT_WANDB_PROJECT = "ad-early-detection-prognosis"

# Source trees snapshotted into each run's source/ so a past run can be read back
# with the exact code that produced it. CLASSIFIER/common is included because the
# survival notebook imports tracking/provenance and reads GAAE checkpoints.
_SOURCE_ROOTS = [
    "PROGNOSER/model",
    "PROGNOSER/common",
    "PROGNOSER/src",
    "PROGNOSER/run_experiment.py",
    "PROGNOSER/experiments.yaml",
    "CLASSIFIER/common",
    "DATA/src/splitting",
]

# Embedding-consuming methods that need a precomputed cache before launch.
_EMBEDDING_METHODS = {"cox_embedding", "cox_combined", "rsf", "deepsurv", "lstm_surv"}


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
def _embedding_strategy(exp: dict) -> str | None:
    return build_experiment(exp).get("embedding_strategy")


def _preflight(exp: dict, require_clean: bool) -> dict:
    """Validate everything cheap before spending GPU time. Returns git info."""
    notebook = _PROGNOSER_ROOT / exp["notebook"]
    if not notebook.is_file():
        raise FileNotFoundError(f"Experiment {exp['id']!r}: notebook {notebook} not found.")

    # Embedding-based methods need the cache built first — fail loud with the fix.
    if exp["method"] in _EMBEDDING_METHODS:
        combo = exp["network_combo"]
        strategy = _embedding_strategy(exp)
        cache_file = _EMBEDDINGS_CACHE / f"{combo}_{strategy}_embeddings.parquet"
        if not cache_file.is_file():
            raise FileNotFoundError(
                f"Experiment {exp['id']!r}: method={exp['method']!r} needs the GAAE "
                f"embedding cache {cache_file} which does not exist. Build it first:\n"
                f"    python -m PROGNOSER.src.build_subject_embeddings "
                f"--combo {combo} --strategy {strategy}"
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
    print(f"\n=== Running experiment: {exp['id']} ({exp['method']} / {exp['network_combo']}) ===")
    git = _preflight(exp, require_clean)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_git = git.get("short_commit") or "nogit"
    display_name = generate_run_name(_OUTPUTS / exp["id"])
    run_name = f"{display_name}-{short_git}-{timestamp}"
    run_dir = _OUTPUTS / exp["id"] / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved EXPERIMENT config alongside the run.
    resolved_experiment = build_experiment(exp)
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved_experiment, indent=2))

    # Save the exact source that produced this run (code snapshot).
    snapshot_source_dirs(run_dir, _SOURCE_ROOTS, repo_root=_REPO_ROOT)

    params = build_parameter_dict(exp)
    params["RUN_DIR"] = str(run_dir)
    params["RUN_NAME"] = run_name

    input_nb = _PROGNOSER_ROOT / exp["notebook"]
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
        method=exp["method"],
        network_combo=exp["network_combo"],
        notebook=str(input_nb.relative_to(_PROGNOSER_ROOT)),
    )

    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    # WANDB_PROJECT is fixed by main() (prognosis project, or a genuine shell
    # override) — never the classifier default that .env injects.

    try:
        import papermill as pm
        from papermill.exceptions import PapermillExecutionError
    except ImportError as exc:
        _write_status(run_dir, state="failed", finished_at=_now(),
                      error=f"papermill not installed: {exc}")
        raise

    print(f"  notebook : {input_nb.relative_to(_PROGNOSER_ROOT)}")
    print(f"  run_dir  : {run_dir.relative_to(_PROGNOSER_ROOT)}")
    print(f"  log      : {log_path.relative_to(_PROGNOSER_ROOT)}")
    t0 = time.monotonic()
    try:
        with open(log_path, "w") as logf, Heartbeat(run_name):
            pm.execute_notebook(
                str(input_nb),
                str(output_nb),
                parameters=params,
                cwd=str(_PROGNOSER_ROOT),
                kernel_name="python3",
                progress_bar=False,
                stdout_file=logf,
                stderr_file=logf,
            )
    except PapermillExecutionError as exc:
        elapsed = time.monotonic() - t0
        # Surface the actual notebook error (cell, ename, traceback) in the terminal.
        error_detail = f"{exc.ename}: {exc.evalue}"
        nb_tb = "\n".join(exc.traceback) if exc.traceback else ""
        _write_status(run_dir, state="failed", finished_at=_now(),
                      duration_seconds=round(elapsed, 1),
                      error=error_detail, cell=f"In [{exc.exec_count}]",
                      notebook_traceback=nb_tb)
        collect_results(_OUTPUTS)
        print(color(f"  ✗ FAILED  ({format_elapsed(elapsed)}) — notebook error in cell In [{exc.exec_count}]:", "red"), file=sys.stderr)
        print(f"  {'-' * 70}", file=sys.stderr)
        if nb_tb:
            from papermill.exceptions import strip_color
            print(f"  {strip_color(nb_tb)}", file=sys.stderr)
        else:
            print(f"  {error_detail}", file=sys.stderr)
        print(f"  {'-' * 70}", file=sys.stderr)
        print(f"  Output notebook : {output_nb}", file=sys.stderr)
        print(f"  Run log         : {log_path}", file=sys.stderr)
        return False
    except Exception:
        elapsed = time.monotonic() - t0
        _write_status(run_dir, state="failed", finished_at=_now(),
                      duration_seconds=round(elapsed, 1),
                      error=traceback.format_exc(limit=3))
        collect_results(_OUTPUTS)
        print(color(f"  ✗ FAILED  ({format_elapsed(elapsed)}) — see {log_path}", "red"), file=sys.stderr)
        return False

    elapsed = time.monotonic() - t0
    _update_latest_symlink(exp["id"], run_dir)
    _write_status(run_dir, state="done", finished_at=_now(), exit_code=0,
                  duration_seconds=round(elapsed, 1))
    rows = collect_results(_OUTPUTS)
    row = next((r for r in rows if r.get("run_dir", "").endswith(run_dir.name)), {})
    metric_summary = {k[len("metric."):]: v for k, v in row.items() if k.startswith("metric.")}
    print(color(f"  ✓ DONE  ({format_elapsed(elapsed)})", "green"))
    if metric_summary:
        # train/val/test c-index, IBS, time-AUC — survival has no k-fold CV block.
        print(f"     metrics: {format_metric_summary(metric_summary)}")
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
            cwd=str(_PROGNOSER_ROOT),
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
    sel.add_argument("--method", help="Run every experiment with this survival method (e.g. cox_clinical).")
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
    if args.method:
        targets = [e for e in registry if e.get("method") == args.method]
        if not targets:
            raise ValueError(f"No experiments with method={args.method!r}.")
        return targets
    return registry  # --all


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Capture a genuine shell override before .env (which carries the classifier
    # default) is loaded; PROGNOSER otherwise owns its own W&B project.
    shell_project = os.environ.get("WANDB_PROJECT")
    load_dotenv()
    os.environ["WANDB_PROJECT"] = shell_project or _DEFAULT_WANDB_PROJECT
    args = parse_args(argv)

    if args.status:
        cmd_status()
        return 0
    if args.collect:
        cmd_collect()
        return 0

    if not (args.id or args.all or args.method):
        print("Nothing to do: pass --id, --all, --method, --status, or --collect.", file=sys.stderr)
        return 2

    targets = resolve_targets(args)

    if args.dry_run:
        for exp in targets:
            params = build_parameter_dict(exp)
            print(f"\n# {exp['id']}")
            print(json.dumps(params, indent=2, default=str))
        return 0

    if args.background:
        launch_background(argv)
        return 0

    queue_start = time.monotonic()
    failures = []
    for exp in targets:
        try:
            ok = run_one(exp, no_wandb=args.no_wandb, require_clean=args.require_clean)
        except Exception as exc:  # preflight / papermill-import errors
            ok = False
            print(color(f"  ✗ ERROR: {exc}", "red"), file=sys.stderr)
        if not ok:
            failures.append(exp["id"])

    if len(targets) > 1:
        n_ok = len(targets) - len(failures)
        total = format_elapsed(time.monotonic() - queue_start)
        line = f"=== Summary: {n_ok}/{len(targets)} succeeded  (total {total}) ==="
        print("\n" + color(line, "red" if failures else "green"))
        if failures:
            print(color("Failed: " + ", ".join(failures), "red"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
