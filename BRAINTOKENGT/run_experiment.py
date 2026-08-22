#!/usr/bin/env python3
"""
Experiment runner for the Brain-TokenGT competitor baseline.

Mirrors ``CLASSIFIER/run_experiment.py``: each entry in ``experiments/*.yaml`` is
executed with papermill, its parameters injected from the registry, and its
artifacts written to ``outputs/<id>/runs/<display_name>-<git>-<timestamp>/``.

The registry/config/results helpers are IMPORTED from ``CLASSIFIER.common`` rather
than duplicated, so the two packages cannot drift apart in how they resolve
configs or aggregate results.

Examples
--------
    python run_experiment.py --id braintokengt-smoke --no-wandb
    python run_experiment.py --id braintokengt-delcode-whole-brain
    python run_experiment.py --mode longitudinal
    python run_experiment.py --id <id> --dry-run
    python run_experiment.py --status
    python run_experiment.py --collect
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

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CLASSIFIER.common.experiment_utils import (  # noqa: E402
    build_config,
    build_parameter_dict,
    collect_results,
    find_latest_run,
    find_run_dir,
    load_experiment,
    load_registry,
    read_statuses,
)
from SHARED.provenance import (  # noqa: E402
    capture_git_provenance,
    patch_run_summary,
    snapshot_source_dirs,
)
from SHARED.run_naming import generate_run_name  # noqa: E402
from SHARED.runner_io import (  # noqa: E402
    Heartbeat,
    RunLifecycle,
    assert_not_already_running,
    color,
    follow_run_log,
    format_cv_summary,
    format_elapsed,
    format_metric_summary,
    render_status_table,
    watch_status_table,
)

_REGISTRY = _PACKAGE_ROOT / "experiments"
_OUTPUTS = _PACKAGE_ROOT / "outputs"

# Snapshotted into each run's source/ so a past run can be read back with the exact
# code that produced it. The pristine upstream checkout is included deliberately:
# the equivalence claim is only meaningful against a known upstream revision.
_SOURCE_ROOTS = [
    "BRAINTOKENGT/model",
    "BRAINTOKENGT/adapter.py",
    "BRAINTOKENGT/configs",
    "BRAINTOKENGT/experiments",
    "BRAINTOKENGT/run_experiment.py",
    "BRAINTOKENGT/tests",
    "Brain-TokenGT/model_transformer.py",
    "Brain-TokenGT/model_grcu.py",
    "CLASSIFIER/common",
    "CLASSIFIER/adapters",
    "DATA/DELCODE/src/splitting",
    "SHARED",
]


def load_dotenv(path: Path = _REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def _preflight(exp: dict, require_clean: bool) -> dict:
    # fritz and frieda share this outputs/ tree, so an id may already be
    # training on the OTHER box. Check before any run dir or latest pointer is
    # touched -- two runs of one id race outputs/<id>/latest.
    assert_not_already_running(_OUTPUTS, exp["id"])

    notebook = _PACKAGE_ROOT / exp["notebook"]
    if not notebook.is_file():
        raise FileNotFoundError(f"Experiment {exp['id']!r}: notebook {notebook} not found.")

    upstream = _REPO_ROOT / "Brain-TokenGT" / "model_transformer.py"
    if not upstream.is_file():
        raise FileNotFoundError(
            f"Pristine upstream checkout missing at {upstream}. It is the reference for "
            "the equivalence regression test and is snapshotted into every run; the "
            "'we ran the authors' implementation' claim is unverifiable without it."
        )

    git = capture_git_provenance()
    if git.get("dirty"):
        msg = f"[preflight] git tree is dirty at commit {git.get('short_commit')}."
        if require_clean:
            raise RuntimeError(msg + " Re-run without --require-clean to proceed anyway.")
        print("WARNING:", msg, "Results will record dirty=True.", file=sys.stderr)
    return git


def run_one(exp: dict, *, no_wandb: bool, require_clean: bool) -> bool:
    print(f"\n=== Running experiment: {exp['id']} ({exp['model']} / {exp['mode']}) ===")
    git = _preflight(exp, require_clean)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_git = git.get("short_commit") or "nogit"
    display_name = generate_run_name(_OUTPUTS / exp["id"])
    run_name = f"{display_name}-{short_git}-{timestamp}"
    run_dir = _OUTPUTS / exp["id"] / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _update_latest_pointer(exp["id"], run_dir)

    resolved_config = build_config(exp, _PACKAGE_ROOT)
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2))
    snapshot_source_dirs(run_dir, _SOURCE_ROOTS, repo_root=_REPO_ROOT)

    params = build_parameter_dict(exp, _PACKAGE_ROOT)
    params["RUN_DIR"] = str(run_dir)
    params["RUN_NAME"] = run_name

    input_nb = _PACKAGE_ROOT / exp["notebook"]
    output_nb = run_dir / f"{run_name}.ipynb"
    log_path = run_dir / "run.log"

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        exp_id=exp["id"],
        run_name=run_name,
        git_info=git,
        notebook_path=str(input_nb.relative_to(_PACKAGE_ROOT)),
    )

    with lifecycle:
        if no_wandb or exp.get("wandb") is False:
            os.environ["WANDB_MODE"] = "disabled"
        os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "ad-early-detection"))
        os.environ.setdefault("WANDB_SILENT", "true")

        try:
            import papermill as pm
            from papermill.exceptions import PapermillExecutionError
        except ImportError as exc:
            lifecycle.mark_failed(error=f"papermill not installed: {exc}")
            raise

        print(f"  notebook : {input_nb.relative_to(_PACKAGE_ROOT)}")
        print(f"  run_dir  : {run_dir.relative_to(_PACKAGE_ROOT)}")
        print(f"  log      : {log_path.relative_to(_PACKAGE_ROOT)}")
        t0 = time.monotonic()
        try:
            with open(log_path, "w") as logf, Heartbeat(run_name):
                pm.execute_notebook(
                    str(input_nb),
                    str(output_nb),
                    parameters=params,
                    cwd=str(_PACKAGE_ROOT),
                    kernel_name="python3",
                    progress_bar=False,
                    stdout_file=logf,
                    stderr_file=logf,
                )
        except PapermillExecutionError as exc:
            elapsed = time.monotonic() - t0
            nb_tb = "\n".join(exc.traceback) if exc.traceback else ""
            lifecycle.mark_failed(
                error=f"{exc.ename}: {exc.evalue}",
                duration_seconds=round(elapsed, 1),
                notebook_traceback=nb_tb,
                cell=f"In [{exc.exec_count}]",
            )
            collect_results(_OUTPUTS)
            print(
                color(
                    f"  ✗ FAILED  ({format_elapsed(elapsed)}) — notebook error in cell "
                    f"In [{exc.exec_count}]:",
                    "red",
                ),
                file=sys.stderr,
            )
            if nb_tb:
                from papermill.exceptions import strip_color

                print(f"  {strip_color(nb_tb)}", file=sys.stderr)
            print(f"  Run log : {log_path}", file=sys.stderr)
            return False
        except (Exception, KeyboardInterrupt) as exc:
            elapsed = time.monotonic() - t0
            if isinstance(exc, KeyboardInterrupt):
                lifecycle._write_status(
                    state="interrupted",
                    finished_at=lifecycle._now(),
                    duration_seconds=round(elapsed, 1),
                    error="Interrupted by user (KeyboardInterrupt)",
                )
                lifecycle._write_summary(
                    state="interrupted",
                    error="Interrupted by user (KeyboardInterrupt)",
                    duration=round(elapsed, 1),
                )
                collect_results(_OUTPUTS)
                print(
                    color(f"  INTERRUPTED ({format_elapsed(elapsed)}) — see {log_path}", "dim"),
                    file=sys.stderr,
                )
                return False
            lifecycle.mark_failed(
                error=traceback.format_exc(limit=3),
                duration_seconds=round(elapsed, 1),
            )
            collect_results(_OUTPUTS)
            print(
                color(f"  ✗ FAILED  ({format_elapsed(elapsed)}) — see {log_path}", "red"),
                file=sys.stderr,
            )
            return False

        elapsed = time.monotonic() - t0
        _update_latest_pointer(exp["id"], run_dir)
        lifecycle.mark_done(duration_seconds=round(elapsed, 1))
        rows = collect_results(_OUTPUTS)
        row = next((r for r in rows if r.get("run_dir", "").endswith(run_dir.name)), {})
        cv_summary = {k[len("cv.") :]: v for k, v in row.items() if k.startswith("cv.")}
        metric_summary = {k[len("metric.") :]: v for k, v in row.items() if k.startswith("metric.")}
        print(color(f"  ✓ DONE  ({format_elapsed(elapsed)})", "green"))
        cv_line = format_cv_summary(cv_summary)
        if cv_line:
            print(color(f"     CV:   {cv_line}", "green"))
        if metric_summary:
            print(color(f"     test: {format_metric_summary(metric_summary)}", "green"))
        return True


def _update_latest_pointer(exp_id: str, run_dir: Path) -> None:
    latest = _OUTPUTS / exp_id / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(Path("runs") / run_dir.name)
    except OSError:
        (latest.parent / "latest.txt").write_text(str(run_dir.name))


def launch_background(argv: list[str]) -> None:
    child_args = [a for a in argv if a != "--background"]
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    launch_log = _OUTPUTS / "launch.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args]
    with open(launch_log, "a") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PACKAGE_ROOT),
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"Launched in background (pid {proc.pid}).")
    print(f"  launcher log : {launch_log}")
    print("  track with   : python run_experiment.py --status")


def cmd_status(
    *,
    watch: bool = False,
    interval: float = 2.0,
    limit: int | None = None,
    experiment_id: str | None = None,
) -> None:
    if watch:
        watch_status_table(
            lambda: read_statuses(_OUTPUTS, experiment_id=experiment_id),
            interval=interval,
            limit=limit,
        )
    else:
        statuses = read_statuses(_OUTPUTS, experiment_id=experiment_id, limit=limit)
        render_status_table(statuses, limit=limit)


def cmd_follow(target: str, lines: int | None = None) -> None:
    run_dir = find_run_dir(_OUTPUTS, target)
    if run_dir is None:
        print(f"No run found matching {target!r} under {_OUTPUTS}.", file=sys.stderr)
        return
    follow_run_log(run_dir, lines=lines or 50)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--id", help="Target single experiment with this id.")
    sel.add_argument("--run", help="Target a specific run by name (e.g. crimson-galaxy-4-5e33e2170-2026-08-22_12-54-20).")
    sel.add_argument("--all", action="store_true", help="Run every experiment sequentially.")
    sel.add_argument("--mode", help="Run every experiment with this mode.")
    p.add_argument("--dry-run", action="store_true", help="Print merged parameters and exit.")
    p.add_argument("--background", action="store_true", help="Detach and run in the background.")
    p.add_argument("--status", action="store_true", help="Print a table of runs and exit.")
    p.add_argument(
        "--follow",
        "-f",
        "--tail",
        "--log",
        nargs="?",
        const=True,
        default=False,
        metavar="TARGET",
        dest="follow",
        help="Stream live execution logs for a run name or experiment (e.g. --follow <run_name> or --id <id> --follow).",
    )
    p.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Continuously update the status table in the terminal as experiments run.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds when watching status (default: 2.0).",
    )
    p.add_argument(
        "--limit",
        "-n",
        "--lines",
        type=int,
        default=None,
        help="Limit status table output or initial log lines to N.",
    )
    p.add_argument("--collect", action="store_true", help="Rebuild RESULTS.csv and exit.")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B for this invocation.")
    p.add_argument(
        "--require-clean", action="store_true", help="Refuse to run if the git tree is dirty."
    )
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
    return registry


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    load_dotenv()
    args = parse_args(argv)

    if args.follow:
        target = args.follow if isinstance(args.follow, str) else (args.run or args.id)
        if not target:
            print(
                "Please specify which run or experiment to follow, e.g. --follow <run_name_or_id> or --id <id> --follow.",
                file=sys.stderr,
            )
            return 2
        cmd_follow(target, lines=args.limit)
        return 0

    if args.status or args.watch or (
        args.limit is not None and not (args.id or args.all or args.mode or args.collect)
    ):
        cmd_status(
            watch=args.watch,
            interval=args.interval,
            limit=args.limit,
            experiment_id=args.id,
        )
        return 0
    if args.collect:
        rows = collect_results(_OUTPUTS)
        print(f"Collected {len(rows)} run(s) into {_OUTPUTS / 'RESULTS.csv'}")
        return 0

    if not (args.id or args.all or args.mode):
        print(
            "Nothing to do: pass --id, --all, --mode, --status, --watch, --follow, or --collect.",
            file=sys.stderr,
        )
        return 2

    targets = resolve_targets(args)

    if args.dry_run:
        for exp in targets:
            print(f"\n# {exp['id']}")
            print(json.dumps(build_parameter_dict(exp, _PACKAGE_ROOT), indent=2, default=str))
        return 0

    if args.background:
        launch_background(argv)
        return 0

    queue_start = time.monotonic()
    failures = []
    for exp in targets:
        try:
            ok = run_one(exp, no_wandb=args.no_wandb, require_clean=args.require_clean)
        except Exception as exc:
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
