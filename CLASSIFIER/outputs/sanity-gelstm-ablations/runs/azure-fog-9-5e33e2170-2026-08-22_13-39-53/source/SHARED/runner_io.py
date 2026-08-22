"""
Terminal niceties shared by the CLASSIFIER and PROGNOSER experiment runners.

Colored pass/fail markers, elapsed-time formatting, and a live "heartbeat"
elapsed counter while a long notebook executes. All coloring is a no-op when
stdout is not a TTY (or ``NO_COLOR`` is set), so redirected output and the
per-run ``run.log`` stay free of ANSI escape codes.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from SHARED import hosts

_ANSI = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "white": "\033[37m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def supports_color(stream=None) -> bool:
    """True only for an interactive TTY with ``NO_COLOR`` unset."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def color(text: str, name: str, *, stream=None) -> str:
    """Wrap ``text`` in the ANSI color ``name`` when the stream supports color."""
    if not supports_color(stream):
        return text
    code = _ANSI.get(name)
    return f"{code}{text}{_ANSI['reset']}" if code else text


def format_elapsed(seconds: float | int | None, *, always_show_hours: bool = False) -> str:
    """Format a duration as ``MM:SS`` or ``HH:MM:SS`` (or ``H:MM:SS``). Returns '-' if None/invalid."""
    if seconds is None or not isinstance(seconds, (int, float)):
        return "-"
    seconds_int = max(0, int(round(seconds)))
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    if always_show_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def infer_notebook_duration(notebook_path: str | Path) -> float | None:
    """Extract runtime duration in seconds from an executed notebook JSON."""
    path = Path(notebook_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    # 1. Check top-level papermill duration
    pm_meta = data.get("metadata", {}).get("papermill", {})
    dur = pm_meta.get("duration")
    if isinstance(dur, (int, float)) and dur >= 0:
        return round(float(dur), 1)

    # 2. Check top-level papermill start_time / end_time
    st_str = pm_meta.get("start_time")
    et_str = pm_meta.get("end_time")
    if st_str and et_str:
        try:
            st = datetime.fromisoformat(st_str.replace("Z", "+00:00"))
            et = datetime.fromisoformat(et_str.replace("Z", "+00:00"))
            delta = (et - st).total_seconds()
            if delta >= 0:
                return round(delta, 1)
        except Exception:
            pass

    # 3. Sum cell-level papermill durations or check cell execution timestamps
    cell_durations: List[float] = []
    earliest_start: Optional[datetime] = None
    latest_end: Optional[datetime] = None

    for cell in data.get("cells", []):
        c_pm = cell.get("metadata", {}).get("papermill", {})
        c_dur = c_pm.get("duration")
        if isinstance(c_dur, (int, float)) and c_dur > 0:
            cell_durations.append(float(c_dur))

        exec_meta = cell.get("metadata", {}).get("execution", {})
        busy_str = exec_meta.get("iopub.status.busy") or exec_meta.get("iopub.execute_input")
        idle_str = exec_meta.get("iopub.status.idle") or exec_meta.get("shell.execute_reply")
        if busy_str and idle_str:
            try:
                b_dt = datetime.fromisoformat(busy_str.replace("Z", "+00:00"))
                i_dt = datetime.fromisoformat(idle_str.replace("Z", "+00:00"))
                if earliest_start is None or b_dt < earliest_start:
                    earliest_start = b_dt
                if latest_end is None or i_dt > latest_end:
                    latest_end = i_dt
            except Exception:
                pass

    if cell_durations:
        return round(sum(cell_durations), 1)
    if earliest_start and latest_end:
        delta = (latest_end - earliest_start).total_seconds()
        if delta >= 0:
            return round(delta, 1)

    return None


def is_process_alive(pid: int | None, started_at: str | None = None) -> bool:
    """Check whether a process with the given PID is currently active."""
    if pid is None or not isinstance(pid, int) or pid <= 0:
        return False

    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        if not proc.is_running():
            return False
        if proc.status() in (
            getattr(psutil, "STATUS_ZOMBIE", "zombie"),
            getattr(psutil, "STATUS_DEAD", "dead"),
        ):
            return False
        return True
    except Exception:
        pass

    # Fallback when psutil is unavailable or errors:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def infer_last_active_time(run_dir: str | Path) -> datetime | None:
    """Find the latest modification timestamp among artifacts in a run directory."""
    path = Path(run_dir)
    if not path.is_dir():
        return None
    latest_mtime: float | None = None
    try:
        for item in path.iterdir():
            if item.name == "status.json":
                continue  # ignore status.json itself so we see when actual work was done
            try:
                mtime = item.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
            except Exception:
                pass
    except Exception:
        pass
    if latest_mtime is not None:
        return datetime.fromtimestamp(latest_mtime)
    return None


def reconcile_run_status(
    status_file_or_dir: str | Path,
    status_dict: Optional[Dict[str, Any]] = None,
    *,
    write_disk: bool = True,
) -> Dict[str, Any]:
    """Inspect and reconcile run status, detecting killed/reaped runs and creating fallback summaries.

    If the status dictionary indicates ``state == 'running'`` but its process is no longer alive,
    updates state to ``'killed'``, determines elapsed duration from file modification timestamps,
    updates ``status.json`` on disk, and creates a fallback ``run_summary.json`` if missing.
    """
    path = Path(status_file_or_dir)
    if path.is_dir():
        status_file = path / "status.json"
        run_dir = path
    else:
        status_file = path
        run_dir = path.parent

    if status_dict is None:
        if status_file.is_file():
            try:
                status_dict = json.loads(status_file.read_text())
            except Exception:
                status_dict = {}
        else:
            status_dict = {}

    if not isinstance(status_dict, dict):
        status_dict = {}

    status = dict(status_dict)
    state = status.get("state")

    if state == "running":
        pid = status.get("pid")
        started_at = status.get("started_at")
        if pid is not None:
            alive = is_process_alive(pid, started_at)
        elif started_at:
            try:
                st = datetime.fromisoformat(started_at)
                # Without PID, if started in the last 60s assume active, else dead
                alive = (datetime.now() - st).total_seconds() < 60.0
            except Exception:
                alive = False
        else:
            alive = True

        if alive:
            if started_at:
                try:
                    st = datetime.fromisoformat(started_at)
                    status["duration_seconds"] = round((datetime.now() - st).total_seconds(), 1)
                except Exception:
                    pass
        else:
            # Process is DEAD! Mark as killed and compute actual duration from artifacts
            status["state"] = "killed"
            last_active = infer_last_active_time(run_dir)
            if started_at and last_active:
                try:
                    st = datetime.fromisoformat(started_at)
                    dur = max(0.0, (last_active - st).total_seconds())
                    status["duration_seconds"] = round(dur, 1)
                    status["finished_at"] = last_active.isoformat(timespec="seconds")
                except Exception:
                    pass
            elif started_at and not status.get("duration_seconds"):
                status["finished_at"] = status.get("finished_at") or started_at
                status["duration_seconds"] = 0.0

            if not status.get("error"):
                pid_info = f"PID {pid} " if pid else ""
                status["error"] = f"Process {pid_info}terminated abruptly (OOM kill or SIGKILL)"

            if write_disk and run_dir.is_dir():
                try:
                    # 1. Update status.json
                    status_file.write_text(json.dumps(status, indent=2))

                    # 2. Write or update run_summary.json so collect_results / provenance catches it
                    summary_file = run_dir / "run_summary.json"
                    if not summary_file.is_file():
                        summary_data = {
                            "experiment_id": status.get("experiment_id"),
                            "run_name": status.get("run_name") or run_dir.name,
                            "state": "killed",
                            "status": "killed",
                            "error": status.get("error"),
                            "started_at": status.get("started_at"),
                            "finished_at": status.get("finished_at"),
                            "duration_seconds": status.get("duration_seconds"),
                            "git": {
                                "short_commit": status.get("git_commit"),
                                "dirty": status.get("git_dirty"),
                            },
                            "metrics": {},
                        }
                        summary_file.write_text(json.dumps(summary_data, indent=2))
                    else:
                        try:
                            s_data = json.loads(summary_file.read_text())
                            if "state" not in s_data or s_data.get("state") == "running":
                                s_data["state"] = "killed"
                            if "status" not in s_data:
                                s_data["status"] = "killed"
                            if "error" not in s_data:
                                s_data["error"] = status.get("error")
                            if status.get("duration_seconds") is not None and (
                                "duration_seconds" not in s_data
                                or s_data["duration_seconds"] is None
                            ):
                                s_data["duration_seconds"] = status.get("duration_seconds")
                            summary_file.write_text(json.dumps(s_data, indent=2))
                        except Exception:
                            pass
                except Exception:
                    pass

    return status


def infer_run_duration(run_dir: str | Path) -> float | None:
    """Infer runtime duration in seconds for a run directory.

    Checks ``status.json``, ``run_summary.json``, or executed notebooks.
    """
    path = Path(run_dir)
    if not path.is_dir():
        return None

    # 1. status.json (with automatic dead process reconciliation)
    status_file = path / "status.json"
    if status_file.is_file():
        try:
            status = reconcile_run_status(status_file, write_disk=False)
            dur = status.get("duration_seconds")
            state = status.get("state")
            st_str = status.get("started_at")
            fn_str = status.get("finished_at")

            # Actively running: elapsed time from started_at to now
            if state == "running" and st_str:
                st = datetime.fromisoformat(st_str)
                delta = (datetime.now() - st).total_seconds()
                if delta >= 0:
                    return round(delta, 1)

            if isinstance(dur, (int, float)) and dur >= 0:
                return round(float(dur), 1)

            if st_str and fn_str:
                st = datetime.fromisoformat(st_str)
                fn = datetime.fromisoformat(fn_str)
                delta = (fn - st).total_seconds()
                if delta >= 0:
                    return round(delta, 1)
        except Exception:
            pass

    # 2. run_summary.json
    summary_file = path / "run_summary.json"
    if summary_file.is_file():
        try:
            summary = json.loads(summary_file.read_text())
            dur = summary.get("duration_seconds")
            if isinstance(dur, (int, float)) and dur >= 0:
                return round(float(dur), 1)
        except Exception:
            pass

    # 3. Executed notebooks in run_dir
    for nb in sorted(path.glob("*.ipynb")):
        dur = infer_notebook_duration(nb)
        if dur is not None:
            return dur

    return None


def _fmt_num(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def format_metric_summary(metrics: dict) -> str:
    """One-line ``key value`` rendering of a flat metric dict (floats to 3 dp)."""
    return "  ".join(f"{k} {_fmt_num(v)}" for k, v in metrics.items())


def format_cv_summary(cv: dict) -> str:
    """Render the ``cv.*`` ledger columns as ``N folds — val_auc mean±std …``.

    ``cv`` keys are the un-prefixed ledger names (``n_folds``, ``best_fold``,
    ``val_auc_mean``, ``val_auc_std``, …). Returns ``""`` if there is nothing to
    show.
    """
    if not cv:
        return ""
    n = cv.get("n_folds")
    head = f"{int(n)} folds" if isinstance(n, (int, float)) else "CV"
    bases = sorted({k[: -len("_mean")] for k in cv if k.endswith("_mean")})
    parts = []
    for base in bases:
        mean = cv.get(f"{base}_mean")
        std = cv.get(f"{base}_std")
        if mean is None:
            continue
        parts.append(
            f"{base} {mean:.3f}±{std:.3f}"
            if isinstance(std, (int, float))
            else f"{base} {mean:.3f}"
        )
    tail = ""
    if isinstance(cv.get("best_fold"), (int, float)):
        bv = cv.get("best_val_auc")
        tail = f"  | best fold {int(cv['best_fold'])}"
        if isinstance(bv, (int, float)):
            tail += f" (val_auc {bv:.3f})"
    body = "  ".join(parts)
    return f"{head} — {body}{tail}" if body else f"{head}{tail}"


class Heartbeat:
    """Context manager printing a live ``\\r ⏱  <label> elapsed MM:SS`` line.

    Only active on a color-capable TTY; otherwise a no-op (so background/piped
    runs add nothing to logs). The notebook's own output goes to the run log, so
    this single rewriting line owns the terminal while the body executes.
    """

    def __init__(self, label: str, *, interval: float = 2.0, stream=None):
        self.label = label
        self.interval = interval
        self.stream = stream or sys.stdout
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start = 0.0
        self._active = supports_color(self.stream)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = format_elapsed(time.monotonic() - self._start)
            msg = color(f"\r  ⏱  {self.label} — elapsed {elapsed}", "dim", stream=self.stream)
            self.stream.write(msg)
            self.stream.flush()

    def __enter__(self) -> "Heartbeat":
        self._start = time.monotonic()
        if self._active:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
        if self._active:
            # Clear the heartbeat line so the final result prints cleanly.
            self.stream.write("\r\033[K")
            self.stream.flush()


class RunLifecycle:
    """Context manager for robust run status and provenance tracking during execution.

    Traps termination signals (SIGINT, SIGTERM, SIGHUP) and handles unhandled
    exceptions so that ``status.json`` and ``run_summary.json`` are always written
    before process exit.
    """

    def __init__(
        self,
        run_dir: str | Path,
        exp_id: str,
        run_name: str,
        git_info: Optional[Dict[str, Any]] = None,
        notebook_path: Optional[str | Path] = None,
        extra_status: Optional[Dict[str, Any]] = None,
    ):
        self.run_dir = Path(run_dir)
        self.exp_id = exp_id
        self.run_name = run_name
        self.git_info = git_info or {}
        self.notebook_path = str(notebook_path) if notebook_path else None
        self.extra_status = extra_status or {}
        self.status_path = self.run_dir / "status.json"
        self.summary_path = self.run_dir / "run_summary.json"
        self._start_time = 0.0
        self._started_iso = ""
        self._orig_handlers: Dict[int, Any] = {}
        self._completed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _write_status(self, **fields) -> None:
        status: Dict[str, Any] = {}
        if self.status_path.is_file():
            try:
                status = json.loads(self.status_path.read_text())
            except Exception:
                status = {}
        status.update(fields)
        try:
            self.status_path.write_text(json.dumps(status, indent=2))
        except Exception:
            pass

    def _write_summary(
        self,
        state: str,
        error: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        try:
            summary: Dict[str, Any] = {}
            if self.summary_path.is_file():
                try:
                    summary = json.loads(self.summary_path.read_text())
                except Exception:
                    summary = {}
            summary.setdefault("experiment_id", self.exp_id)
            summary.setdefault("run_name", self.run_name)
            summary["state"] = state
            summary["status"] = state
            if error:
                summary["error"] = error
            if duration is not None:
                summary["duration_seconds"] = duration
            summary.setdefault("started_at", self._started_iso)
            summary["finished_at"] = self._now()
            summary.setdefault(
                "git",
                {
                    "short_commit": self.git_info.get("short_commit") or self.git_info.get("commit"),
                    "dirty": self.git_info.get("dirty"),
                },
            )
            summary.setdefault("metrics", {})
            self.summary_path.write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else f"SIG#{signum}"
        elapsed = round(time.monotonic() - self._start_time, 1)
        state = "interrupted" if signum == signal.SIGINT else "killed"
        err_msg = f"Terminated by signal {sig_name}"
        self._write_status(
            state=state,
            finished_at=self._now(),
            duration_seconds=elapsed,
            error=err_msg,
            signal=sig_name,
            exit_code=128 + signum,
        )
        self._write_summary(state=state, error=err_msg, duration=elapsed)
        orig = self._orig_handlers.get(signum)
        if orig and orig not in (signal.SIG_IGN, signal.SIG_DFL, None):
            orig(signum, frame)
        else:
            sys.exit(128 + signum)

    def __enter__(self) -> "RunLifecycle":
        self._start_time = time.monotonic()
        self._started_iso = self._now()
        init_status = {
            "experiment_id": self.exp_id,
            "run_name": self.run_name,
            "state": "running",
            "pid": os.getpid(),
            # Which box owns this pid. Without it, a --status run on the OTHER
            # box checks this pid against its own process table, finds nothing,
            # and rewrites this healthy run's state to "killed" on shared disk.
            "host": hosts.local_host(),
            "started_at": self._started_iso,
            "git_commit": self.git_info.get("short_commit") or self.git_info.get("commit"),
            "git_dirty": self.git_info.get("dirty"),
            **self.extra_status,
        }
        if self.notebook_path:
            init_status["notebook"] = self.notebook_path
        self._write_status(**init_status)

        # Install signal handlers on main thread
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is None:
                continue
            try:
                self._orig_handlers[sig] = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass

        self._start_heartbeat()
        return self

    def _start_heartbeat(self) -> None:
        """Stamp a heartbeat file so the OTHER box can tell this run is alive.

        The terminal Heartbeat class is a TTY spinner and is inert for the
        background/piped runs that dispatch launches, so it cannot serve as
        liveness evidence. This one writes to the shared tree instead.
        """
        hosts.write_heartbeat(self.run_dir)

        def _beat() -> None:
            while not self._heartbeat_stop.wait(hosts.HEARTBEAT_INTERVAL_SECONDS):
                hosts.write_heartbeat(self.run_dir)

        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._heartbeat_thread = None

    def mark_done(self, duration_seconds: Optional[float] = None) -> None:
        self._completed = True
        elapsed = (
            duration_seconds
            if duration_seconds is not None
            else round(time.monotonic() - self._start_time, 1)
        )
        self._write_status(
            state="done",
            finished_at=self._now(),
            exit_code=0,
            duration_seconds=elapsed,
        )
        if self.summary_path.is_file():
            try:
                summary = json.loads(self.summary_path.read_text())
                summary["duration_seconds"] = elapsed
                summary.setdefault("state", "done")
                summary.setdefault("status", "done")
                self.summary_path.write_text(json.dumps(summary, indent=2))
            except Exception:
                pass

    def mark_failed(
        self,
        error: str,
        duration_seconds: Optional[float] = None,
        notebook_traceback: Optional[str] = None,
        cell: Optional[str] = None,
    ) -> None:
        self._completed = True
        elapsed = (
            duration_seconds
            if duration_seconds is not None
            else round(time.monotonic() - self._start_time, 1)
        )
        fields: Dict[str, Any] = {
            "state": "failed",
            "finished_at": self._now(),
            "duration_seconds": elapsed,
            "error": error,
        }
        if notebook_traceback:
            fields["notebook_traceback"] = notebook_traceback
        if cell:
            fields["cell"] = cell
        self._write_status(**fields)
        self._write_summary(state="failed", error=error, duration=elapsed)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        for sig, handler in self._orig_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

        if exc_type is not None and not self._completed:
            elapsed = round(time.monotonic() - self._start_time, 1)
            if issubclass(exc_type, KeyboardInterrupt):
                state = "interrupted"
                err = "Interrupted by user (KeyboardInterrupt)"
            else:
                state = "failed"
                err = f"{exc_type.__name__}: {exc}"
            self._write_status(
                state=state,
                finished_at=self._now(),
                duration_seconds=elapsed,
                error=err,
            )
            self._write_summary(state=state, error=err, duration=elapsed)

        return False


def render_status_table(
    statuses: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
    stream=None,
    recent_window_seconds: float = 3600.0,
) -> None:
    """Print an auto-aligned and color-coded table of experiment run statuses."""
    if not statuses:
        print("No runs found under outputs/.", file=stream)
        return

    if limit is not None:
        if limit <= 0:
            print("No runs to display (limit=0).", file=stream)
            return
        statuses = statuses[:limit]

    exp_names = [str(s.get("experiment_id") or "") for s in statuses]
    exp_width = max(len("EXPERIMENT"), max((len(name) for name in exp_names), default=0))

    state_names = [str(s.get("state") or "") for s in statuses]
    state_width = max(len("STATE"), 12, max((len(name) for name in state_names), default=0))

    started_width = 19
    dur_width = 10
    git_width = 10

    col_state = "STATE"
    col_exp = "EXPERIMENT"
    col_started = "STARTED"
    col_dur = "DURATION"
    col_git = "GIT"

    header = (
        f"{col_state:<{state_width}} "
        f"{col_exp:<{exp_width}} "
        f"{col_started:<{started_width}} "
        f"{col_dur:<{dur_width}} "
        f"{col_git:<{git_width}} "
        f"RUN"
    )
    print(header, file=stream)
    print("-" * len(header), file=stream)

    now = datetime.now()
    recent_killed_or_stopped_count = 0

    for s in statuses:
        state = str(s.get("state", "?"))
        if state == "done":
            colored_state = color(f"{state:<{state_width}}", "green", stream=stream)
        elif state == "running":
            colored_state = color(f"{state:<{state_width}}", "cyan", stream=stream)
        elif state in ("failed", "killed"):
            colored_state = color(f"{state:<{state_width}}", "red", stream=stream)
        elif state in ("interrupted", "cancelled"):
            colored_state = color(f"{state:<{state_width}}", "dim", stream=stream)
        else:
            colored_state = f"{state:<{state_width}}"

        # Check if this stopped/killed/failed run happened within recent_window_seconds
        if state in ("killed", "interrupted", "cancelled", "failed"):
            end_time = None
            if s.get("finished_at"):
                try:
                    end_time = datetime.fromisoformat(str(s["finished_at"]))
                except Exception:
                    pass
            if end_time is None and s.get("started_at"):
                try:
                    st = datetime.fromisoformat(str(s["started_at"]))
                    dur_sec = s.get("duration_seconds") or 0.0
                    end_time = st + timedelta(seconds=float(dur_sec))
                except Exception:
                    pass
            if end_time is not None:
                age_sec = (now - end_time).total_seconds()
                if 0 <= age_sec <= recent_window_seconds:
                    recent_killed_or_stopped_count += 1

        exp_id = str(s.get("experiment_id", "?"))
        started = str(s.get("started_at", "?")).replace("T", " ")[:19]
        dur = s.get("duration_seconds")
        if dur is None and state == "running" and s.get("started_at"):
            try:
                st = datetime.fromisoformat(s["started_at"])
                dur = (now - st).total_seconds()
            except Exception:
                dur = None
        dur_str = format_elapsed(dur, always_show_hours=True) if dur is not None else "-"
        git = s.get("git_commit")
        git_str = str(git) if git and git != "None" else "-"
        run_name = str(s.get("run_name", "?"))

        print(
            f"{colored_state} "
            f"{exp_id:<{exp_width}} "
            f"{started:<{started_width}} "
            f"{dur_str:<{dur_width}} "
            f"{git_str:<{git_width}} "
            f"{run_name}",
            file=stream,
        )

    if recent_killed_or_stopped_count > 0:
        note = color(
            f"\n  ⚠  {recent_killed_or_stopped_count} run(s) were stopped or killed in the last hour and may need re-launching.",
            "yellow",
            stream=stream,
        )
        print(note, file=stream)


def watch_status_table(
    status_fn,
    *,
    interval: float = 2.0,
    limit: Optional[int] = None,
    stream=None,
    max_iterations: Optional[int] = None,
) -> None:
    """Continuously refresh and update the status table in the terminal.

    Parameters
    ----------
    status_fn : Callable[[], List[Dict[str, Any]]]
        Zero-argument callable returning the latest statuses list.
    interval : float
        Refresh interval in seconds (default: 2.0).
    limit : int, optional
        Maximum number of runs to display in the table.
    stream : TextIO, optional
        Output stream (default: sys.stdout).
    max_iterations : int, optional
        Maximum number of loop iterations before stopping (useful for testing).
    """
    stream = stream or sys.stdout
    iterations = 0
    try:
        while True:
            statuses = status_fn()
            if supports_color(stream):
                # Clear terminal screen and move cursor to top-left
                stream.write("\033[H\033[2J")
                stream.flush()

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            limit_str = f" | showing top {limit}" if limit is not None else ""
            header_note = color(
                f"⏱  Experiment Status — Updated: {now_str} (refresh: {interval:g}s{limit_str} — press Ctrl+C to exit)\n",
                "dim",
                stream=stream,
            )
            print(header_note, file=stream)
            render_status_table(statuses, limit=limit, stream=stream)
            stream.flush()

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        if supports_color(stream):
            stream.write("\n")
            stream.flush()


def follow_run_log(
    run_dir: str | Path,
    *,
    lines: Optional[int] = 50,
    poll_interval: float = 0.5,
    stream=None,
    max_iterations: Optional[int] = None,
) -> None:
    """Stream live logs and status for a single experiment run.

    Parameters
    ----------
    run_dir : str | Path
        Directory of the run (e.g. outputs/<id>/runs/<run_name>/ or outputs/<id>/latest).
    lines : int, optional
        Number of trailing lines to display initially (default: 50). Pass 0 or None to show all.
    poll_interval : float
        Polling interval in seconds when following live logs (default: 0.5).
    stream : file-like, optional
        Output stream (default sys.stdout).
    max_iterations : int, optional
        Maximum poll iterations before stopping (useful for testing).
    """
    stream = stream or sys.stdout
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        print(f"Run directory not found: {run_path}", file=stream)
        return

    status_file = run_path / "status.json"
    log_file = run_path / "run.log"

    def _read_status_dict() -> Dict[str, Any]:
        if status_file.is_file():
            try:
                return reconcile_run_status(status_file, write_disk=False)
            except Exception:
                pass
        return {}

    status = _read_status_dict()
    exp_id = status.get("experiment_id") or run_path.parents[1].name
    run_name = status.get("run_name") or run_path.name
    state = str(status.get("state", "unknown"))
    started_at = str(status.get("started_at", "-")).replace("T", " ")[:19]
    pid = status.get("pid")

    if state == "done":
        colored_state = color(state, "green", stream=stream)
    elif state == "running":
        colored_state = color(state, "cyan", stream=stream)
    elif state in ("interrupted", "cancelled"):
        colored_state = color(state, "dim", stream=stream)
    elif state in ("failed", "killed"):
        colored_state = color(state, "red", stream=stream)
    else:
        colored_state = state

    print(f"=== Experiment: {exp_id} ===", file=stream)
    print(f"  run      : {run_name}", file=stream)
    print(f"  state    : {colored_state}", file=stream)
    print(f"  started  : {started_at}" + (f" (pid {pid})" if pid else ""), file=stream)
    print(f"  log      : {log_file.name}", file=stream)
    print(f"  {'-' * 60}", file=stream)
    stream.flush()

    file_offset = 0
    if log_file.is_file():
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                if lines and len(all_lines) > lines:
                    initial_lines = all_lines[-lines:]
                else:
                    initial_lines = all_lines
                for l in initial_lines:
                    stream.write(l)
                stream.flush()
                file_offset = f.tell()
        except Exception:
            file_offset = 0

    iterations = 0
    try:
        while True:
            if log_file.is_file():
                try:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(file_offset)
                        new_chunk = f.read()
                        if new_chunk:
                            stream.write(new_chunk)
                            stream.flush()
                            file_offset = f.tell()
                except Exception:
                    pass

            status = _read_status_dict()
            current_state = status.get("state")

            if current_state in ("done", "failed", "killed", "interrupted", "cancelled"):
                if log_file.is_file():
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(file_offset)
                            new_chunk = f.read()
                            if new_chunk:
                                stream.write(new_chunk)
                                stream.flush()
                    except Exception:
                        pass

                dur = status.get("duration_seconds")
                dur_str = format_elapsed(dur, always_show_hours=True) if dur is not None else "-"
                print(f"  {'-' * 60}", file=stream)
                if current_state == "done":
                    print(color(f"  ✓ DONE ({dur_str})", "green", stream=stream), file=stream)
                elif current_state in ("failed", "killed"):
                    err = status.get("error") or current_state
                    marker = "✗ KILLED" if current_state == "killed" else "✗ FAILED"
                    print(color(f"  {marker} ({dur_str}) — {err}", "red", stream=stream), file=stream)
                elif current_state in ("interrupted", "cancelled"):
                    print(color(f"  INTERRUPTED ({dur_str})", "dim", stream=stream), file=stream)
                stream.flush()
                break

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(color("\n[Detached from log stream]", "dim", stream=stream), file=stream)
        stream.flush()


