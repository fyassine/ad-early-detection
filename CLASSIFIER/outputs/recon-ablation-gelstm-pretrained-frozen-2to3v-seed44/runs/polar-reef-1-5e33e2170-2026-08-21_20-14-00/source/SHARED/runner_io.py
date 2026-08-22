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
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

_ANSI = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
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


def format_elapsed(seconds: float | int | None) -> str:
    """Format a duration as ``MM:SS`` (or ``H:MM:SS`` past an hour). Returns '-' if None/invalid."""
    if seconds is None or not isinstance(seconds, (int, float)):
        return "-"
    seconds_int = max(0, int(round(seconds)))
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
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


def infer_run_duration(run_dir: str | Path) -> float | None:
    """Infer runtime duration in seconds for a run directory.

    Checks ``status.json``, ``run_summary.json``, or executed notebooks.
    """
    path = Path(run_dir)
    if not path.is_dir():
        return None

    # 1. status.json
    status_file = path / "status.json"
    if status_file.is_file():
        try:
            status = json.loads(status_file.read_text())
            dur = status.get("duration_seconds")
            if isinstance(dur, (int, float)) and dur >= 0:
                return round(float(dur), 1)
            st_str = status.get("started_at")
            fn_str = status.get("finished_at")
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


def render_status_table(statuses: List[Dict[str, Any]], *, stream=None) -> None:
    """Print an auto-aligned and color-coded table of experiment run statuses."""
    if not statuses:
        print("No runs found under outputs/.", file=stream)
        return

    exp_names = [str(s.get("experiment_id") or "") for s in statuses]
    exp_width = max(len("EXPERIMENT"), max((len(name) for name in exp_names), default=0))

    state_width = 8
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

    for s in statuses:
        state = str(s.get("state", "?"))
        if state == "done":
            colored_state = color(f"{state:<{state_width}}", "green", stream=stream)
        elif state == "running":
            colored_state = color(f"{state:<{state_width}}", "yellow", stream=stream)
        elif state == "failed":
            colored_state = color(f"{state:<{state_width}}", "red", stream=stream)
        else:
            colored_state = f"{state:<{state_width}}"

        exp_id = str(s.get("experiment_id", "?"))
        started = str(s.get("started_at", "?")).replace("T", " ")[:19]
        dur = s.get("duration_seconds")
        dur_str = format_elapsed(dur) if dur is not None else "-"
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
