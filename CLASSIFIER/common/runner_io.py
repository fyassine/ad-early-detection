"""
Terminal niceties shared by the CLASSIFIER and PROGNOSER experiment runners.

Colored pass/fail markers, elapsed-time formatting, and a live "heartbeat"
elapsed counter while a long notebook executes. All coloring is a no-op when
stdout is not a TTY (or ``NO_COLOR`` is set), so redirected output and the
per-run ``run.log`` stay free of ANSI escape codes.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from types import TracebackType
from typing import Optional, Type

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


def format_elapsed(seconds: float) -> str:
    """Format a duration as ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


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
        parts.append(f"{base} {mean:.3f}±{std:.3f}" if isinstance(std, (int, float)) else f"{base} {mean:.3f}")
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
