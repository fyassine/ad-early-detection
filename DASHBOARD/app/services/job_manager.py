"""
job_manager.py — Detached precompute job lifecycle.

Launches app.precompute as a detached subprocess using start_new_session=True
so the computation continues even if the uvicorn server process stops.

NOTE: This keeps the subprocess alive across server restarts (Ctrl+C, crashes)
but NOT across Docker container stops — all container-internal processes receive
SIGKILL on `docker stop`. To survive container restarts, run the server directly
with uvicorn (not in Docker) or mount the cache volume externally.

Public surface
--------------
start_job(csv_path, scan_folders, data_root, cache_root, density) -> str
    Launch (or re-attach to) the precompute job for this dataset. Returns job_id.
get_status(job_id) -> dict
    Read the job's status JSON. Returns {status, stage, progress, ...}.
list_jobs() -> list[dict]
    All jobs whose status files exist under $cache_root/jobs/.
is_running(csv_path, scan_folders) -> bool
    True if a job for this exact (csv, folders) pair is alive on the OS.
cancel_job(job_id) -> bool
    Send SIGTERM to the job process. Returns True if signal was sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import DASHBOARD_CACHE_ROOT, DATA_ROOT


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def _jobs_dir(cache_root: Optional[Path] = None) -> Path:
    d = (cache_root or DASHBOARD_CACHE_ROOT) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def canonical_job_id(csv_path: str, scan_folders: list[str]) -> str:
    """Deterministic job_id from the dataset selection (same as CohortStats cache key)."""
    h = hashlib.sha1()
    h.update(csv_path.encode("utf-8"))
    for f in sorted(scan_folders):
        h.update(b"\x00")
        h.update(f.encode("utf-8"))
    return h.hexdigest()[:20]


def _status_path(job_id: str, cache_root: Optional[Path] = None) -> Path:
    return _jobs_dir(cache_root) / f"{job_id}.json"


def _pid_path(job_id: str, cache_root: Optional[Path] = None) -> Path:
    return _jobs_dir(cache_root) / f"{job_id}.pid"


def _log_path(job_id: str, cache_root: Optional[Path] = None) -> Path:
    return _jobs_dir(cache_root) / f"{job_id}.log"


def _is_pid_alive(pid: int) -> bool:
    """Return True if the OS process with this PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ──────────────────────────────────────────────────────────────────────────── #
# Public API                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def is_running(csv_path: str, scan_folders: list[str],
               cache_root: Optional[Path] = None) -> bool:
    """Return True if a live precompute job for this dataset is running."""
    job_id = canonical_job_id(csv_path, scan_folders)
    pid_file = _pid_path(job_id, cache_root)
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False
    alive = _is_pid_alive(pid)
    if not alive:
        # Stale PID file — clean up
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
    return alive


def get_status(job_id: str, cache_root: Optional[Path] = None) -> dict:
    """Read the job status JSON, augmented with a live PID check."""
    sp = _status_path(job_id, cache_root)
    if not sp.exists():
        return {"job_id": job_id, "status": "unknown"}
    try:
        data = json.loads(sp.read_text())
    except Exception:
        return {"job_id": job_id, "status": "error", "error": "unreadable status file"}

    # Cross-check the PID file so status reflects actual OS state.
    if data.get("status") == "running":
        pid = data.get("pid")
        if pid and not _is_pid_alive(int(pid)):
            data["status"] = "interrupted"
            data["note"] = "Process no longer found — may have been killed."
    return data


def list_jobs(cache_root: Optional[Path] = None) -> list[dict]:
    """Return status for every job in the jobs directory."""
    jd = _jobs_dir(cache_root)
    jobs = []
    for p in sorted(jd.glob("*.json")):
        job_id = p.stem
        jobs.append(get_status(job_id, cache_root))
    return jobs


def start_job(
    csv_path: str,
    scan_folders: list[str],
    data_root: str = DATA_ROOT,
    cache_root: Optional[Path] = None,
    density: float = 0.20,
) -> tuple[str, bool]:
    """
    Ensure a precompute job is running for this dataset.

    Returns ``(job_id, already_running)`` where ``already_running`` is True
    when an existing job was found and not restarted.
    """
    cache_root = cache_root or DASHBOARD_CACHE_ROOT
    job_id = canonical_job_id(csv_path, scan_folders)

    if is_running(csv_path, scan_folders, cache_root):
        # Re-attach the log tailer so progress is visible even if the server
        # was restarted while the job was running.
        log_path = _log_path(job_id, cache_root)
        status_path = _status_path(job_id, cache_root)
        _start_log_tailer(job_id, log_path, status_path)
        return job_id, True

    _launch(
        job_id=job_id,
        csv_path=csv_path,
        scan_folders=scan_folders,
        data_root=data_root,
        cache_root=cache_root,
        density=density,
    )
    return job_id, False


def cancel_job(job_id: str, cache_root: Optional[Path] = None) -> bool:
    """Send SIGTERM to the job process. Returns True if signal was delivered."""
    pid_file = _pid_path(job_id, cache_root)
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


# ──────────────────────────────────────────────────────────────────────────── #
# Real-time log tailer                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

_TERMINAL_STATUSES = {"done", "failed", "cancelled", "interrupted"}


def _tail_log_worker(
    job_id: str,
    log_path: Path,
    status_path: Path,
    poll_interval: float = 0.4,
) -> None:
    """
    Background daemon thread: follows the subprocess log file line-by-line
    and forwards each line to the server's stdout with a ``[precompute]``
    prefix so operators see real-time progress in the uvicorn terminal.

    Stops when the job enters a terminal state (done / failed / cancelled)
    and the log file has been fully drained.
    """
    prefix = f"[precompute:{job_id[:12]}]"
    file_pos = 0

    def _drain() -> None:
        nonlocal file_pos
        if not log_path.exists():
            return
        try:
            with log_path.open("r", errors="replace") as fh:
                fh.seek(file_pos)
                chunk = fh.read()
                if chunk:
                    for line in chunk.splitlines():
                        print(f"{prefix} {line}", flush=True)
                file_pos = fh.tell()
        except OSError:
            pass

    while True:
        _drain()

        # Check whether the job has reached a terminal state.
        status = "running"
        try:
            data = json.loads(status_path.read_text()) if status_path.exists() else {}
            status = data.get("status", "running")
        except Exception:
            pass

        if status in _TERMINAL_STATUSES:
            _drain()  # final drain before exiting
            print(f"{prefix} ── job {status.upper()} ──", flush=True)
            break

        time.sleep(poll_interval)


def _start_log_tailer(job_id: str, log_path: Path, status_path: Path) -> None:
    """Spawn the log-tailer daemon thread (non-blocking)."""
    t = threading.Thread(
        target=_tail_log_worker,
        args=(job_id, log_path, status_path),
        daemon=True,  # dies automatically if the server process exits
        name=f"logtail-{job_id[:8]}",
    )
    t.start()


# ──────────────────────────────────────────────────────────────────────────── #
# Internal launch                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _launch(
    job_id: str,
    csv_path: str,
    scan_folders: list[str],
    data_root: str,
    cache_root: Path,
    density: float,
) -> None:
    """Launch app.precompute as a detached subprocess."""
    # Resolve the DASHBOARD/ directory (parent of this file's app/ package).
    dashboard_dir = Path(__file__).resolve().parents[2]

    log_file = _log_path(job_id, cache_root)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "app.precompute",
        "--data-root", data_root,
        "--csv-path", csv_path,
        "--scan-folders", ",".join(scan_folders),
        "--job-id", job_id,
        "--cache-root", str(cache_root),
        "--density", str(density),
    ]

    # Write the initial status file BEFORE launching so callers can poll
    # immediately after start_job() returns.
    sp = _status_path(job_id, cache_root)
    sp.write_text(json.dumps({
        "job_id": job_id,
        "status": "starting",
        "stage": "queued",
        "progress": 0.0,
        "csv_path": csv_path,
        "scan_folders": scan_folders,
        "density": density,
        "pid": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
    }, indent=2))

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(dashboard_dir),
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # detach from server's signal group
            close_fds=True,
        )

    # Write PID file (precompute.py will also write this, but we write it
    # immediately so is_running() works before the subprocess starts Python).
    _pid_path(job_id, cache_root).write_text(str(proc.pid))

    print(f"[precompute] ── job {job_id[:12]}… started (pid={proc.pid})")
    print(f"[precompute]    csv   : {csv_path}")
    print(f"[precompute]    log   : {log_file}", flush=True)

    # Tail the subprocess log file to the server's own stdout so operators
    # see real-time progress in the uvicorn terminal without needing a
    # separate `tail -f` session.
    status_file = _status_path(job_id, cache_root)
    _start_log_tailer(job_id, log_file, status_file)


# ──────────────────────────────────────────────────────────────────────────── #
# Watched workspaces                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

_WATCHED_FILE = "watched_workspaces.json"


def register_workspace(
    csv_path: str,
    scan_folders: list[str],
    cache_root: Optional[Path] = None,
) -> None:
    """
    Add this (csv, folders) combo to watched_workspaces.json so the server
    can auto-restart its precompute job on the next startup.
    """
    cache_root = cache_root or DASHBOARD_CACHE_ROOT
    path = cache_root / _WATCHED_FILE
    try:
        workspaces: list[dict] = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        workspaces = []

    entry = {"csv_path": csv_path, "scan_folders": sorted(scan_folders)}
    if entry not in workspaces:
        workspaces.append(entry)
        path.write_text(json.dumps(workspaces, indent=2))


def load_watched_workspaces(cache_root: Optional[Path] = None) -> list[dict]:
    cache_root = cache_root or DASHBOARD_CACHE_ROOT
    path = cache_root / _WATCHED_FILE
    try:
        return json.loads(path.read_text()) if path.exists() else []
    except Exception:
        return []
