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
# Logs + lifecycle constants                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

# DASHBOARD/ is two levels above app/services/job_manager.py
DASHBOARD_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = DASHBOARD_DIR / "logs"
PRECOMPUTE_LOGS_DIR = LOGS_DIR / "precompute"
JOBS_AUDIT_PATH = LOGS_DIR / "jobs.jsonl"
LATEST_SYMLINK = PRECOMPUTE_LOGS_DIR / "latest.log"
LOG_KEEP = 20

MAX_JOB_AGE_S = int(os.environ.get("PRECOMPUTE_MAX_AGE_S", "1800"))      # 30 min
STALL_THRESHOLD_S = int(os.environ.get("PRECOMPUTE_STALL_S", "300"))     # 5 min
WATCHDOG_INTERVAL_S = int(os.environ.get("PRECOMPUTE_WATCHDOG_S", "60"))

_AUDIT_LOCK = threading.Lock()
_WATCHDOG_STARTED = False


def _ensure_log_dirs() -> None:
    PRECOMPUTE_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _audit(event: str, job_id: str, **extra) -> None:
    """Append one JSON-line lifecycle event to logs/jobs.jsonl."""
    _ensure_log_dirs()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "job_id": job_id,
        **extra,
    }
    line = json.dumps(record, default=str) + "\n"
    with _AUDIT_LOCK:
        try:
            with JOBS_AUDIT_PATH.open("a") as fh:
                fh.write(line)
        except OSError:
            pass  # never let logging crash the caller


def _new_precompute_log_path(job_id: str) -> Path:
    _ensure_log_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PRECOMPUTE_LOGS_DIR / f"{ts}_{job_id}.log"


def _update_latest_symlink(target: Path) -> None:
    try:
        if LATEST_SYMLINK.is_symlink() or LATEST_SYMLINK.exists():
            LATEST_SYMLINK.unlink()
        LATEST_SYMLINK.symlink_to(target.name)  # relative within precompute/
    except OSError:
        pass


def _rotate_precompute_logs(keep: int = LOG_KEEP) -> None:
    """Delete the oldest precompute logs, keeping only the most recent ``keep``."""
    if not PRECOMPUTE_LOGS_DIR.exists():
        return
    files = [
        p for p in PRECOMPUTE_LOGS_DIR.iterdir()
        if p.is_file() and not p.is_symlink() and p.name != "latest.log"
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


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
    """
    Resolve the active log path for ``job_id``. Reads ``log_path`` from the
    status JSON if present (set by ``_launch`` to a fresh timestamped file in
    ``logs/precompute/``). Falls back to the legacy location under
    ``.cache/jobs/<id>.log`` for in-flight jobs predating the change.
    """
    sp = _status_path(job_id, cache_root)
    if sp.exists():
        try:
            data = json.loads(sp.read_text())
            recorded = data.get("log_path")
            if recorded:
                return Path(recorded)
        except Exception:
            pass
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
        _audit("cancelled", job_id, pid=pid, reason="user SIGTERM")
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


# ──────────────────────────────────────────────────────────────────────────── #
# Runaway/stall enforcement                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

def _kill_pid(pid: int, grace_s: float = 5.0) -> str:
    """SIGTERM, wait up to grace_s, escalate to SIGKILL. Returns final state."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return "gone"
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return "terminated"
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return "terminated"
    return "killed"


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def sweep_runaway_jobs(cache_root: Optional[Path] = None) -> dict:
    """
    Scan all .pid files and kill anything that exceeds MAX_JOB_AGE_S or has
    not updated its status JSON in STALL_THRESHOLD_S. Returns a summary dict.
    Safe to call repeatedly (idempotent).
    """
    jd = _jobs_dir(cache_root)
    summary = {"checked": 0, "killed_runaway": 0, "killed_stale": 0, "cleaned_stale_pid": 0}
    now = datetime.now(timezone.utc)
    for pid_file in jd.glob("*.pid"):
        summary["checked"] += 1
        job_id = pid_file.stem
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            summary["cleaned_stale_pid"] += 1
            continue
        if not _is_pid_alive(pid):
            pid_file.unlink(missing_ok=True)
            summary["cleaned_stale_pid"] += 1
            _audit("cleaned_stale_pid", job_id, pid=pid, reason="pid file orphaned (process already gone)")
            continue

        sp = _status_path(job_id, cache_root)
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text())
        except Exception:
            continue
        if str(data.get("status", "")).lower() not in {"starting", "running"}:
            continue

        started = _parse_iso(data.get("started_at"))
        age_s = (now - started).total_seconds() if started else None
        try:
            stall_s = time.time() - sp.stat().st_mtime
        except OSError:
            stall_s = None

        if age_s is not None and age_s > MAX_JOB_AGE_S:
            outcome = _kill_pid(pid)
            pid_file.unlink(missing_ok=True)
            _audit("killed_runaway", job_id, pid=pid, age_s=int(age_s),
                   stage=data.get("stage"), outcome=outcome,
                   reason=f"exceeded MAX_JOB_AGE_S={MAX_JOB_AGE_S}")
            summary["killed_runaway"] += 1
            continue

        if stall_s is not None and stall_s > STALL_THRESHOLD_S:
            outcome = _kill_pid(pid)
            pid_file.unlink(missing_ok=True)
            _audit("killed_stale", job_id, pid=pid, stall_s=int(stall_s),
                   stage=data.get("stage"), outcome=outcome,
                   reason=f"status JSON unchanged for {int(stall_s)}s "
                          f"(>STALL_THRESHOLD_S={STALL_THRESHOLD_S})")
            summary["killed_stale"] += 1
    return summary


def _watchdog_loop(cache_root: Path) -> None:
    while True:
        try:
            sweep_runaway_jobs(cache_root)
        except Exception as e:
            print(f"[watchdog] sweep error: {e}", flush=True)
        time.sleep(WATCHDOG_INTERVAL_S)


def start_watchdog(cache_root: Optional[Path] = None) -> None:
    """Spawn the watchdog daemon thread (idempotent — only one per process)."""
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:
        return
    _WATCHDOG_STARTED = True
    cr = cache_root or DASHBOARD_CACHE_ROOT
    t = threading.Thread(
        target=_watchdog_loop, args=(cr,),
        daemon=True, name="precompute-watchdog",
    )
    t.start()
    print(f"[watchdog] started (interval={WATCHDOG_INTERVAL_S}s, "
          f"max_age={MAX_JOB_AGE_S}s, stall={STALL_THRESHOLD_S}s)", flush=True)


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
            event = "finished" if status == "done" else status
            _audit(event, job_id, status=status)
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
    log_file = _new_precompute_log_path(job_id)
    _rotate_precompute_logs()

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
    # immediately after start_job() returns. ``log_path`` lets re-attach
    # callers find the active log without searching.
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
        "log_path": str(log_file),
    }, indent=2))

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(DASHBOARD_DIR),
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # detach from server's signal group
            close_fds=True,
        )

    # Write PID file (precompute.py will also write this, but we write it
    # immediately so is_running() works before the subprocess starts Python).
    _pid_path(job_id, cache_root).write_text(str(proc.pid))

    _update_latest_symlink(log_file)
    _audit("started", job_id, pid=proc.pid, log_path=str(log_file),
           csv_path=csv_path, scan_folders=scan_folders)

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
