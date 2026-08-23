"""Training-host inventory and cross-host run liveness.

fritz and frieda are twin GPU boxes that mount the SAME NFS export at the SAME
path, so ``outputs/`` is one shared tree: a run launched on either box is
visible from both. That is what makes two-GPU training practical, and it is
also what breaks the naive liveness check.

``status.json`` records a ``pid``. A pid is only meaningful on the host that
produced it, so checking it locally against a run that is executing on the
other box is a category error: the pid is absent from the local process table,
the run looks dead, and ``reconcile_run_status`` rewrites a HEALTHY run's state
to ``killed`` on shared disk. Every runner shares that code path, so this
module is the single place that knows which host a run belongs to.

Liveness is therefore resolved per host:

* **own host** -- check the process table directly, as before;
* **other host** -- trust the on-disk heartbeat that ``RunLifecycle`` refreshes
  every ``HEARTBEAT_INTERVAL_SECONDS`` onto the shared tree. No ssh, so status
  still works when the other box is unreachable or its sshd is down.

Anything that cannot be established returns ``"unknown"`` rather than
``"dead"``. Callers must not mark a run killed on ``"unknown"`` -- guessing
"dead" corrupts a live run's status, while guessing "alive" merely leaves a
stale row that the next reconcile with real evidence clears up.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

# The two lab GPU boxes. Names are lowercase; real hostnames are "FRiTZ" and
# "FRiEDA", so every comparison goes through _normalise().
TRAINING_HOSTS: tuple[str, ...] = ("fritz", "frieda")

# RunLifecycle refreshes the heartbeat this often; a run is presumed dead once
# its heartbeat is this stale. The gap is deliberately wide (5 missed beats):
# NFS attribute caching and a loaded box both delay a write, and a false "dead"
# is far more damaging than a slow "alive".
HEARTBEAT_INTERVAL_SECONDS = 30.0
HEARTBEAT_STALE_AFTER_SECONDS = 150.0

HEARTBEAT_FILENAME = "heartbeat"

Liveness = Literal["alive", "dead", "unknown"]


def _normalise(host: str | None) -> str:
    """Lowercase, strip any domain suffix: 'FRiEDA.local' -> 'frieda'."""
    if not host:
        return ""
    return str(host).strip().lower().split(".")[0]


def local_host() -> str:
    """This machine's short hostname, normalised."""
    return _normalise(socket.gethostname())


def is_local(host: str | None) -> bool:
    """Does ``host`` name the machine this process runs on?"""
    return bool(host) and _normalise(host) == local_host()


def other_hosts() -> list[str]:
    """Known training hosts that are not this one."""
    return [h for h in TRAINING_HOSTS if not is_local(h)]


def write_heartbeat(run_dir: str | Path, when: datetime | None = None) -> None:
    """Stamp the run's heartbeat file so other hosts can see it is alive.

    Best-effort: a failed heartbeat must never take down a training run, and a
    missed beat only costs liveness resolution, which degrades to "unknown".
    """
    when = when or datetime.now()
    try:
        path = Path(run_dir) / HEARTBEAT_FILENAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(when.isoformat(timespec="seconds"))
        os.replace(tmp, path)  # atomic, so a reader never sees a partial stamp
    except OSError:
        pass


def read_heartbeat(run_dir: str | Path) -> datetime | None:
    """The run's last heartbeat, or None if it has none or it is unreadable."""
    path = Path(run_dir) / HEARTBEAT_FILENAME
    try:
        return datetime.fromisoformat(path.read_text().strip())
    except (OSError, ValueError):
        return None


def heartbeat_liveness(
    run_dir: str | Path,
    *,
    now: datetime | None = None,
    stale_after: float = HEARTBEAT_STALE_AFTER_SECONDS,
) -> Liveness:
    """Judge a remote run purely from its heartbeat freshness."""
    beat = read_heartbeat(run_dir)
    if beat is None:
        # Either a run that predates heartbeats, or one that has not written its
        # first beat yet. No evidence either way -- never claim it is dead.
        return "unknown"
    age = (now or datetime.now()) - beat
    return "dead" if age.total_seconds() > stale_after else "alive"


def run_liveness(
    status: dict,
    run_dir: str | Path,
    *,
    pid_alive,
    now: datetime | None = None,
) -> Liveness:
    """Is the run described by ``status`` still executing, on whichever host owns it?

    ``pid_alive`` is injected (rather than imported) to keep this module free of
    a circular import with runner_io, which owns the process-table check.
    """
    host = status.get("host")
    if host and not is_local(host):
        return heartbeat_liveness(run_dir, now=now)
    # Own host, or a run so old it never recorded one: the pid is meaningful here.
    pid = status.get("pid")
    if pid is None:
        return "unknown"
    return "alive" if pid_alive(pid, status.get("started_at")) else "dead"


def _ssh_available() -> bool:
    return shutil.which("ssh") is not None


def run_on(host: str, command: str, *, timeout: float = 15.0) -> subprocess.CompletedProcess:
    """Run a shell command on ``host`` -- directly if local, over ssh otherwise.

    ssh uses BatchMode so an unreachable or unauthenticated host fails fast
    instead of blocking on a password prompt.
    """
    # Only ever dispatch to a box in the known inventory. Guarding here rather
    # than at each call site means no caller can turn a stray status.json field
    # or CLI argument into an ssh target.
    if _normalise(host) not in TRAINING_HOSTS:
        raise ValueError(
            f"unknown training host {host!r}; expected one of {', '.join(TRAINING_HOSTS)}"
        )
    if is_local(host):
        argv = ["bash", "-c", command]
    else:
        if not _ssh_available():
            raise RuntimeError(f"cannot reach {host}: no ssh client on {local_host()}")
        argv = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def gpu_free_mib(host: str, *, timeout: float = 15.0) -> int | None:
    """Free GPU memory on ``host`` in MiB, or None if it cannot be determined.

    Free memory is the scheduling signal rather than utilisation percent:
    utilisation is instantaneous and swings wildly between kernel launches,
    while memory tracks what is actually resident and is what makes a second
    job fail to start.
    """
    snap = gpu_snapshot(host, timeout=timeout)
    return None if snap is None else snap["free_mib"]


def gpu_snapshot(host: str, *, timeout: float = 15.0) -> dict | None:
    """Free/total GPU memory (MiB) and utilisation (%) for ``host``.

    Returns None if the box cannot be queried at all, so callers can report it
    as unreachable rather than as an idle GPU with zero memory in use.
    """
    query = (
        "nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    try:
        proc = run_on(host, query, timeout=timeout)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        total_s, used_s, util_s = (
            p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")
        )
        total, used = int(total_s), int(used_s)
    except ValueError:
        return None
    return {
        "host": _normalise(host),
        "total_mib": total,
        "used_mib": used,
        "free_mib": total - used,
        "util_pct": int(util_s),
    }


def rank_by_free_gpu(candidates: tuple[str, ...] | list[str] = TRAINING_HOSTS) -> list[str]:
    """Reachable hosts, most free GPU memory first.

    Hosts whose GPU cannot be queried (box down, sshd refusing, no nvidia-smi)
    are dropped rather than ranked last: scheduling work onto a box we cannot
    even talk to just fails later and more confusingly.
    """
    measured = [(h, gpu_free_mib(h)) for h in candidates]
    reachable = [(h, free) for h, free in measured if free is not None]
    reachable.sort(key=lambda pair: pair[1], reverse=True)
    return [h for h, _ in reachable]


def assign_hosts(
    experiment_ids: list[str],
    *,
    parallel_threshold: int = 2,
    ranked: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Decide which box runs each experiment: [(host, experiment_id), ...].

    Policy:

    * up to ``parallel_threshold`` experiments -- put them all on the box with
      the most free GPU memory, leaving the other one clear for interactive
      work or a colleague's job;
    * more than that -- use BOTH boxes at once, dealing round-robin starting
      from the freer one, so a long sweep finishes in roughly half the time.

    Nothing here is host-count-specific: with one reachable box everything
    lands on it, which is exactly the degraded behaviour we want.
    """
    if not experiment_ids:
        return []
    ranked = ranked if ranked is not None else rank_by_free_gpu()
    if not ranked:
        raise RuntimeError(
            "No training host could be reached (GPU query failed on "
            f"{', '.join(TRAINING_HOSTS)}). Check the boxes are up and ssh works."
        )
    if len(experiment_ids) <= parallel_threshold:
        return [(ranked[0], exp_id) for exp_id in experiment_ids]
    return [(ranked[i % len(ranked)], exp_id) for i, exp_id in enumerate(experiment_ids)]
