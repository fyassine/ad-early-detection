#!/usr/bin/env python3
"""Report experiment-runner training jobs on the host this runs on.

Used by ``scripts/dispatch.sh`` to decide whether an experiment id is already
being trained somewhere before launching it on the other box. fritz and frieda
share one NFS tree, and ``run_experiment.py`` rewrites ``outputs/<id>/latest``
without locking, so two hosts on the same id would race that symlink.

Matching is done on exact argv, not on a substring of the command line. A
substring match treats ``run_experiment.py --status --watch``, an editor, or a
shell command that merely mentions the id as a live training job, which would
block legitimate launches.

Usage
-----
    python3 runner_jobs.py            # list live runner jobs: "<pid>\t<id>"
    python3 runner_jobs.py <exp_id>   # exit 0 if that id is training, else 1
"""

from __future__ import annotations

import os
import sys


def _argv_of(pid: str) -> list[str]:
    """Return a process's argv, or [] if it is unreadable or already gone."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except (OSError, ProcessLookupError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _experiment_id(argv: list[str]) -> str | None:
    """The --id value if argv is a run_experiment.py training invocation."""
    if len(argv) < 2:
        return None
    if "python" not in os.path.basename(argv[0]):
        return None
    if os.path.basename(argv[1]) != "run_experiment.py":
        return None
    # --status/--collect/--follow are read-only inspectors, not training runs.
    if {"--status", "--collect", "--dry-run"} & set(argv):
        return None
    if "--id" not in argv:
        return None
    idx = argv.index("--id")
    return argv[idx + 1] if idx + 1 < len(argv) else None


def live_jobs() -> list[tuple[str, str]]:
    """(pid, experiment_id) for every runner training job on this host."""
    me = os.getpid()
    jobs: list[tuple[str, str]] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        exp_id = _experiment_id(_argv_of(pid))
        if exp_id is not None:
            jobs.append((pid, exp_id))
    return jobs


def main(argv: list[str]) -> int:
    jobs = live_jobs()
    if not argv:
        for pid, exp_id in jobs:
            print(f"{pid}\t{exp_id}")
        return 0
    if len(argv) > 1:
        raise ValueError(f"expected at most one experiment id, got {argv!r}")
    target = argv[0]
    for pid, exp_id in jobs:
        if exp_id == target:
            print(pid)
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
