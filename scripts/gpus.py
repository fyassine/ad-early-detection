#!/usr/bin/env python3
"""Which GPU box should I launch on? One table, one verdict.

    python3 scripts/gpus.py

Free GPU memory decides, not utilisation percent: utilisation is instantaneous
and swings between kernel launches, while memory tracks what is resident and is
what actually makes a second job fail to start. A box at 100% util with 20 GB
free will happily take another run; a box at 0% util with 1 GB free will not.

Training jobs are read from the shared outputs/ tree rather than over ssh, so
the job counts stay correct even when the other box is unreachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from SHARED import hosts  # noqa: E402
from SHARED.runner_io import active_runs  # noqa: E402

# Every package with its own experiment runner and outputs/ tree.
_PACKAGES = ("CLASSIFIER", "PROGNOSER", "ABI", "BRAINTOKENGT")


def _jobs_by_host() -> dict[str, list[str]]:
    """Live training runs across all packages, grouped by the box running them."""
    jobs: dict[str, list[str]] = {h: [] for h in hosts.TRAINING_HOSTS}
    for pkg in _PACKAGES:
        outputs = _REPO_ROOT / pkg / "outputs"
        if not outputs.is_dir():
            continue
        for run in active_runs(outputs):
            host = hosts._normalise(run.get("host")) or "unknown"
            jobs.setdefault(host, []).append(str(run.get("experiment_id")))
    return jobs


def main() -> int:
    jobs = _jobs_by_host()
    snaps = {h: hosts.gpu_snapshot(h) for h in hosts.TRAINING_HOSTS}
    here = hosts.local_host()

    print(f"{'HOST':<9}{'FREE GPU':>12}{'UTIL':>7}   JOBS")
    print("-" * 52)
    for host in hosts.TRAINING_HOSTS:
        snap = snaps[host]
        label = f"{host}{' *' if host == here else ''}"
        running = jobs.get(host, [])
        if snap is None:
            print(f"{label:<9}{'unreachable':>12}{'-':>7}   -")
            continue
        free_gb = snap["free_mib"] / 1024
        detail = f"{len(running)}"
        if running:
            detail += "  (" + ", ".join(sorted(running)[:2])
            detail += ", ..." if len(running) > 2 else ""
            detail += ")"
        print(f"{label:<9}{free_gb:>9.1f} GB{snap['util_pct']:>6}%   {detail}")

    reachable = [h for h in hosts.TRAINING_HOSTS if snaps[h] is not None]
    if not reachable:
        print("\nNo GPU box could be reached — check the boxes are up and ssh works.")
        return 1

    best = max(reachable, key=lambda h: snaps[h]["free_mib"])
    free_gb = snaps[best]["free_mib"] / 1024
    n = len(jobs.get(best, []))
    load = "idle" if n == 0 else f"{n} run{'s' if n != 1 else ''} already there"
    print(f"\n-> use {best}: {free_gb:.1f} GB free, {load}")
    print(f"   scripts/dispatch.sh --id <exp-id>       # picks {best} automatically")
    print("   (more than 2 ids in one call uses both boxes at once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
