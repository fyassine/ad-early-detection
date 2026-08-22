#!/usr/bin/env python3
"""Decide which GPU box each experiment should run on.

Thin CLI over ``SHARED.hosts.assign_hosts`` so ``scripts/dispatch.sh`` and any
agent can share one scheduling policy instead of reimplementing it. Queries
free GPU memory on each box, then prints one ``<host>\\t<experiment_id>`` line
per experiment, in launch order.

    python3 scripts/pick_hosts.py exp-a exp-b exp-c
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SHARED import hosts  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        raise ValueError("expected at least one experiment id")
    for host, exp_id in hosts.assign_hosts(argv):
        print(f"{host}\t{exp_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
