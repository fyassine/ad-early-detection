# GPU dispatch — two boxes, one shared tree

## The setup

Training runs on two twin workstations, both with a single TITAN RTX (24 GB):

| Host | Address | Notes |
|------|---------|-------|
| `fritz` | 138.245.113.6 | Shared with a colleague whose jobs often pin the GPU |
| `frieda` | 138.245.113.9 | Usually idle |

They mount the **same NFS export at the same path** (`/mnt/e/fyassine/ad-early-detection`),
so the repo, the `.venv`, the data and `outputs/` are one shared tree. There is nothing to
sync and no second venv to maintain — a run launched on either box is visible from both,
which is why `--status`, `--follow` and `--collect` work from whichever box you are on.
Bare `frieda` does not resolve in DNS; `~/.ssh/config` pins the IP and keeps a persistent
multiplexed connection open.

## Choosing a GPU

**Always check which GPU is freer before launching, and prefer that one.** `gpus` answers
that in one line (see below). Free GPU *memory* is the signal, not utilisation percent —
utilisation swings between kernel launches, while memory tracks what is actually resident
and is what makes a second job fail to start.

**Default to using both boxes at once when there are more than 2 experiments to run.**
A seed sweep or an ablation block finishes in roughly half the time; leaving a whole idle
GPU untouched while a sweep runs serially is the waste this rule exists to prevent. For
one or two experiments, put them on the freer box and leave the other clear.

That policy is implemented, not just documented — do not hand-roll it:

```bash
gpus                                              # which box? free GPU + jobs + a verdict
scripts/dispatch.sh --id exp-a                    # -> freer box
scripts/dispatch.sh --id a --id b --id c          # -> both boxes simultaneously
scripts/dispatch.sh --plan --id a --id b --id c   # show the assignment, launch nothing
scripts/dispatch.sh --host fritz --id exp-a       # override when you need a specific box
scripts/dispatch.sh --list                        # raw per-box inventory (gpus is friendlier)
```

`gpus` is a shell alias on both boxes for `.venv/bin/python scripts/gpus.py`; call the
script directly in any non-interactive context, where aliases do not exist.

`--host auto` is the default and calls `SHARED.hosts.assign_hosts`. If you need the policy
from Python, call that function rather than duplicating the rule.

## Never run one experiment id twice

`run_experiment.py` rewrites `outputs/<id>/latest` with **no locking**. The same id
started twice — on one box or across both — leaves that symlink pointing at whichever run
touched it last, so downstream notebooks silently read the wrong run. Split work across
the boxes **by experiment id**, never by running one id in two places.

Every `run_experiment.py` (CLASSIFIER, PROGNOSER, ABI, BRAINTOKENGT) enforces this in
`_preflight` via `assert_not_already_running`, which checks live runs on **both** boxes
before creating any run directory. `dispatch.sh` refuses the same case up front.

## How a run on the other box is detected

`status.json` records the owning `host` alongside the `pid`, and `RunLifecycle` refreshes
a `heartbeat` file in the run directory every 30 s.

- **Own host** — the pid is checked against the process table, as before.
- **Other host** — its pid is meaningless locally, so liveness comes from heartbeat
  freshness on the shared tree. No ssh, so status still works when the other box is down.

A run whose liveness cannot be established is reported `unknown` and treated as **alive**.
Never mark a run dead on a guess: writing `killed` onto a healthy run corrupts it on
shared disk for every host, which is exactly the bug this design replaced. Guessing
"alive" only leaves a stale row that the next reconcile with real evidence clears up.

## Working on the other box

Neither box has `tmux` or `screen`. Detached jobs use `setsid`/`nohup` — `dispatch.sh`
already does this and writes `outputs/<id>/dispatch-<host>-<timestamp>.log`.
