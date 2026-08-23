#!/usr/bin/env bash
#
# Dispatch experiment-runner jobs to fritz (local) or frieda (over ssh).
#
# Both boxes mount the SAME NFS export at the SAME path
# (/mnt/e/fyassine/ad-early-detection), so there is nothing to sync: the repo,
# the .venv, the DELCODE data and outputs/ are one shared tree. A job launched
# on either host writes into the same outputs/<id>/runs/... directory, which is
# why --status / --follow / --collect run from either box and see everything.
#
# The one thing the shared tree does NOT give you is mutual exclusion:
# run_experiment.py rewrites outputs/<id>/latest with no locking, so two hosts
# running the SAME experiment id would race on that symlink. This script
# refuses that case up front rather than letting it corrupt a pointer.
#
# Host choice defaults to --host auto: the box with more FREE GPU MEMORY takes
# the work, and once there are more than 2 experiments in one invocation BOTH
# boxes run simultaneously (dealt round-robin, freer box first) so a sweep
# finishes in roughly half the time. Pass --host fritz|frieda to override.
#
# Usage
#   scripts/dispatch.sh --id exp-a                       # -> freer box
#   scripts/dispatch.sh --id a --id b --id c             # -> both boxes at once
#   scripts/dispatch.sh --host fritz --pkg CLASSIFIER --id some-id -- --no-wandb
#   scripts/dispatch.sh --plan --id a --id b --id c      # show assignment, launch nothing
#   scripts/dispatch.sh --list
#
set -euo pipefail

REPO=/mnt/e/fyassine/ad-early-detection
PY="$REPO/.venv/bin/python"
PKG=BRAINTOKENGT
HOST=auto
PLAN=0
IDS=()
EXTRA=()

die() { echo "dispatch: $*" >&2; exit 1; }

# Run a command on a host: locally when it is this box, otherwise over the
# persistent ssh master socket (ControlPersist, see ~/.ssh/config).
remote() {
    local host="$1" cmd="$2"
    if [[ "$host" == "$(hostname | tr '[:upper:]' '[:lower:]')" ]]; then
        bash -c "$cmd"
    else
        ssh -n -o BatchMode=yes "$host" "bash -c $(printf %q "$cmd")"
    fi
}

# Is experiment <id> already being TRAINED on <host>?
# Delegates to runner_jobs.py, which matches exact argv rather than a substring
# of the command line -- a substring match counts --status/--follow inspectors
# and any shell command mentioning the id as live jobs, blocking real launches.
id_running_on() {
    local host="$1" id="$2"
    remote "$host" "python3 '$REPO/scripts/runner_jobs.py' '$id' >/dev/null 2>&1"
}

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---------------------------------------------------------------- arg parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)  HOST="${2:-}"; shift 2 ;;
        --pkg)   PKG="${2:-}";  shift 2 ;;
        --id)    IDS+=("${2:-}"); shift 2 ;;
        --list)  HOST="__list__"; shift ;;
        --plan)  PLAN=1; shift ;;
        -h|--help) usage 0 ;;
        --)      shift; EXTRA=("$@"); break ;;
        *)       die "unknown argument '$1' (see --help)" ;;
    esac
done

# ------------------------------------------------------------------ --list
if [[ "$HOST" == "__list__" ]]; then
    for h in fritz frieda; do
        echo "=== $h ==="
        if ! remote "$h" 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "no gpu"
             echo "training jobs:"
             jobs=$(python3 '"'$REPO/scripts/runner_jobs.py'"' 2>/dev/null || true)
             if [ -n "$jobs" ]; then echo "$jobs" | sed "s/^/  /"; else echo "  (none)"; fi'; then
            echo "  unreachable"
        fi
    done
    exit 0
fi

# ------------------------------------------------------------- validation
case "$HOST" in
    auto|fritz|frieda) ;;
    *) die "unknown host '$HOST'; expected 'auto', 'fritz' or 'frieda'" ;;
esac
[[ ${#IDS[@]} -gt 0 ]] || die "at least one --id is required"
[[ -d "$REPO/$PKG" ]] || die "package directory not found: $REPO/$PKG"
[[ -x "$PY" ]] || die "interpreter not found: $PY"

# ------------------------------------- refuse duplicate ids on EITHER host
for id in "${IDS[@]}"; do
    for h in fritz frieda; do
        if id_running_on "$h" "$id"; then
            die "experiment '$id' is ALREADY running on $h — two hosts on one id race on outputs/$id/latest"
        fi
    done
done

# ------------------------------------------------------------- assignment
# One "<host>\t<id>" line per experiment. For --host auto the policy lives in
# SHARED/hosts.py (free-GPU ranking + the >2 rule); an explicit host pins all.
ASSIGNMENT=""
if [[ "$HOST" == "auto" ]]; then
    ASSIGNMENT=$("$PY" "$REPO/scripts/pick_hosts.py" "${IDS[@]}") \
        || die "could not choose a host (is either box reachable?)"
else
    for id in "${IDS[@]}"; do
        ASSIGNMENT+="$HOST"$'\t'"$id"$'\n'
    done
    ASSIGNMENT=${ASSIGNMENT%$'\n'}
fi

if [[ "$PLAN" == "1" ]]; then
    echo "planned assignment (nothing launched):"
    echo "$ASSIGNMENT" | while IFS=$'\t' read -r h i; do echo "  $h  <-  $i"; done
    exit 0
fi

# ------------------------------------------------------------------ launch
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
while IFS=$'\t' read -r host id; do
    [[ -n "$host" && -n "$id" ]] || continue
    CMD="'$PY' '$REPO/scripts/launch_background.py' --pkg '$PKG' --id '$id' ${EXTRA[*]:-}"
    OUT=$(remote "$host" "$CMD")
    echo "  [$host] $OUT"
done <<< "$ASSIGNMENT"

echo
echo "follow from either box:"
echo "  cd $REPO/$PKG && $PY run_experiment.py --status --watch"
