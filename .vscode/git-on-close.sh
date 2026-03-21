#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_FILE=".vscode/git-on-close.log"
cleanup_ran=0
SSH_KEY_PATH="${GIT_ON_CLOSE_SSH_KEY:-/mnt/e/fyassine/.ssh/id_rsa}"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG_FILE"
}

cleanup() {
  if [ "$cleanup_ran" -eq 1 ]; then
    return 0
  fi
  cleanup_ran=1

  log "cleanup triggered"

  # Only auto-commit/push when the current branch is dev.
  local branch=""
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [ "$branch" != "dev" ]; then
    log "skipped: current branch is '$branch'"
    return 0
  fi

  # Skip if this folder is not a git repository.
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "skipped: not inside git work tree"
    return 0
  fi

  # Stage all changes and create commit only when there are staged diffs.
  if ! git add -A; then
    log "error: git add failed"
    return 1
  fi

  if ! git diff --cached --quiet; then
    if git commit -m "auto: save on VS Code close $(date '+%Y-%m-%d %H:%M:%S')"; then
      log "commit created"
    else
      log "error: git commit failed"
      return 1
    fi
  else
    log "no staged changes, no commit"
  fi

  # Push dev branch to origin.
  if [ -f "$SSH_KEY_PATH" ]; then
    export GIT_SSH_COMMAND="ssh -i $SSH_KEY_PATH -o IdentitiesOnly=yes"
    log "using ssh key: $SSH_KEY_PATH"
  else
    log "warning: ssh key not found at $SSH_KEY_PATH"
  fi

  if git push origin dev; then
    log "push origin dev succeeded"
  else
    log "error: push origin dev failed"
    return 1
  fi
}

trap cleanup EXIT INT TERM

if [ "${GIT_ON_CLOSE_RUN_ONCE:-0}" = "1" ]; then
  cleanup
  exit 0
fi

# Keep process alive so cleanup runs when VS Code stops this task.
log "watcher started"
while true; do
  sleep 3600
done
