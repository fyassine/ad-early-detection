#!/usr/bin/env bash
# =============================================================================
# run_chain.sh — detached, resumable Fritz-side pipeline chain
#
# Runs Step 1 (organize_bids) → Step 2 (push_bids_to_core) unattended, so the
# push to CORE happens automatically after organise finishes even once the
# operator has closed Claude and dropped their SSH connection.
#
# Surviving logout + the claude-session-guard:
#   Launch with
#     setsid nohup bash run_chain.sh >OUT 2>&1 </dev/null & disown
#   • exe becomes /bin/bash — the guard only kills processes whose /proc/<pid>/exe
#     is the Claude binary, so bash/fslmerge/rsync are never targeted.
#   • setsid puts it in its own session, reparented to PID 1; with
#     KillUserProcesses=no it outlives logout.
#   It calls the pipeline scripts by ABSOLUTE path in the main checkout, so it
#   does not depend on any git worktree persisting.
#
# Scope (chosen 2026-07-18, per explicit request): organize → push to CORE.
# This intentionally couples two steps that pipeline-Fritz-CORE.md keeps
# separate — done here so the push runs after disconnect. If organize fails,
# the push is skipped and the chain stops (no empty/partial push to CORE).
# =============================================================================

set -uo pipefail

REPO_ROOT="/mnt/e/fyassine/ad-early-detection"
FRITZ_DIR="${REPO_ROOT}/DATA/PREPROCESSING/src/fritz"
DATASET="both"

# ─── FSL (organize_bids needs fslmerge/fslnvols/fslval; not on PATH by default)
export FSLDIR=/usr/local/fsl
export PATH="${FSLDIR}/bin:${PATH}"
# shellcheck disable=SC1091
source "${FSLDIR}/etc/fslconf/fsl.sh" 2>/dev/null || true
export FSLOUTPUTTYPE=NIFTI_GZ

# ─── Logging + status markers ────────────────────────────────────────────────
LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/chain"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="${LOG_DIR}/chain_status_${TS}.txt"
LATEST="${LOG_DIR}/chain_status_latest.txt"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
mark() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS_FILE" >>"$LATEST"; }

: >"$LATEST"
SESS="$(ps -o sess= -p $$ 2>/dev/null | tr -d ' ')"
mark "chain:started pid=$$ session=${SESS} dataset=${DATASET} ts=${TS}"

log "================================================================"
log " run_chain.sh — organize → push to CORE (detached)"
log " Dataset : ${DATASET}"
log " PID/sess: $$ / ${SESS}"
log " Status  : ${STATUS_FILE}"
log "================================================================"

# ─── Step 1: organize BIDS (resumable) ───────────────────────────────────────
mark "organize:running"
log "STEP 1: organize_bids.sh --dataset ${DATASET}"
if bash "${FRITZ_DIR}/organize_bids.sh" --dataset "${DATASET}"; then
    mark "organize:done"
    log "STEP 1 complete."
else
    rc=$?
    mark "organize:FAILED rc=${rc}"
    log "STEP 1 FAILED (rc=${rc}); NOT pushing to CORE. Stopping."
    exit "${rc}"
fi

# ─── Step 2: push BIDS → CORE ─────────────────────────────────────────────────
mark "push:running"
log "STEP 2: push_bids_to_core.sh --dataset ${DATASET}"
if bash "${FRITZ_DIR}/push_bids_to_core.sh" --dataset "${DATASET}"; then
    mark "push:done"
    log "STEP 2 complete. BIDS pushed to CORE."
else
    rc=$?
    mark "push:FAILED rc=${rc}"
    log "STEP 2 FAILED (rc=${rc})."
    exit "${rc}"
fi

mark "chain:done"
log "================================================================"
log " CHAIN COMPLETE — organize + push both succeeded."
log " Next (ON CORE): bash .../scripts/core/submit_array.sh \\"
log "                 --dataset <oasis3|adni> --stage fmriprep"
log "================================================================"
