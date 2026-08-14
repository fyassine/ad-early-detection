#!/usr/bin/env bash
# =============================================================================
# push_scripts_to_core.sh
#
# Step 3 of the manual Fritz → CORE pipeline: ship the CORE-side scripts
# (src/core/*.slurm and src/core/submit_array.sh) to CORE and create the
# directories SLURM needs, so you can then SSH in and submit by hand.
#
# Landing at:
#   /data2/core-rad-fni/flakhal/preprocessing/scripts/core/
#
# Also mkdir -p's the job log dirs
# (.../preprocessing/logs/{fmriprep,postprocessing}) — sbatch does NOT create
# the directories named in a job's #SBATCH --output/--error, and the job fails
# immediately if they are missing, which they will be on a fresh tree.
#
# Rerun this whenever a .slurm file or submit_array.sh changes locally; it is
# the only thing that updates CORE's copy. See
# DATA/PREPROCESSING/pipeline-Fritz-CORE.md for the full step sequence.
#
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the
# repo-root .env file. Key-based SSH auth (plain `ssh`/`rsync`) is the
# default and preferred method — CORE_PASSWORD in .env is ignored unless
# --use-password is passed, in which case `sshpass` is used for
# non-interactive password auth.
#
# Usage:
#   bash push_scripts_to_core.sh [--dry-run] [--use-password]
# =============================================================================

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CORE_SRC_DIR="${REPO_ROOT}/DATA/PREPROCESSING/src/core"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/push_scripts"
LOG_FILE="${LOG_DIR}/push_scripts_$(date +%Y%m%d_%H%M%S).log"

# The CORE-side files this ships. The *_v1.slurm scripts are deliberately
# excluded — they are the legacy DELCODE-non-converter versions (per-session
# INPUT_BASE/<session>/sub-* layout), not used for OASIS3/ADNI.
CORE_FILES=(
    "fmriprep_array_oasis_adni.slurm"
    "postprocessing_array_oasis_adni.slurm"
    "submit_array.sh"
)

# ─── Parse arguments ────────────────────────────────────────────────────
DRY_RUN=false
USE_PASSWORD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --use-password) USE_PASSWORD=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Credentials ─────────────────────────────────────────────────────────
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    echo "ERROR: ${REPO_ROOT}/.env not found. CORE_USER/CORE_HOST (and optionally CORE_PASSWORD) must be set there." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.env"

if [[ -z "${CORE_USER:-}" || -z "${CORE_HOST:-}" ]]; then
    echo "ERROR: CORE_USER and CORE_HOST must be set in ${REPO_ROOT}/.env." >&2
    exit 1
fi

# Where the scripts live on CORE, and the log dirs the .slurm files' #SBATCH
# --output/--error directives point at. Both must stay consistent with
# src/core/submit_array.sh's PREP_ROOT and the .slurm files' hardcoded paths.
CORE_SCRIPTS_ROOT="${CORE_SCRIPTS_ROOT:-/data2/core-rad-fni/flakhal/preprocessing/scripts}"
CORE_LOGS_ROOT="${CORE_LOGS_ROOT:-/data2/core-rad-fni/flakhal/preprocessing/logs}"

SSH_CMD=(ssh "${CORE_USER}@${CORE_HOST}")
RSYNC_RSH="ssh"
if $USE_PASSWORD; then
    [[ -n "${CORE_PASSWORD:-}" ]] || {
        echo "ERROR: --use-password was passed but CORE_PASSWORD is not set in ${REPO_ROOT}/.env." >&2
        exit 1
    }
    command -v sshpass &>/dev/null || {
        echo "ERROR: CORE_PASSWORD is set in .env but 'sshpass' is not installed. Install it (apt install sshpass) or drop --use-password and use key-based auth instead." >&2
        exit 1
    }
    SSH_CMD=(sshpass -p "${CORE_PASSWORD}" ssh "${CORE_USER}@${CORE_HOST}")
    RSYNC_RSH="sshpass -p ${CORE_PASSWORD} ssh"
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# ─── Main ───────────────────────────────────────────────────────────────
log "================================================================"
log " Push CORE scripts → CORE (${CORE_USER}@${CORE_HOST})"
log " Dest    : ${CORE_SCRIPTS_ROOT}/core/"
log " Dry-run : $DRY_RUN"
log " Log     : $LOG_FILE"
log "================================================================"

src_paths=()
for f in "${CORE_FILES[@]}"; do
    p="${CORE_SRC_DIR}/${f}"
    [[ -f "$p" ]] || die "Expected CORE file not found: ${p}"
    src_paths+=("$p")
    log "  will ship: ${f}"
done

if $DRY_RUN; then
    log "(--dry-run) Would run: ssh mkdir -p '${CORE_SCRIPTS_ROOT}/core' '${CORE_LOGS_ROOT}/fmriprep' '${CORE_LOGS_ROOT}/postprocessing'"
    log "(--dry-run) Would run: rsync -avz -e \"$RSYNC_RSH\" ${src_paths[*]} ${CORE_USER}@${CORE_HOST}:${CORE_SCRIPTS_ROOT}/core/"
    exit 0
fi

"${SSH_CMD[@]}" "mkdir -p '${CORE_SCRIPTS_ROOT}/core' '${CORE_LOGS_ROOT}/fmriprep' '${CORE_LOGS_ROOT}/postprocessing'" \
    || die "Could not create script/log directories on CORE."
log "Created ${CORE_SCRIPTS_ROOT}/core and ${CORE_LOGS_ROOT}/{fmriprep,postprocessing} on CORE."

rsync -avz -e "$RSYNC_RSH" \
    "${src_paths[@]}" \
    "${CORE_USER}@${CORE_HOST}:${CORE_SCRIPTS_ROOT}/core/"

log "================================================================"
log " All done! Next step — ON CORE:"
log "   ssh ${CORE_USER}@${CORE_HOST}"
log "   bash ${CORE_SCRIPTS_ROOT}/core/submit_array.sh --dataset oasis3 --stage fmriprep"
log "================================================================"
