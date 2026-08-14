#!/usr/bin/env bash
# =============================================================================
# pull_derivatives_from_core.sh
#
# Rsyncs fMRIPrep + postprocessing derivatives for OASIS3/ADNI back from CORE
# to Fritz, landing at:
#   DATA/OASIS3/derivatives/{fmriprep,postprocessed}/
#   DATA/ADNI/derivatives/{fmriprep,postprocessed}/
#
# This is the mirror image of push_bold_and_smri_to_core.sh — same flag/log/
# credential conventions, but the storage check runs in the opposite
# direction (local free space vs. remote source size) since this direction
# writes to Fritz. Safe to rerun anytime: rsync only transfers new/changed
# files, so this is how you "pull incrementally as sessions finish" — no
# separate watch/daemon mode, just rerun it (optionally under your own
# `watch bash ...` or a cron entry).
#
# If a stage (fmriprep or postprocessed) hasn't produced any remote output
# yet for a dataset, that half is skipped with a warning, not an error.
#
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the
# repo-root .env file. Key-based SSH auth (plain `ssh`/`rsync`) is the
# default and preferred method — CORE_PASSWORD in .env is ignored unless
# --use-password is passed, in which case `sshpass` is used for
# non-interactive password auth.
#
# Usage:
#   bash pull_derivatives_from_core.sh [--dataset oasis3|adni|both] [--dry-run] [--use-password]
# =============================================================================

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_RESET=$'\033[0m'
else
    C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_RESET=""
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
info()    { echo "${C_CYAN}[$(ts)] $*${C_RESET}"; }
success() { echo "${C_GREEN}[$(ts)] $*${C_RESET}"; }
warn()    { echo "${C_YELLOW}[$(ts)] WARN: $*${C_RESET}"; }
error()   { echo "${C_RED}[$(ts)] ERROR: $*${C_RESET}" >&2; }
die()     { error "$*"; exit 1; }

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/pull_derivatives"
LOG_FILE="${LOG_DIR}/pull_derivatives_$(date +%Y%m%d_%H%M%S).log"

# ─── Parse arguments ────────────────────────────────────────────────────
DATASET="both"
DRY_RUN=false
USE_PASSWORD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2,,}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --use-password) USE_PASSWORD=true; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$DATASET" =~ ^(oasis3|adni|both)$ ]] || die "--dataset must be oasis3, adni, or both"

# ─── Credentials ─────────────────────────────────────────────────────────
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    die "${REPO_ROOT}/.env not found. CORE_USER/CORE_HOST (and optionally CORE_PASSWORD) must be set there."
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.env"

if [[ -z "${CORE_USER:-}" || -z "${CORE_HOST:-}" ]]; then
    die "CORE_USER and CORE_HOST must be set in ${REPO_ROOT}/.env."
fi

# CORE outputs root — must match src/core/*.slurm's OUTPUT_BASE. flakhal owns
# /data2/core-rad-fni/flakhal/ (no per-user quota, unlike /home — see
# DATA/PREPROCESSING/src/logs/probe_report.txt for the earlier 2026-07-06
# probe against a colleague's tree).
CORE_OUTPUTS_ROOT="${CORE_OUTPUTS_ROOT:-/data2/core-rad-fni/flakhal/preprocessing/outputs}"

SSH_CMD=(ssh "${CORE_USER}@${CORE_HOST}")
RSYNC_RSH="ssh"
if $USE_PASSWORD; then
    [[ -n "${CORE_PASSWORD:-}" ]] || die "--use-password was passed but CORE_PASSWORD is not set in ${REPO_ROOT}/.env."
    command -v sshpass &>/dev/null || die "CORE_PASSWORD is set in .env but 'sshpass' is not installed. Install it (apt install sshpass) or drop --use-password and use key-based auth instead."
    SSH_CMD=(sshpass -p "${CORE_PASSWORD}" ssh "${CORE_USER}@${CORE_HOST}")
    RSYNC_RSH="sshpass -p ${CORE_PASSWORD} ssh"
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

folder_name() {
    case "$1" in
        oasis3) echo "OASIS3" ;;
        adni) echo "ADNI" ;;
    esac
}

# ─── Storage check (reversed direction: local free space vs. remote size) ─
check_storage() {
    local remote_dir="$1"
    local label="$2"

    local required_kb
    required_kb=$("${SSH_CMD[@]}" "du -sk '${remote_dir}' 2>/dev/null | cut -f1") || { warn "${label}: could not query remote size (rsync will still catch missing dirs)."; return 0; }
    [[ -z "$required_kb" ]] && return 0
    local required_gb=$(( (required_kb + 1024 * 1024 - 1) / (1024 * 1024) ))

    local avail_kb
    avail_kb=$(df -Pk "$REPO_ROOT" | tail -1 | awk '{print $4}')
    local avail_gb=$(( avail_kb / (1024 * 1024) ))

    info "${label}: remote size = ${required_gb}G, local free space = ${avail_gb}G"

    local margin_gb=$(( required_gb / 10 + 1 ))
    if (( avail_gb < required_gb + margin_gb )); then
        die "${label}: not enough local free space — need ~${required_gb}G (+${margin_gb}G margin), have ${avail_gb}G. Aborting before rsync."
    fi
}

# ─── Rsync one stage (fmriprep or postprocessed) for one dataset ────────
pull_stage() {
    local dataset_key="$1"
    local dataset_name="$2"
    local stage="$3"

    local remote_dir="${CORE_OUTPUTS_ROOT}/${dataset_key}/${stage}"
    local local_dir="${REPO_ROOT}/DATA/${dataset_name}/derivatives/${stage}"

    local remote_exists
    remote_exists=$("${SSH_CMD[@]}" "[[ -d '${remote_dir}' ]] && echo yes || echo no")
    if [[ "$remote_exists" != "yes" ]]; then
        warn "${dataset_name}/${stage}: no remote output yet at ${remote_dir} — skipping."
        return
    fi

    info "════════════════════════════════════════"
    info "Pulling ${dataset_name}/${stage} ← CORE"
    info "  Source : ${CORE_USER}@${CORE_HOST}:${remote_dir}/"
    info "  Dest   : ${local_dir}/"
    info "════════════════════════════════════════"

    check_storage "$remote_dir" "${dataset_name}/${stage}"

    if $DRY_RUN; then
        info "(--dry-run) Would run: rsync -avuzh --progress -e \"$RSYNC_RSH\" \"${CORE_USER}@${CORE_HOST}:${remote_dir}/\" \"${local_dir}/\""
        return
    fi

    mkdir -p "$local_dir"
    rsync -avuzh --progress \
        -e "$RSYNC_RSH" \
        "${CORE_USER}@${CORE_HOST}:${remote_dir}/" \
        "${local_dir}/"

    success "${dataset_name}/${stage}: pull complete."
}

# ─── Main ───────────────────────────────────────────────────────────────
info "================================================================"
info " Pull derivatives ← CORE (${CORE_USER}@${CORE_HOST})"
info " Dataset : $DATASET"
info " Dry-run : $DRY_RUN"
info " Log     : $LOG_FILE"
info "================================================================"

for key in oasis3 adni; do
    if [[ "$DATASET" == "$key" || "$DATASET" == "both" ]]; then
        name=$(folder_name "$key")
        pull_stage "$key" "$name" "fmriprep"
        pull_stage "$key" "$name" "postprocessed"
    fi
done

success "================================================================"
success " All done!"
success "================================================================"
