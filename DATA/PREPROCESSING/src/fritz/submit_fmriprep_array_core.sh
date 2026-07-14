#!/usr/bin/env bash
# =============================================================================
# submit_fmriprep_array_core.sh
#
# Sizes and submits the OASIS3/ADNI SLURM array job on CORE
# (src/core/fmriprep_array_oasis_adni.slurm or
# src/core/postprocessing_array_oasis_adni.slurm), without waiting for the
# raw/BIDS rsync to CORE to finish first — fMRIPrep processes subjects
# independently, and each array task waits (bounded) for its own subject to
# land if the transfer hasn't reached it yet (see the .slurm file's
# arrival-wait guard). Meant to be run right after (or while)
# run_fritz_pipeline.sh's push is running in the background.
#
# For --stage fmriprep, the array size is taken from the LOCAL, already
# BIDS-organized subject count (DATA/<COHORT>/BIDS/sub-*) — this is final
# before the rsync even starts, since organize_bids_dataset() runs locally
# first. For --stage postprocessing, the size is taken from the REMOTE
# fMRIPrep output subject count (it depends on fMRIPrep having produced
# something first, so there's no "early" case for it).
#
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the
# repo-root .env file (same convention as push_bold_and_smri_to_core.sh).
#
# Usage:
#   bash submit_fmriprep_array_core.sh [--dataset oasis3|adni|both] [--dry-run]
#                                       [--concurrency 10] [--stage fmriprep|postprocessing]
#                                       [--limit N] [--no-password]
#   bash submit_fmriprep_array_core.sh --status [--dataset oasis3|adni|both]
#
# --no-password forces key-based SSH auth (unsets CORE_PASSWORD after
# sourcing .env) even if CORE_PASSWORD is set there — for when key auth to
# CORE_HOST already works via ~/.ssh/config and sshpass isn't installed.
#
# --limit N mirrors run_fritz_pipeline.sh's --limit smoketest: it points at
# the local "_smoketest"-suffixed BIDS dir (DATA/<COHORT>/BIDS_smoketest) and
# submits with DATASET=<dataset>_smoketest, so the array job reads from and
# writes to the same "_smoketest"-suffixed CORE data/outputs tree that
# run_fritz_pipeline.sh --limit N populated — never the real dataset. N
# itself is unused here (the array size always comes from however many
# subjects actually landed in the smoketest BIDS dir); it only exists so the
# smoketest intent is explicit at the call site, matching run_fritz_pipeline.sh.
# =============================================================================

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
else
    C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""; C_RESET=""
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
CORE_SRC_DIR="${REPO_ROOT}/DATA/PREPROCESSING/src/core"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/submit_array"
LOG_FILE="${LOG_DIR}/submit_array_$(date +%Y%m%d_%H%M%S).log"

# ─── Parse arguments ────────────────────────────────────────────────────
DATASET="both"
DRY_RUN=false
CONCURRENCY=10
STAGE="fmriprep"
STATUS_MODE=false
LIMIT=""
NO_PASSWORD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2,,}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --stage) STAGE="${2,,}"; shift 2 ;;
        --status) STATUS_MODE=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --no-password) NO_PASSWORD=true; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$DATASET" =~ ^(oasis3|adni|both)$ ]] || die "--dataset must be oasis3, adni, or both"
[[ "$STAGE" =~ ^(fmriprep|postprocessing)$ ]] || die "--stage must be fmriprep or postprocessing"

# Matches run_fritz_pipeline.sh's SMOKETEST_SUFFIX: when --limit is set, read
# from/write to the "_smoketest"-suffixed local BIDS dir and remote
# data/outputs trees instead of the real dataset's.
SMOKETEST_SUFFIX=""
if [[ -n "$LIMIT" ]]; then
    SMOKETEST_SUFFIX="_smoketest"
fi

# ─── Credentials ─────────────────────────────────────────────────────────
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    die "${REPO_ROOT}/.env not found. CORE_USER/CORE_HOST (and optionally CORE_PASSWORD) must be set there."
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.env"

# --no-password: force key-based auth even if CORE_PASSWORD is set in .env
# (e.g. when key auth to CORE_HOST is already configured via ~/.ssh/config
# and sshpass isn't installed/wanted). See credentials.md — CORE_PASSWORD is
# documented as optional for exactly this case.
$NO_PASSWORD && unset CORE_PASSWORD

if [[ -z "${CORE_USER:-}" || -z "${CORE_HOST:-}" ]]; then
    die "CORE_USER and CORE_HOST must be set in ${REPO_ROOT}/.env."
fi

# flakhal has no write access to /data2/core-rad-fni/Delcode_faschmit/
# (probed 2026-07-06, see DATA/PREPROCESSING/src/logs/probe_report.txt).
# Everything below lives under the CORE home dir instead — must match
# run_fritz_pipeline.sh's CORE_DEST and push_bold_and_smri_to_core.sh's
# CORE_DEST_ROOT, and src/core/*.slurm's INPUT_BASE/OUTPUT_BASE/WORK_DIR.
#
# CORE data/outputs root (raw + BIDS input) — matches push_bold_and_smri_to_core.sh.
CORE_DEST_ROOT="${CORE_DEST_ROOT:-/home/flakhal/preprocessing/data}"
# CORE outputs root (fMRIPrep/postprocessed derivatives).
CORE_OUTPUTS_ROOT="${CORE_OUTPUTS_ROOT:-/home/flakhal/preprocessing/outputs}"
# Where the .slurm scripts themselves live on CORE (synced here before submitting).
CORE_SCRIPTS_ROOT="${CORE_SCRIPTS_ROOT:-/home/flakhal/preprocessing/scripts}"

SSH_CMD=(ssh "${CORE_USER}@${CORE_HOST}")
RSYNC_RSH="ssh"
if [[ -n "${CORE_PASSWORD:-}" ]]; then
    command -v sshpass &>/dev/null || die "CORE_PASSWORD is set in .env but 'sshpass' is not installed. Install it (apt install sshpass) or unset CORE_PASSWORD and use key-based auth instead."
    SSH_CMD=(sshpass -p "${CORE_PASSWORD}" ssh "${CORE_USER}@${CORE_HOST}")
    RSYNC_RSH="sshpass -p ${CORE_PASSWORD} ssh"
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# dataset key ("oasis3"/"adni") -> repo folder name ("OASIS3"/"ADNI")
folder_name() {
    case "$1" in
        oasis3) echo "OASIS3" ;;
        adni) echo "ADNI" ;;
    esac
}

# ─── Status mode: squeue snapshot, color-coded by task state ────────────
show_status() {
    local dataset_name="$1"
    local job_name="${STAGE}_oasis_adni"
    info "════════════════════════════════════════"
    info "squeue snapshot for job-name=${job_name} (dataset=${dataset_name}), user=${CORE_USER}@${CORE_HOST}"
    info "════════════════════════════════════════"
    local rows
    rows=$("${SSH_CMD[@]}" "squeue -u '${CORE_USER}' -n '${job_name}' --format='%.10i %.9P %.20j %.8T %.10M %.6D %R' --noheader" 2>/dev/null) \
        || { warn "Could not query squeue (no matching jobs, or SSH failed)."; return; }
    if [[ -z "$rows" ]]; then
        warn "No active/pending tasks found for job-name=${job_name}."
        return
    fi
    while IFS= read -r row; do
        case "$row" in
            *RUNNING*)   echo "${C_GREEN}${row}${C_RESET}" ;;
            *PENDING*)   echo "${C_YELLOW}${row}${C_RESET}" ;;
            *FAILED*|*CANCELLED*|*TIMEOUT*) echo "${C_RED}${row}${C_RESET}" ;;
            *COMPLETED*) echo "${C_DIM}${row}${C_RESET}" ;;
            *) echo "$row" ;;
        esac
    done <<< "$rows"
}

# ─── Submit: sync scripts, size array, sbatch ───────────────────────────
submit_dataset() {
    local dataset_key="$1"
    local dataset_name
    dataset_name=$(folder_name "$dataset_key")
    local export_dataset="${dataset_key}${SMOKETEST_SUFFIX}"

    local slurm_file
    if [[ "$STAGE" == "fmriprep" ]]; then
        slurm_file="fmriprep_array_oasis_adni.slurm"
    else
        slurm_file="postprocessing_array_oasis_adni.slurm"
    fi

    info "════════════════════════════════════════"
    info "Submitting ${STAGE} array for ${dataset_name}${SMOKETEST_SUFFIX}"
    info "════════════════════════════════════════"

    local n
    if [[ "$STAGE" == "fmriprep" ]]; then
        local bids_dir="${REPO_ROOT}/DATA/${dataset_name}/BIDS${SMOKETEST_SUFFIX}"
        [[ -d "$bids_dir" ]] || die "${dataset_name}: local BIDS dir not found at ${bids_dir} — run run_fritz_pipeline.sh first to organize it."
        n=$(find "$bids_dir" -maxdepth 1 -type d -name "sub-*" | wc -l)
    else
        local remote_fmriprep_dir="${CORE_OUTPUTS_ROOT}/${export_dataset}/fmriprep"
        n=$("${SSH_CMD[@]}" "find '${remote_fmriprep_dir}' -maxdepth 1 -type d -name 'sub-*' 2>/dev/null | wc -l") \
            || die "${dataset_name}: could not query remote fMRIPrep output count via SSH."
    fi

    if [[ "$n" -eq 0 ]]; then
        warn "${dataset_name}: no subjects found for stage=${STAGE} — skipping."
        return
    fi
    success "${dataset_name}: array size = ${n} (concurrency ${CONCURRENCY})"

    if $DRY_RUN; then
        info "(--dry-run) Would rsync ${CORE_SRC_DIR}/${slurm_file} -> ${CORE_USER}@${CORE_HOST}:${CORE_SCRIPTS_ROOT}/core/"
        info "(--dry-run) Would run: sbatch --array=1-${n}%${CONCURRENCY} --export=DATASET=${export_dataset} ${CORE_SCRIPTS_ROOT}/core/${slurm_file}"
        return
    fi

    # sbatch does not create --output/--error directories itself — the job
    # fails immediately if they're missing, which they will be on a fresh
    # /home/flakhal/preprocessing tree. Must match the .slurm file's
    # hardcoded #SBATCH --output/--error path (a sibling of CORE_SCRIPTS_ROOT).
    local core_logs_dir
    core_logs_dir="$(dirname "$CORE_SCRIPTS_ROOT")/logs/${STAGE}"
    "${SSH_CMD[@]}" "mkdir -p '${CORE_SCRIPTS_ROOT}/core' '${core_logs_dir}'"
    rsync -avz -e "$RSYNC_RSH" "${CORE_SRC_DIR}/${slurm_file}" "${CORE_USER}@${CORE_HOST}:${CORE_SCRIPTS_ROOT}/core/"

    local sbatch_out
    sbatch_out=$("${SSH_CMD[@]}" "sbatch --array=1-${n}%${CONCURRENCY} --export=DATASET=${export_dataset} '${CORE_SCRIPTS_ROOT}/core/${slurm_file}'") \
        || die "${dataset_name}: sbatch submission failed."
    success "${dataset_name}: ${sbatch_out}"
}

# ─── Main ───────────────────────────────────────────────────────────────
MODE_LABEL="Submit"
$STATUS_MODE && MODE_LABEL="Status check"

info "================================================================"
info " ${MODE_LABEL} ${STAGE} array — ${CORE_USER}@${CORE_HOST}"
info " Dataset : $DATASET"
info " Dry-run : $DRY_RUN"
info " Log     : $LOG_FILE"
info "================================================================"

for key in oasis3 adni; do
    if [[ "$DATASET" == "$key" || "$DATASET" == "both" ]]; then
        if $STATUS_MODE; then
            show_status "$(folder_name "$key")"
        else
            submit_dataset "$key"
        fi
    fi
done

success "================================================================"
success " All done!"
success "================================================================"
