#!/usr/bin/env bash
# =============================================================================
# push_bids_to_core.sh
#
# Step 2 of the manual Fritz → CORE pipeline: rsync the BIDS trees organized by
# organize_bids.sh (DATA/<COHORT>/BIDS) from Fritz to CORE, landing at the
# fMRIPrep INPUT_BASE that src/core/fmriprep_array_oasis_adni.slurm reads:
#   /data2/core-rad-fni/flakhal/preprocessing/data/oasis3
#   /data2/core-rad-fni/flakhal/preprocessing/data/adni
#
# This must finish before submitting the fMRIPrep array on CORE — the array job
# no longer waits for subjects to arrive, it fails loudly if they are missing.
# See DATA/PREPROCESSING/pipeline-Fritz-CORE.md for the full step sequence.
#
# This is separate from push_bold_and_smri_to_core.sh, which ships the raw
# pre-BIDS source trees to a different CORE path (.../data/raw/<COHORT>/).
#
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the
# repo-root .env file. Key-based SSH auth (plain `ssh`/`rsync`) is the
# default and preferred method — CORE_PASSWORD in .env is ignored unless
# --use-password is passed, in which case `sshpass` is used for
# non-interactive password auth.
#
# Usage:
#   bash push_bids_to_core.sh [--dataset oasis3|adni|both] [--dry-run]
#                             [--limit N] [--use-password]
#
# --limit N mirrors organize_bids.sh's --limit smoketest: it reads from the
# "_smoketest"-suffixed local BIDS dir (DATA/<COHORT>/BIDS_smoketest) and ships
# it to the matching "_smoketest" CORE dataset dir, never the real dataset. N
# itself is unused here (whatever landed in the smoketest BIDS dir is what gets
# pushed); it exists so the smoketest intent is explicit at the call site,
# matching organize_bids.sh.
# =============================================================================

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/push_bids"
LOG_FILE="${LOG_DIR}/push_bids_$(date +%Y%m%d_%H%M%S).log"

# ─── Parse arguments ────────────────────────────────────────────────────
DATASET="both"
DRY_RUN=false
LIMIT=""
USE_PASSWORD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2,,}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --use-password) USE_PASSWORD=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$DATASET" != "oasis3" && "$DATASET" != "adni" && "$DATASET" != "both" ]]; then
    echo "ERROR: --dataset must be oasis3, adni, or both" >&2
    exit 1
fi

# Matches organize_bids.sh's SMOKETEST_SUFFIX.
SMOKETEST_SUFFIX=""
if [[ -n "$LIMIT" ]]; then
    SMOKETEST_SUFFIX="_smoketest"
fi

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

# CORE destination root for BIDS input. flakhal owns /data2/core-rad-fni/flakhal/
# (75T filesystem, no per-user quota — unlike /home, which enforces one; see
# DATA/PREPROCESSING/src/logs/probe_report.txt for the earlier 2026-07-06 probe
# against a colleague's tree). Must match src/core/*.slurm's INPUT_BASE and
# src/core/submit_array.sh's PREP_ROOT. Overridable via CORE_DEST in .env.
#   -> ${CORE_DEST}/oasis3, ${CORE_DEST}/adni
CORE_DEST="${CORE_DEST:-/data2/core-rad-fni/flakhal/preprocessing/data}"

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

# dataset key ("oasis3"/"adni") -> repo folder name ("OASIS3"/"ADNI")
folder_name() {
    case "$1" in
        oasis3) echo "OASIS3" ;;
        adni) echo "ADNI" ;;
    esac
}

# ─── Storage check ──────────────────────────────────────────────────────
check_storage() {
    local src="$1"
    local dataset_name="$2"

    local required_kb
    required_kb=$(du -sk "$src" | cut -f1)
    local required_gb=$(( (required_kb + 1024 * 1024 - 1) / (1024 * 1024) ))

    log "${dataset_name}: local BIDS size = ${required_gb}G ($src)"

    local avail_kb
    avail_kb=$("${SSH_CMD[@]}" "df -Pk '${CORE_DEST}' | tail -1 | awk '{print \$4}'") \
        || die "Could not query free space on CORE (${CORE_USER}@${CORE_HOST})"
    local avail_gb=$(( avail_kb / (1024 * 1024) ))

    log "${dataset_name}: CORE free space at ${CORE_DEST} = ${avail_gb}G"

    # This check only guards the BIDS rsync. Budget the full working set
    # separately: fMRIPrep derivatives + workdir land on the same /data2
    # filesystem and dwarf the BIDS input — plan for ~1 TB total across both
    # datasets, not just this ${required_gb}G.
    # Require at least 10% headroom over the transfer size.
    local margin_gb=$(( required_gb / 10 + 1 ))
    if (( avail_gb < required_gb + margin_gb )); then
        die "${dataset_name}: not enough free space on CORE — need ~${required_gb}G (+${margin_gb}G margin) for the BIDS push alone, have ${avail_gb}G. Aborting before rsync."
    fi
    log "${dataset_name}: BIDS-push storage check OK (${avail_gb}G available, ${required_gb}G needed)."
    log "${dataset_name}: NOTE — this does not reserve space for fMRIPrep derivatives/workdir (~1 TB total working set). Verify separately."
}

# ─── Rsync ──────────────────────────────────────────────────────────────
push_dataset() {
    local dataset_key="$1"
    local dataset_name
    dataset_name=$(folder_name "$dataset_key")

    local src="${REPO_ROOT}/DATA/${dataset_name}/BIDS${SMOKETEST_SUFFIX}"
    local export_dataset="${dataset_key}${SMOKETEST_SUFFIX}"
    local dest_dir="${CORE_DEST}/${export_dataset}"
    local dest="${CORE_USER}@${CORE_HOST}:${dest_dir}/"

    # Fail loudly rather than shipping an empty tree: the array job on CORE
    # sizes itself from what actually landed, so an empty push silently
    # produces a zero-subject run instead of an error.
    [[ -d "$src" ]] || die "${dataset_name}: BIDS dir not found at ${src} — run organize_bids.sh --dataset ${dataset_key}${LIMIT:+ --limit $LIMIT} first."
    local n_subjects
    n_subjects=$(find "$src" -maxdepth 1 -mindepth 1 -type d -name "sub-*" | wc -l)
    (( n_subjects > 0 )) || die "${dataset_name}: no sub-* directories under ${src} — organize_bids.sh produced nothing to push."

    log "════════════════════════════════════════"
    log "Pushing ${dataset_name}${SMOKETEST_SUFFIX} BIDS → CORE"
    log "  Source   : ${src}/ (${n_subjects} subjects)"
    log "  Dest     : ${dest}"
    log "════════════════════════════════════════"

    check_storage "$src" "$dataset_name"

    if $DRY_RUN; then
        log "(--dry-run) Would run: rsync -avuzh --progress -e \"$RSYNC_RSH\" \"$src/\" \"$dest\""
        return
    fi

    "${SSH_CMD[@]}" "mkdir -p '${dest_dir}'"

    rsync -avuzh --progress \
        -e "$RSYNC_RSH" \
        --exclude="*.zip" \
        "${src}/" \
        "$dest"

    log "rsync ${dataset_name}${SMOKETEST_SUFFIX} complete (${n_subjects} subjects)."
}

# ─── Main ───────────────────────────────────────────────────────────────
log "================================================================"
log " Push BIDS → CORE (${CORE_USER}@${CORE_HOST})"
log " Dataset : $DATASET"
log " Dry-run : $DRY_RUN"
log " Log     : $LOG_FILE"
log "================================================================"

for key in oasis3 adni; do
    if [[ "$DATASET" == "$key" || "$DATASET" == "both" ]]; then
        push_dataset "$key"
    fi
done

log "================================================================"
log " All done! Next step: bash push_scripts_to_core.sh, then submit"
log " the array ON CORE with src/core/submit_array.sh."
log "================================================================"
