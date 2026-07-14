#!/usr/bin/env bash
# =============================================================================
# push_bold_and_smri_to_core.sh
#
# Rsyncs the raw __bold_and_smri__ trees (OASIS3, ADNI) from Fritz to CORE,
# landing at:
#   /home/flakhal/preprocessing/data/raw/OASIS3/__bold_and_smri__
#   /home/flakhal/preprocessing/data/raw/ADNI/__bold_and_smri__
#
# This is separate from run_fritz_pipeline.sh's rsync, which ships the
# *organized BIDS output* to a different CORE path
# (/home/flakhal/preprocessing/data/<dataset>). This script ships the
# raw pre-BIDS source trees.
#
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the
# repo-root .env file. CORE_PASSWORD is optional — if unset, plain `ssh`/
# `rsync` is used (key-based auth); if set, `sshpass` is used for
# non-interactive auth.
#
# Usage:
#   bash push_bold_and_smri_to_core.sh [--dataset oasis3|adni|both] [--dry-run]
# =============================================================================

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

OASIS3_SRC="${REPO_ROOT}/DATA/OASIS3/__bold_and_smri__"
ADNI_SRC="${REPO_ROOT}/DATA/ADNI/__bold_and_smri__"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/push_raw"
LOG_FILE="${LOG_DIR}/push_bold_and_smri_$(date +%Y%m%d_%H%M%S).log"

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

# CORE destination root. flakhal has no write access to
# /data2/core-rad-fni/Delcode_faschmit/ (probed 2026-07-06, see
# DATA/PREPROCESSING/src/logs/probe_report.txt — permission denied on
# everything under that tree). Lands under the CORE home dir instead — check
# `df -h /home` on CORE before a big push; the netapp-backed home mount had
# 8.5T free as of the probe, but there's no per-user quota guarantee.
# Overridable via CORE_DEST_ROOT in .env.
#   -> ${CORE_DEST_ROOT}/OASIS3/__bold_and_smri__, .../ADNI/__bold_and_smri__
CORE_DEST_ROOT="${CORE_DEST_ROOT:-/home/flakhal/preprocessing/data/raw}"

SSH_CMD=(ssh "${CORE_USER}@${CORE_HOST}")
RSYNC_RSH="ssh"
if [[ -n "${CORE_PASSWORD:-}" ]]; then
    command -v sshpass &>/dev/null || {
        echo "ERROR: CORE_PASSWORD is set in .env but 'sshpass' is not installed. Install it (apt install sshpass) or unset CORE_PASSWORD and use key-based auth instead." >&2
        exit 1
    }
    SSH_CMD=(sshpass -p "${CORE_PASSWORD}" ssh "${CORE_USER}@${CORE_HOST}")
    RSYNC_RSH="sshpass -p ${CORE_PASSWORD} ssh"
fi

# ─── Parse arguments ────────────────────────────────────────────────────
DATASET="both"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2,,}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$DATASET" != "oasis3" && "$DATASET" != "adni" && "$DATASET" != "both" ]]; then
    echo "ERROR: --dataset must be oasis3, adni, or both" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# ─── Storage check ──────────────────────────────────────────────────────
check_storage() {
    local src="$1"
    local dataset_name="$2"

    [[ -d "$src" ]] || die "Source not found: $src"

    local required_kb
    required_kb=$(du -sk "$src" | cut -f1)
    local required_gb=$(( (required_kb + 1024 * 1024 - 1) / (1024 * 1024) ))

    log "${dataset_name}: local source size = ${required_gb}G ($src)"

    local avail_kb
    avail_kb=$("${SSH_CMD[@]}" "df -Pk '${CORE_DEST_ROOT}' | tail -1 | awk '{print \$4}'") \
        || die "Could not query free space on CORE (${CORE_USER}@${CORE_HOST})"
    local avail_gb=$(( avail_kb / (1024 * 1024) ))

    log "${dataset_name}: CORE free space at ${CORE_DEST_ROOT} = ${avail_gb}G"

    # This check only guards the raw rsync. Budget the full working set
    # separately: fMRIPrep derivatives + workdir land on the SAME /data2
    # filesystem and dwarf the raw input — plan for ~1 TB total across both
    # datasets (raw ~230 G + fmriprep outputs + workdir), not just this ${required_gb}G.
    # Require at least 10% headroom over the raw transfer size.
    local margin_gb=$(( required_gb / 10 + 1 ))
    if (( avail_gb < required_gb + margin_gb )); then
        die "${dataset_name}: not enough free space on CORE — need ~${required_gb}G (+${margin_gb}G margin) for the RAW push alone, have ${avail_gb}G. Aborting before rsync."
    fi
    log "${dataset_name}: raw-push storage check OK (${avail_gb}G available, ${required_gb}G needed)."
    log "${dataset_name}: NOTE — this does not reserve space for fMRIPrep derivatives/workdir (~1 TB total working set). Verify separately."
}

# ─── Rsync ──────────────────────────────────────────────────────────────
push_dataset() {
    local src="$1"
    local dataset_name="$2"
    local dest_dir="${CORE_DEST_ROOT}/${dataset_name}/__bold_and_smri__"
    local dest="${CORE_USER}@${CORE_HOST}:${dest_dir}/"

    log "════════════════════════════════════════"
    log "Pushing ${dataset_name} __bold_and_smri__ → CORE"
    log "  Source : ${src}/"
    log "  Dest   : ${dest}"
    log "════════════════════════════════════════"

    check_storage "$src" "$dataset_name"

    if $DRY_RUN; then
        log "(--dry-run) Would run: rsync -avuzh --progress -e \"$RSYNC_RSH\" \"$src/\" \"$dest\""
        return
    fi

    "${SSH_CMD[@]}" "mkdir -p '${dest_dir}'"

    rsync -avuzh --progress \
        -e "$RSYNC_RSH" \
        "${src}/" \
        "$dest"

    log "rsync ${dataset_name} complete."
}

# ─── Main ───────────────────────────────────────────────────────────────
log "================================================================"
log " Push __bold_and_smri__ → CORE (${CORE_USER}@${CORE_HOST})"
log " Dataset : $DATASET"
log " Dry-run : $DRY_RUN"
log " Log     : $LOG_FILE"
log "================================================================"

if [[ "$DATASET" == "oasis3" || "$DATASET" == "both" ]]; then
    push_dataset "$OASIS3_SRC" "OASIS3"
fi

if [[ "$DATASET" == "adni" || "$DATASET" == "both" ]]; then
    push_dataset "$ADNI_SRC" "ADNI"
fi

log "================================================================"
log " All done!"
log "================================================================"
