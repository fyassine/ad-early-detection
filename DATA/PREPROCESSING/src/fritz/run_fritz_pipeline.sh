#!/usr/bin/env bash
# =============================================================================
# run_fritz_pipeline.sh
#
# Runs preprocessing Steps 1–2 on Fritz for the AD Early Detection project.
# Both OASIS3 (DATA/OASIS3/__bold_and_smri__) and ADNI
# (DATA/ADNI/__bold_and_smri__, built by DATA/ADNI/src/unzip/*.py before this
# script runs) already share the same raw layout —
# sub-*/ses-d<NNNN>/{anat,func}/... — so both datasets go through the same
# organize_bids_dataset() step:
#   Step 1.2  Merge fMRI runs (fslmerge)            [both, multi-run sessions]
#   Step 1.3  Remove empty anat/ dirs                [both, safety net]
#   Step 2    Organise to BIDS (incl. copying anat/) [both]
#   Step 2.2  Copy dataset_description.json         [both]
#   Step →    rsync BIDS output to CORE             [both, unless --dry-run]
#
# ADNI's DICOM→NIfTI conversion (dcm2niix) now happens locally, before this
# script ever runs, via DATA/ADNI/src/unzip/{build_visit_baselines,
# scan_zip_manifest,convert_to_bids}.py — this script no longer unzips or
# runs dcm2niix itself.
#
# Usage:
#   bash run_fritz_pipeline.sh [--dataset oasis3|adni|both] [--dry-run]
#
# Requirements (Fritz):
#   - FSL 6.0.7 (already installed; $FSLDIR must be set) — fslmerge/fslnvols
#     are used for both datasets.
#   - SSH alias for CORE — The script uses `HOST` as the SSH hostname. Add
#     this to `~/.ssh/config` on Fritz:
#     ```
#     Host HOST
#         HostName srvcorem2.med.uni-muenchen.de
#         User flakhal
#     ```
#     Ensure key-based SSH auth is set up (`ssh-copy-id flakhal@HOST`).
# =============================================================================

set -euo pipefail

# ─── Configurable paths ──────────────────────────────────────────────────────

# Root of the repository on Fritz (absolute path)
REPO_ROOT="/mnt/e/fyassine/ad-early-detection"

# Raw OASIS3 data (already NIfTI, semi-BIDS)
OASIS3_RAW="${REPO_ROOT}/DATA/OASIS3/__bold_and_smri__"
# BIDS output for OASIS3
OASIS3_BIDS="${REPO_ROOT}/DATA/OASIS3/BIDS"

# Raw ADNI data (NIfTI, semi-BIDS — built by DATA/ADNI/src/unzip/*.py)
ADNI_RAW="${REPO_ROOT}/DATA/ADNI/__bold_and_smri__"
# BIDS output for ADNI
ADNI_BIDS="${REPO_ROOT}/DATA/ADNI/BIDS"

# dataset_description.json template (placed next to this script)
DATASET_DESC_TEMPLATE="$(dirname "$0")/dataset_description.json"

# Logs
LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs"
LOG_FILE="${LOG_DIR}/fritz_pipeline_$(date +%Y%m%d_%H%M%S).log"

# ─── CORE rsync target ───────────────────────────────────────────────────────
CORE_USER="flakhal"
CORE_HOST="HOST"                       # SSH alias; add to ~/.ssh/config
CORE_DEST="/home/flakhal/ad-early-detection/BIDS"

# ─── Parse arguments ─────────────────────────────────────────────────────────
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

# ─── Helpers ─────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

require_tool() {
    command -v "$1" &>/dev/null || die "'$1' not found. Please install it first."
}

# ─── Step 2.2 helper: copy dataset_description.json ──────────────────────────
copy_dataset_description() {
    local bids_root="$1"
    if [[ ! -f "$DATASET_DESC_TEMPLATE" ]]; then
        log "WARNING: dataset_description.json template not found at $DATASET_DESC_TEMPLATE"
        log "         Creating a minimal one automatically."
        cat > "$DATASET_DESC_TEMPLATE" << 'EOF'
{
  "Name": "AD Early Detection",
  "BIDSVersion": "1.8.0",
  "DatasetType": "raw",
  "Authors": ["flakhal"]
}
EOF
    fi
    cp "$DATASET_DESC_TEMPLATE" "${bids_root}/dataset_description.json"
    log "Copied dataset_description.json → ${bids_root}/"
}

# =============================================================================
# SHARED: organize a raw sub-*/ses-*/{anat,func} tree into BIDS output
# =============================================================================
# Both OASIS3 (DATA/OASIS3/__bold_and_smri__) and ADNI
# (DATA/ADNI/__bold_and_smri__) share the identical raw shape:
#   sub-*/ses-d<NNNN>/func/*_task-rest[_run-NN]_bold.{nii.gz,json}
#   sub-*/ses-d<NNNN>/anat/*_T1w[_run-NN].{nii.gz,json}
organize_bids_dataset() {
    local raw_dir="$1"
    local bids_dir="$2"
    local dataset_label="$3"

    log "════════════════════════════════════════"
    log "${dataset_label} pipeline starting"
    log "  Raw source : $raw_dir"
    log "  BIDS output: $bids_dir"
    log "════════════════════════════════════════"

    require_tool fslmerge

    mkdir -p "$bids_dir"

    local n_subjects=0 n_merged=0 n_skipped=0 n_anat_copied=0

    for sub_dir in "$raw_dir"/sub-*; do
        [[ -d "$sub_dir" ]] || continue
        sub_id=$(basename "$sub_dir")

        for ses_dir in "$sub_dir"/ses-*; do
            [[ -d "$ses_dir" ]] || continue
            ses_id=$(basename "$ses_dir")
            func_dir="${ses_dir}/func"
            anat_dir="${ses_dir}/anat"

            local has_func=false has_anat=false

            # ── Step 1.2: detect and merge multiple BOLD runs ─────────────────
            if [[ -d "$func_dir" ]]; then
                out_func="${bids_dir}/${sub_id}/${ses_id}/func"

                # Collect runs sorted; run-01 is often incomplete (few volumes)
                mapfile -t runs < <(find "$func_dir" -name "*_run-*_bold.nii.gz" | sort)
                mapfile -t jsons < <(find "$func_dir" -name "*_run-*_bold.json" | sort)

                if [[ ${#runs[@]} -gt 1 ]]; then
                    # Filter out runs with very few volumes (< 50 TRs = likely localiser)
                    valid_runs=()
                    for r in "${runs[@]}"; do
                        nvols=$(fslnvols "$r" 2>/dev/null || echo 0)
                        if [[ "$nvols" -ge 50 ]]; then
                            valid_runs+=("$r")
                        else
                            log "  Skipping short run ($nvols vols): $(basename "$r")"
                        fi
                    done

                    if [[ ${#valid_runs[@]} -gt 1 ]]; then
                        mkdir -p "$out_func"
                        merged="${out_func}/${sub_id}_${ses_id}_task-rest_bold.nii.gz"
                        log "  Merging ${#valid_runs[@]} runs → $(basename "$merged")"
                        fslmerge -t "$merged" "${valid_runs[@]}"
                        first_json="${jsons[0]}"
                        [[ -f "$first_json" ]] && cp "$first_json" \
                            "${out_func}/${sub_id}_${ses_id}_task-rest_bold.json"
                        ((n_merged++)) || true
                        has_func=true
                    elif [[ ${#valid_runs[@]} -eq 1 ]]; then
                        mkdir -p "$out_func"
                        log "  Only one valid run; copying as-is"
                        cp "${valid_runs[0]}" \
                            "${out_func}/${sub_id}_${ses_id}_task-rest_bold.nii.gz"
                        [[ -f "${jsons[0]}" ]] && cp "${jsons[0]}" \
                            "${out_func}/${sub_id}_${ses_id}_task-rest_bold.json"
                        has_func=true
                    else
                        log "  WARNING: No valid runs found for ${sub_id}/${ses_id}."
                        ((n_skipped++)) || true
                    fi
                elif [[ ${#runs[@]} -eq 1 ]]; then
                    mkdir -p "$out_func"
                    log "  Single run for ${sub_id}/${ses_id}, copying as-is"
                    cp "${runs[0]}" \
                        "${out_func}/${sub_id}_${ses_id}_task-rest_bold.nii.gz"
                    [[ -f "${jsons[0]:-}" ]] && cp "${jsons[0]}" \
                        "${out_func}/${sub_id}_${ses_id}_task-rest_bold.json"
                    has_func=true
                else
                    log "  WARNING: No BOLD runs found in $func_dir."
                    ((n_skipped++)) || true
                fi
            fi

            # ── Step 2: copy anatomical (T1w) files ────────────────────────────
            # Copied as-is (already dcm2niix-converted, run- suffixes already
            # baked into the filename by the source layout) — this used to be
            # silently dropped (an empty anat/ dir was mkdir'd, then deleted by
            # Step 1.3 below); now it's actually carried into BIDS output.
            if [[ -d "$anat_dir" ]]; then
                mapfile -t t1_files < <(find "$anat_dir" -name "*_T1w.nii.gz" | sort)
                if [[ ${#t1_files[@]} -gt 0 ]]; then
                    out_anat="${bids_dir}/${sub_id}/${ses_id}/anat"
                    mkdir -p "$out_anat"
                    for t1 in "${t1_files[@]}"; do
                        cp "$t1" "$out_anat/"
                        json="${t1%.nii.gz}.json"
                        [[ -f "$json" ]] && cp "$json" "$out_anat/"
                        ((n_anat_copied++)) || true
                    done
                    has_anat=true
                fi
            fi

            if $has_func || $has_anat; then
                ((n_subjects++)) || true
            fi
        done
    done

    # ── Step 1.3: remove empty anat dirs (safety net) ─────────────────────────
    log "Step 1.3 — Removing empty anat/ directories in BIDS output..."
    find "$bids_dir" -type d -name "anat" -exec sh -c '
        if [ -z "$(ls -A "$1")" ]; then
            echo "  Removing empty anat: $1"
            rm -rf "$1"
        fi
    ' _ {} \;

    # ── Step 2.2: dataset_description.json ────────────────────────────────────
    copy_dataset_description "$bids_dir"

    log "${dataset_label} done — subjects: $n_subjects, func merged: $n_merged, func skipped: $n_skipped, anat files copied: $n_anat_copied"
}

# =============================================================================
# RSYNC → CORE
# =============================================================================
rsync_to_core() {
    local src="$1"
    local dataset_name="$2"
    local dest="${CORE_USER}@${CORE_HOST}:${CORE_DEST}/${dataset_name}/"

    log "════════════════════════════════════════"
    log "Rsyncing ${dataset_name} BIDS → CORE"
    log "  Source : $src"
    log "  Dest   : $dest"
    log "════════════════════════════════════════"

    if $DRY_RUN; then
        log "(--dry-run) Would run: rsync -avuzh --progress \"$src/\" \"$dest\""
        return
    fi

    # Create the remote directory first (rsync won't create nested dirs)
    ssh "${CORE_USER}@${CORE_HOST}" "mkdir -p ${CORE_DEST}/${dataset_name}"

    rsync -avuzh --progress \
        --exclude="*.zip" \
        "${src}/" \
        "$dest" \
        | tee -a "$LOG_FILE"

    log "rsync ${dataset_name} complete."
}

# =============================================================================
# MAIN
# =============================================================================
log "================================================================"
log " Fritz preprocessing pipeline"
log " Dataset : $DATASET"
log " Dry-run : $DRY_RUN"
log " Log     : $LOG_FILE"
log "================================================================"

if [[ "$DATASET" == "oasis3" || "$DATASET" == "both" ]]; then
    organize_bids_dataset "$OASIS3_RAW" "$OASIS3_BIDS" "OASIS3"
    rsync_to_core "$OASIS3_BIDS" "OASIS3"
fi

if [[ "$DATASET" == "adni" || "$DATASET" == "both" ]]; then
    organize_bids_dataset "$ADNI_RAW" "$ADNI_BIDS" "ADNI"
    rsync_to_core "$ADNI_BIDS" "ADNI"
fi

log "================================================================"
log " All done!"
log "================================================================"
