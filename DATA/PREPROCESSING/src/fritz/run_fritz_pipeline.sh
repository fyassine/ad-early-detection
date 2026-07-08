#!/usr/bin/env bash
# =============================================================================
# run_fritz_pipeline.sh
#
# Runs preprocessing Steps 1–2 on Fritz for the AD Early Detection project:
#   Step 1.1  DICOM → NIfTI  (dcm2niix)            [ADNI only]
#   Step 1.2  Merge fMRI runs (fslmerge)            [OASIS3 only, multi-run]
#   Step 1.3  Remove empty anat/ dirs               [both]
#   Step 2    Organise to BIDS                      [both]
#   Step 2.1  Rename BIDS folders if needed         [both]
#   Step 2.2  Copy dataset_description.json         [both]
#   Step →    rsync BIDS output to CORE             [both, unless --dry-run]
#
# Usage:
#   bash run_fritz_pipeline.sh [--dataset oasis3|adni|both] [--dry-run]
#
# Requirements (Fritz):
#   - dcm2niix  (for ADNI; `sudo apt install dcm2niix` or conda)
#   - FSL 6.0.7 (already installed; $FSLDIR must be set)
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

# ADNI: flat directory of DICOM zip files
ADNI_DICOM_FLAT="${REPO_ROOT}/DATA/ADNI/__dicom_zips_flat__"
# Intermediate unpacked ADNI DICOMs
ADNI_DICOM_UNPACKED="${REPO_ROOT}/DATA/ADNI/__dicom_unpacked__"
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
# ██████   █████  ███████ ██  ███████     ██████  ██████  ███████ ██████
# ██   ██ ██   ██ ██      ██  ██          ██   ██ ██   ██ ██      ██   ██
# ██████  ███████ ███████ ██  ███████     ██████  ██████  █████   ██████
# ██   ██ ██   ██      ██ ██       ██     ██      ██   ██ ██      ██
# ██████  ██   ██ ███████ ██  ███████     ██      ██   ██ ███████ ██
# =============================================================================
run_oasis3() {
    log "════════════════════════════════════════"
    log "OASIS3 pipeline starting"
    log "  Raw source : $OASIS3_RAW"
    log "  BIDS output: $OASIS3_BIDS"
    log "════════════════════════════════════════"

    require_tool fslmerge

    mkdir -p "$OASIS3_BIDS"

    # ── Step 1.2 + 2: iterate over subjects / sessions ────────────────────────
    # OASIS3 __bold_and_smri__ structure:
    #   sub-OAS3XXXX/ses-dXXXX/func/
    #     sub-OAS3XXXX_ses-dXXXX_task-rest_run-01_bold.nii.gz
    #     sub-OAS3XXXX_ses-dXXXX_task-rest_run-02_bold.nii.gz   (optional)
    #     ...

    local n_subjects=0 n_merged=0 n_skipped=0

    for sub_dir in "$OASIS3_RAW"/sub-*; do
        [[ -d "$sub_dir" ]] || continue
        sub_id=$(basename "$sub_dir")                          # sub-OAS3XXXX

        for ses_dir in "$sub_dir"/ses-*; do
            [[ -d "$ses_dir" ]] || continue
            ses_id=$(basename "$ses_dir")                      # ses-dXXXX
            func_dir="${ses_dir}/func"
            [[ -d "$func_dir" ]] || continue

            # Destination BIDS directories
            out_func="${OASIS3_BIDS}/${sub_id}/${ses_id}/func"
            out_anat="${OASIS3_BIDS}/${sub_id}/${ses_id}/anat"
            mkdir -p "$out_func" "$out_anat"

            # ── Step 1.2: detect and merge multiple BOLD runs ─────────────────
            # Collect runs sorted; run-01 is often incomplete (few volumes)
            mapfile -t runs < <(find "$func_dir" -name "*_run-*_bold.nii.gz" | sort)
            mapfile -t jsons < <(find "$func_dir" -name "*_run-*_bold.json" | sort)

            if [[ ${#runs[@]} -gt 1 ]]; then
                # Filter out runs with very few volumes (< 50 TRs = likely localiser)
                # Use first valid run's JSON for the merged file
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
                    merged="${out_func}/${sub_id}_${ses_id}_task-rest_bold.nii.gz"
                    log "  Merging ${#valid_runs[@]} runs → $(basename "$merged")"
                    fslmerge -t "$merged" "${valid_runs[@]}"
                    # Copy JSON sidecar from first valid run
                    first_json="${jsons[0]}"
                    [[ -f "$first_json" ]] && cp "$first_json" \
                        "${out_func}/${sub_id}_${ses_id}_task-rest_bold.json"
                    ((n_merged++)) || true
                elif [[ ${#valid_runs[@]} -eq 1 ]]; then
                    log "  Only one valid run; copying as-is"
                    cp "${valid_runs[0]}" \
                        "${out_func}/${sub_id}_${ses_id}_task-rest_bold.nii.gz"
                    [[ -f "${jsons[0]}" ]] && cp "${jsons[0]}" \
                        "${out_func}/${sub_id}_${ses_id}_task-rest_bold.json"
                else
                    log "  WARNING: No valid runs found for ${sub_id}/${ses_id}, skipping."
                    ((n_skipped++)) || true
                    continue
                fi
            elif [[ ${#runs[@]} -eq 1 ]]; then
                log "  Single run for ${sub_id}/${ses_id}, copying as-is"
                cp "${runs[0]}" \
                    "${out_func}/${sub_id}_${ses_id}_task-rest_bold.nii.gz"
                [[ -f "${jsons[0]:-}" ]] && cp "${jsons[0]}" \
                    "${out_func}/${sub_id}_${ses_id}_task-rest_bold.json"
            else
                log "  WARNING: No BOLD runs found in $func_dir, skipping."
                ((n_skipped++)) || true
                continue
            fi

            ((n_subjects++)) || true
        done
    done

    # ── Step 1.3: remove empty anat dirs ──────────────────────────────────────
    log "Step 1.3 — Removing empty anat/ directories in BIDS output..."
    find "$OASIS3_BIDS" -type d -name "anat" -exec sh -c '
        if [ -z "$(ls -A "$1")" ]; then
            echo "  Removing empty anat: $1"
            rm -rf "$1"
        fi
    ' _ {} \;

    # ── Step 2.2: dataset_description.json ────────────────────────────────────
    copy_dataset_description "$OASIS3_BIDS"

    log "OASIS3 done — subjects processed: $n_subjects, merged: $n_merged, skipped: $n_skipped"
}

# =============================================================================
# █████  ██████  ███    ██ ██
# ██   ██ ██   ██ ████   ██ ██
# ███████ ██   ██ ██ ██  ██ ██
# ██   ██ ██   ██ ██  ██ ██ ██
# ██   ██ ██████  ██   ████ ██
# =============================================================================
run_adni() {
    log "════════════════════════════════════════"
    log "ADNI pipeline starting"
    log "  DICOM zips : $ADNI_DICOM_FLAT"
    log "  Unpacked   : $ADNI_DICOM_UNPACKED"
    log "  BIDS output: $ADNI_BIDS"
    log "════════════════════════════════════════"

    require_tool dcm2niix

    mkdir -p "$ADNI_DICOM_UNPACKED" "$ADNI_BIDS"

    # ── Step 1.1a: unzip DICOMs ───────────────────────────────────────────────
    # Filename convention: <SiteID>_S_<SubjectID>_<SeriesID>.zip
    # e.g. 002_S_1261_1270025.zip  →  subject 002_S_1261, series 1270025

    log "Step 1.1 — Unpacking DICOM zips from ${ADNI_DICOM_FLAT}/ ..."
    for zip_file in "$ADNI_DICOM_FLAT"/*.zip; do
        [[ -f "$zip_file" ]] || continue
        basename_noext=$(basename "$zip_file" .zip)   # e.g. 002_S_1261_1270025

        # Extract subject part (everything before last underscore = series ID)
        subject_raw="${basename_noext%_*}"             # e.g. 002_S_1261
        series_id="${basename_noext##*_}"              # e.g. 1270025

        unzip_dir="${ADNI_DICOM_UNPACKED}/${subject_raw}/${series_id}"
        if [[ -d "$unzip_dir" ]]; then
            log "  Already unpacked: ${subject_raw}/${series_id}, skipping."
            continue
        fi
        mkdir -p "$unzip_dir"
        log "  Unzipping $(basename "$zip_file") → ${subject_raw}/${series_id}/"
        unzip -q "$zip_file" -d "$unzip_dir"
    done

    # ── Step 1.1b: dcm2niix per subject ───────────────────────────────────────
    log "Step 1.1 — Converting DICOMs to NIfTI with dcm2niix..."

    local n_subjects=0 n_failed=0

    for sub_raw_dir in "$ADNI_DICOM_UNPACKED"/*/; do
        [[ -d "$sub_raw_dir" ]] || continue
        subject_raw=$(basename "$sub_raw_dir")  # e.g. 002_S_1261

        # Map ADNI subject ID → BIDS sub-ID
        # Convention: replace _ with nothing and remove site prefix
        # 002_S_1261 → sub-ADNI002S1261
        sub_bids="sub-ADNI${subject_raw//_/}"

        out_func="${ADNI_BIDS}/${sub_bids}/ses-01/func"
        out_anat="${ADNI_BIDS}/${sub_bids}/ses-01/anat"
        mkdir -p "$out_func" "$out_anat"

        log "  Converting ${subject_raw} → ${sub_bids} ..."

        # dcm2niix writes all series into a temp dir; we then sort by type
        tmp_nii="${ADNI_DICOM_UNPACKED}/${subject_raw}/__nifti__"
        mkdir -p "$tmp_nii"

        if ! dcm2niix \
                -z y \
                -f "%n_%p_%s" \
                -o "$tmp_nii" \
                "$sub_raw_dir" 2>>"$LOG_FILE"; then
            log "  WARNING: dcm2niix failed for $subject_raw"
            ((n_failed++)) || true
            continue
        fi

        # ── Step 2: sort NIfTI files into BIDS structure ──────────────────────
        # Look for resting-state BOLD (fMRI) and T1w anatomical files
        # ADNI naming varies, but keywords are reliable:
        # Confirmed ADNI series names (full zip scan of all 732 zips, 2026-07-06):
        #   ALL zips in __dicom_zips_flat__ are fMRI ONLY — no T1w/sMRI present.
        #   Confirmed BOLD series names:
        #     Axial_rsfMRI__Eyes_Open_          Axial_rsfMRI__EYES_OPEN_
        #     Axial_rsFMRI_Eyes_Open            Axial_rsfMRI__Eyes_Open__-phase_P_to_A
        #     Axial_MB_rsfMRI__Eyes_Open_       Axial_MB_rsfMRI_AP
        #     Axial_MB_rsfMRI__Eyes_Open____straight_no_angle
        #     Axial_fcMRI__Eyes_Open_           Axial_fcMRI__EYES_OPEN_
        #     Axial_fcMRI_0_angle__EYES_OPEN_   Axial_fcMRI
        #     Axial_RESTING_fcMRI__EYES_OPEN_
        #     Resting_State_fMRI                Extended_Resting_State_fMRI
        #   T1w: NOT AVAILABLE in this download. fMRIPrep must run with
        #        --fs-no-reconall or use a study-specific template.

        local bold_file t1_file

        # Broad regex to catch all confirmed fMRI series names above
        bold_file=$(find "$tmp_nii" -maxdepth 1 -name "*.nii.gz" \
            | grep -iE "rsfmri|rsfMRI|fcmri|fmri|resting|bold|rest" | head -n 1 || true)
        # T1w: not expected in current download; grep kept for future compatibility
        t1_file=$(find "$tmp_nii" -maxdepth 1 -name "*.nii.gz" \
            | grep -iE "mprage|t1w|smri|sagittal" | head -n 1 || true)

        if [[ -n "$bold_file" ]]; then
            cp "$bold_file" \
                "${out_func}/${sub_bids}_ses-01_task-rest_bold.nii.gz"
            json="${bold_file%.nii.gz}.json"
            [[ -f "$json" ]] && cp "$json" \
                "${out_func}/${sub_bids}_ses-01_task-rest_bold.json"
            log "    BOLD  → ${sub_bids}_ses-01_task-rest_bold.nii.gz"
        else
            log "    WARNING: No BOLD file identified for $subject_raw"
        fi

        if [[ -n "$t1_file" ]]; then
            cp "$t1_file" \
                "${out_anat}/${sub_bids}_ses-01_T1w.nii.gz"
            json="${t1_file%.nii.gz}.json"
            [[ -f "$json" ]] && cp "$json" \
                "${out_anat}/${sub_bids}_ses-01_T1w.json"
            log "    T1w   → ${sub_bids}_ses-01_T1w.nii.gz"
        else
            log "    WARNING: No T1w file identified for $subject_raw"
        fi

        ((n_subjects++)) || true
    done

    # ── Step 1.3: remove empty anat dirs ──────────────────────────────────────
    log "Step 1.3 — Removing empty anat/ directories..."
    find "$ADNI_BIDS" -type d -name "anat" -exec sh -c '
        if [ -z "$(ls -A "$1")" ]; then
            echo "  Removing empty anat: $1"
            rm -rf "$1"
        fi
    ' _ {} \;

    # ── Step 2.2: dataset_description.json ────────────────────────────────────
    copy_dataset_description "$ADNI_BIDS"

    log "ADNI done — subjects processed: $n_subjects, failed: $n_failed"
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
    run_oasis3
    rsync_to_core "$OASIS3_BIDS" "OASIS3"
fi

if [[ "$DATASET" == "adni" || "$DATASET" == "both" ]]; then
    run_adni
    rsync_to_core "$ADNI_BIDS" "ADNI"
fi

log "================================================================"
log " All done!"
log "================================================================"
