#!/usr/bin/env bash
# =============================================================================
# organize_bids.sh
#
# Step 1 of the manual Fritz → CORE pipeline: organise raw NIfTI trees into
# BIDS, locally on Fritz. This script is purely local — it never touches CORE.
# Pushing the result is a separate, explicit step (push_bids_to_core.sh), so
# the organized tree can be inspected before it ships. See
# DATA/PREPROCESSING/pipeline-Fritz-CORE.md for the full step sequence.
#
# Both OASIS3 (DATA/OASIS3/__bold_and_smri__) and ADNI
# (DATA/ADNI/__bold_and_smri__, built by DATA/ADNI/src/unzip/*.py before this
# script runs) already share the same raw layout —
# sub-*/ses-d<NNNN>/{anat,func}/... — so both datasets go through the same
# organize_bids_dataset() step:
#   Step 1.2  Merge fMRI runs (fslmerge)            [both, multi-run sessions]
#   Step 1.3  Remove empty anat/ dirs                [both, safety net]
#   Step 2    Organise to BIDS (incl. copying anat/) [both]
#   Step 2.2  Copy dataset_description.json         [both]
#
# Output: DATA/<COHORT>/BIDS[_smoketest]/sub-*/ses-*/{anat,func}/
#
# ADNI's DICOM→NIfTI conversion (dcm2niix) happens locally, before this script
# ever runs, via DATA/ADNI/src/unzip/{build_visit_baselines,scan_zip_manifest,
# convert_to_bids}.py — this script does not unzip or run dcm2niix itself.
#
# Usage:
#   bash organize_bids.sh [--dataset oasis3|adni|both] [--limit N]
#
# --limit N organizes only the first N subjects (sorted) into a separate
# "_smoketest"-suffixed local BIDS dir (e.g. DATA/OASIS3/BIDS_smoketest), so a
# smoke test never touches the real dataset's output. push_bids_to_core.sh
# takes the same --limit N to ship it to the matching "_smoketest" CORE tree.
#
# Requirements (Fritz):
#   - FSL 6.0.7 (already installed; $FSLDIR must be set) — fslmerge/fslnvols/
#     fslval are used for both datasets.
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
LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/organize_bids"
LOG_FILE="${LOG_DIR}/organize_bids_$(date +%Y%m%d_%H%M%S).log"

# ─── Parse arguments ─────────────────────────────────────────────────────────
DATASET="both"
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2,,}"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# When --limit is set, this is a smoke test: organize only the first N
# subjects (sorted) into a separate "_smoketest"-suffixed local BIDS dir, so it
# never mixes with the real dataset's output.
SMOKETEST_SUFFIX=""
if [[ -n "$LIMIT" ]]; then
    SMOKETEST_SUFFIX="_smoketest"
fi

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

# Extract the BIDS task label from a *_bold.nii.gz filename. Falls back to
# "rest" for the handful of legacy files that carry no _task- entity, so they
# still group together instead of being dropped.
bold_task_label() {
    local base; base=$(basename "$1")
    if [[ "$base" == *_task-* ]]; then
        base=${base#*_task-}
        printf '%s' "${base%%_*}"
    else
        printf 'rest'
    fi
}

# Whitespace-free "d1xd2xd3" spatial-dimension signature of a NIfTI. Used to
# refuse merging runs from acquisitions with different matrix sizes (fslmerge
# aborts on a size mismatch, which under `set -e` would kill the whole run).
bold_spatial_dims() {
    printf '%sx%sx%s' \
        "$(fslval "$1" dim1 | tr -d '[:space:]')" \
        "$(fslval "$1" dim2 | tr -d '[:space:]')" \
        "$(fslval "$1" dim3 | tr -d '[:space:]')"
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
    require_tool fslnvols
    require_tool fslval

    mkdir -p "$bids_dir"

    local n_subjects=0 n_merged=0 n_skipped=0 n_anat_copied=0 n_existing=0

    mapfile -t sub_dirs < <(find "$raw_dir" -maxdepth 1 -mindepth 1 -type d -name "sub-*" | sort)
    if [[ -n "$LIMIT" ]]; then
        sub_dirs=("${sub_dirs[@]:0:$LIMIT}")
        log "  --limit ${LIMIT}: restricting to ${#sub_dirs[@]} subject(s)"
    fi

    for sub_dir in "${sub_dirs[@]}"; do
        [[ -d "$sub_dir" ]] || continue
        sub_id=$(basename "$sub_dir")

        for ses_dir in "$sub_dir"/ses-*; do
            [[ -d "$ses_dir" ]] || continue
            ses_id=$(basename "$ses_dir")
            func_dir="${ses_dir}/func"
            anat_dir="${ses_dir}/anat"

            local has_func=false has_anat=false

            # ── Step 1.2: group BOLD runs by task label, then merge ───────────
            # A session may hold several distinct task acquisitions with
            # different spatial dimensions (OASIS3: task-restingstate +
            # task-restingstateMB4, task-rest + task-testrest, ... — ~11.5% of
            # subjects). Merging across task labels makes fslmerge abort on a
            # size mismatch and, under `set -e`, kills the whole run. So merge
            # only within a task label, and within that only runs whose spatial
            # dims match the first valid run — the dim guard also drops
            # SBRef/truncated scans mislabeled as bold runs (a few ADNI
            # sessions). Each task label yields its own BIDS output; downstream
            # fMRIPrep processes them all. Mixed/dropped sessions are catalogued
            # in DATA/PREPROCESSING/README.md for later curation.
            if [[ -d "$func_dir" ]]; then
                out_func="${bids_dir}/${sub_id}/${ses_id}/func"

                mapfile -t all_bold < <(find "$func_dir" -name "*_bold.nii.gz" | sort)

                if [[ ${#all_bold[@]} -eq 0 ]]; then
                    log "  WARNING: No BOLD files found in $func_dir."
                    ((n_skipped++)) || true
                else
                    # Distinct task labels in first-seen order.
                    declare -A _seen_task=()
                    task_labels=()
                    for b in "${all_bold[@]}"; do
                        task=$(bold_task_label "$b")
                        if [[ -z "${_seen_task[$task]:-}" ]]; then
                            _seen_task[$task]=1
                            task_labels+=("$task")
                        fi
                    done
                    unset _seen_task

                    for task in "${task_labels[@]}"; do
                        # ── Resume guard: skip a task whose output already exists ──
                        # Re-running the script must not redo completed work. A
                        # valid existing output (>= 50 vols, readable by fslnvols)
                        # is left untouched; an incomplete one left behind by an
                        # interrupted merge (unreadable or < 50 vols) is removed
                        # and rebuilt.
                        out_bold="${out_func}/${sub_id}_${ses_id}_task-${task}_bold.nii.gz"
                        if [[ -f "$out_bold" ]]; then
                            existing_vols=$(fslnvols "$out_bold" 2>/dev/null || echo 0)
                            if [[ "$existing_vols" -ge 50 ]]; then
                                log "  Exists (${existing_vols} vols), skipping: $(basename "$out_bold")"
                                ((n_existing++)) || true
                                has_func=true
                                continue
                            fi
                            log "  Rebuilding incomplete output (${existing_vols} vols): $(basename "$out_bold")"
                            rm -f "$out_bold" "${out_bold%.nii.gz}.json"
                        fi

                        # Volume-filter this task's runs (< 50 TRs = localiser/SBRef).
                        valid_runs=()
                        for b in "${all_bold[@]}"; do
                            [[ "$(bold_task_label "$b")" == "$task" ]] || continue
                            nvols=$(fslnvols "$b" 2>/dev/null || echo 0)
                            if [[ "$nvols" -ge 50 ]]; then
                                valid_runs+=("$b")
                            else
                                log "  Skipping short run ($nvols vols): $(basename "$b")"
                            fi
                        done
                        if [[ ${#valid_runs[@]} -eq 0 ]]; then
                            log "  No valid runs for ${sub_id}/${ses_id} task-${task}; skipping"
                            ((n_skipped++)) || true
                            continue
                        fi

                        # Keep only runs matching the first valid run's spatial
                        # dims so a mismatched acquisition cannot abort fslmerge.
                        ref_dims=$(bold_spatial_dims "${valid_runs[0]}")
                        matched=()
                        for b in "${valid_runs[@]}"; do
                            d=$(bold_spatial_dims "$b")
                            if [[ "$d" == "$ref_dims" ]]; then
                                matched+=("$b")
                            else
                                log "  Skipping dim-mismatched run (${d} vs ${ref_dims}): $(basename "$b")"
                            fi
                        done

                        mkdir -p "$out_func"
                        out_bold="${out_func}/${sub_id}_${ses_id}_task-${task}_bold.nii.gz"
                        if [[ ${#matched[@]} -gt 1 ]]; then
                            log "  Merging ${#matched[@]} runs (task-${task}) → $(basename "$out_bold")"
                            if ! fslmerge -t "$out_bold" "${matched[@]}"; then
                                log "  WARNING: fslmerge failed for ${sub_id}/${ses_id} task-${task}; skipping"
                                rm -f "$out_bold"
                                ((n_skipped++)) || true
                                continue
                            fi
                        else
                            log "  Single valid run for ${sub_id}/${ses_id} task-${task}; copying as-is"
                            cp "${matched[0]}" "$out_bold"
                        fi
                        src_json="${matched[0]%.nii.gz}.json"
                        [[ -f "$src_json" ]] && cp "$src_json" "${out_bold%.nii.gz}.json"
                        ((n_merged++)) || true
                        has_func=true
                    done
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
                        dest="${out_anat}/$(basename "$t1")"
                        if [[ -f "$dest" ]]; then
                            log "  anat exists, skipping: $(basename "$t1")"
                            ((n_existing++)) || true
                            continue
                        fi
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

    log "${dataset_label} done — subjects: $n_subjects, func merged: $n_merged, func skipped: $n_skipped, anat files copied: $n_anat_copied, already-present (resumed): $n_existing"
}

# =============================================================================
# MAIN
# =============================================================================
log "================================================================"
log " Organize BIDS (Fritz, local only)"
log " Dataset : $DATASET"
log " Log     : $LOG_FILE"
log "================================================================"

if [[ "$DATASET" == "oasis3" || "$DATASET" == "both" ]]; then
    organize_bids_dataset "$OASIS3_RAW" "${OASIS3_BIDS}${SMOKETEST_SUFFIX}" "OASIS3"
fi

if [[ "$DATASET" == "adni" || "$DATASET" == "both" ]]; then
    organize_bids_dataset "$ADNI_RAW" "${ADNI_BIDS}${SMOKETEST_SUFFIX}" "ADNI"
fi

log "================================================================"
log " All done! Next step: bash push_bids_to_core.sh --dataset ${DATASET}"
log "================================================================"
