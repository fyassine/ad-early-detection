#!/usr/bin/env bash
# =============================================================================
# flatten_fmriprep.sh
#
# Builds the fMRIPrep-stage counterpart of the postprocessed flat product
# (DATA/<COHORT>/__fmri_wholebrain_sch200_flat__/fmri/). That one holds
# QC-passing, reoriented, DENOISED BOLD (see postprocess_local.sh). This one
# holds QC-passing, reoriented, RAW (pre-denoising) fMRIPrep preproc BOLD, one
# file per QC-passing session:
#
#   DATA/<COHORT>/__fmriprep_wholebrain_flat__/fmri/sub-<ID>/
#       sub-<ID>_ses-<S>_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_reoriented.nii.gz
#
# Entirely local — reads only DATA/<COHORT>/derivatives/fmriprep/, which must
# already be populated (via pull_derivatives_from_core.sh or
# postprocess_local.sh's pull step). No CORE SSH/credentials, no denoise
# container.
#
# Per subject:
#   1. QC   — qc_motion_gate.py: mean-FD > threshold sessions are excluded
#             (same gate/threshold convention as postprocess_local.sh; reads
#             the same *_desc-confounds_timeseries.tsv, so verdicts agree with
#             the postprocessed flat product's ledger for the same subject).
#   2. reorient — final_reorient.py on each QC-passing session's
#             *_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
#   3. flatten  — copy the reoriented file into the flat product above.
#
# Resumable one-shot: already-flattened subjects are skipped (idempotent), so
# reruns are cheap. Pass --overwrite to redo a subject.
#
# Usage:
#   bash flatten_fmriprep.sh [--dataset oasis3|adni|both] [--fd-threshold MM]
#        [--dummy N] [--max-parallel N] [--overwrite] [--dry-run]
# =============================================================================

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────────
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

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/flatten_fmriprep"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/flatten_fmriprep_$(date +%Y%m%d_%H%M%S).log}"

# fMRIPrep BOLD file this product flattens — same space/resolution the
# postprocessed product and downstream Schaefer-200 extraction use.
BOLD_GLOB_SUFFIX="_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

# ─── Parse arguments ────────────────────────────────────────────────────────
DATASET="both"
FD_THRESHOLD=0.5
DUMMY=10
MAX_PARALLEL=4
OVERWRITE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="${2,,}"; shift 2 ;;
        --fd-threshold) FD_THRESHOLD="$2"; shift 2 ;;
        --dummy)        DUMMY="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --overwrite)    OVERWRITE=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)      sed -n '2,29p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$DATASET" =~ ^(oasis3|adni|both)$ ]] || die "--dataset must be oasis3, adni, or both"
[[ "$MAX_PARALLEL" =~ ^[0-9]+$ && "$MAX_PARALLEL" -ge 1 ]] || die "--max-parallel must be a positive integer"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -n "$PYTHON" ]] || die "no python interpreter found (need the project .venv for pandas/nibabel)."

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

folder_name() { case "$1" in oasis3) echo "OASIS3" ;; adni) echo "ADNI" ;; esac; }

# ─── Subjects with local fMRIPrep output ────────────────────────────────────
# -type d matters: fMRIPrep writes subject-level sub-<ID>.html REPORT FILES
# directly under derivatives/fmriprep/, alongside the sub-<ID>/ directories. A
# bare `sub-*` glob matches both, silently feeding "OAS31334.html" through as
# if it were a subject id (it then fails immediately in QC with no subject
# dir found — no wrong data lands in the flat product, but it doubles the
# work for nothing).
local_fmriprep_subjects() {
    local ds_name="$1"
    find "${REPO_ROOT}/DATA/${ds_name}/derivatives/fmriprep" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' 2>/dev/null \
        | xargs -r -n1 basename \
        | sed 's/^sub-//'
}

# ─── Process one subject end-to-end ─────────────────────────────────────────
process_subject() {
    local ds_name="$1" ds_value="$2" sub="$3"
    local tag="[sub-${sub}]"

    local local_fmriprep_root="${REPO_ROOT}/DATA/${ds_name}/derivatives/fmriprep"
    local flat_root="${REPO_ROOT}/DATA/${ds_name}/__fmriprep_wholebrain_flat__/fmri"
    local flat_sub="${flat_root}/sub-${sub}"
    local qc_csv="${REPO_ROOT}/DATA/${ds_name}/__fmriprep_wholebrain_flat__/__artifacts__/qc_motion.csv"

    # Idempotency: already-flattened subject is skipped unless --overwrite.
    if ! $OVERWRITE && compgen -G "${flat_sub}/*_bold_reoriented.nii.gz" > /dev/null; then
        info "${tag} already flattened — skipping (use --overwrite to redo)."
        return 0
    fi

    if $DRY_RUN; then
        info "${tag} (--dry-run) would: QC ; reorient ; flatten -> ${flat_sub}"
        return 0
    fi

    # 1. QC gate → passing sessions (stdout of qc_motion_gate.py).
    info "${tag} QC (mean-FD > ${FD_THRESHOLD}mm excluded) ..."
    local passing
    if ! passing="$("$PYTHON" "${SCRIPT_DIR}/qc_motion_gate.py" \
            --fmriprep-root "$local_fmriprep_root" --subject "$sub" \
            --dataset "$ds_value" --qc-csv "$qc_csv" \
            --fd-threshold "$FD_THRESHOLD" --scrub-threshold 0.2 --dummy "$DUMMY")"; then
        error "${tag} QC failed (missing/malformed confounds) — skipping."
        return 1
    fi
    if [[ -z "$passing" ]]; then
        warn "${tag} all sessions excluded by motion QC — nothing to flatten."
        return 0
    fi
    info "${tag} QC-passing sessions: $(echo "$passing" | tr '\n' ' ')"

    # 2 + 3. Reorient QC-passing sessions' preproc BOLD and flatten.
    mkdir -p "$flat_sub"
    local ses n_flat=0
    while IFS= read -r ses; do
        [[ -n "$ses" ]] || continue
        local bold
        for bold in "${local_fmriprep_root}/sub-${sub}/${ses}/func/"*"${BOLD_GLOB_SUFFIX}"; do
            [[ -e "$bold" ]] || { warn "${tag} ${ses}: no *${BOLD_GLOB_SUFFIX} produced."; continue; }
            local reoriented="${bold%.nii.gz}_reoriented.nii.gz"
            local -a reorient_flags=()
            $OVERWRITE && reorient_flags+=(--overwrite)
            "$PYTHON" "${SCRIPT_DIR}/final_reorient.py" "$bold" "$reoriented" "${reorient_flags[@]}" \
                || { error "${tag} reorient failed for $(basename "$bold")."; continue; }
            cp -f "$reoriented" "${flat_sub}/" && n_flat=$((n_flat + 1))
        done
    done <<< "$passing"

    if (( n_flat == 0 )); then
        warn "${tag} nothing landed in the flat product (no passing BOLD reoriented)."
        return 1
    fi
    success "${tag} done — ${n_flat} reoriented BOLD -> ${flat_sub}/"
}

# ─── Simple concurrency pool ────────────────────────────────────────────────
# Same "track only our PIDs" reasoning as postprocess_local.sh: exec > >(tee …)
# registers a subshell as a job too, so bare wait/jobs would deadlock.
declare -a PIDS=()
pool_wait_slot() {
    while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
        wait -n "${PIDS[@]}" 2>/dev/null || true
        local alive=()
        local pid
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        PIDS=("${alive[@]}")
    done
}

# ─── Main ───────────────────────────────────────────────────────────────────
info "================================================================"
info " fMRIPrep-stage flatten (local-only, no CORE round-trip)"
info " Dataset      : $DATASET"
info " Max parallel : $MAX_PARALLEL   FD threshold: ${FD_THRESHOLD}mm   dummy: $DUMMY"
info " Dry-run      : $DRY_RUN        Overwrite: $OVERWRITE"
info " Log          : $LOG_FILE"
info "================================================================"

TOTAL_ELIGIBLE=0
for ds_key in oasis3 adni; do
    [[ "$DATASET" == "$ds_key" || "$DATASET" == "both" ]] || continue
    ds_name="$(folder_name "$ds_key")"

    info "${ds_name}: gating on local derivatives/fmriprep/sub-* ..."
    mapfile -t subs < <(local_fmriprep_subjects "$ds_name" | sort -u)
    if [[ ${#subs[@]} -eq 0 ]]; then
        warn "${ds_name}: no local fMRIPrep output yet — nothing to do."
        continue
    fi
    info "${ds_name}: ${#subs[@]} eligible subject(s)."

    for sub in "${subs[@]}"; do
        TOTAL_ELIGIBLE=$((TOTAL_ELIGIBLE + 1))
        pool_wait_slot
        process_subject "$ds_name" "$ds_key" "$sub" &
        PIDS+=("$!")
    done
done

wait "${PIDS[@]}" 2>/dev/null || true
success "================================================================"
success " All eligible subjects processed this pass (${TOTAL_ELIGIBLE} attempted)."
success " Rerun to pick up subjects pulled later."
success "================================================================"
