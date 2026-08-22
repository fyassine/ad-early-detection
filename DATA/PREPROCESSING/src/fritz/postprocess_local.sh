#!/usr/bin/env bash
# =============================================================================
# postprocess_local.sh
#
# Continuous, Fritz-side postprocessing that overlaps with fMRIPrep still
# running on CORE. Instead of waiting for the whole fMRIPrep array to finish and
# then postprocessing on CORE, this pulls each *already-COMPLETED* subject's
# fMRIPrep output to Fritz and denoises it here, so the many idle finished
# subjects get processed while CORE keeps churning through the rest.
#
# Per subject, the flow is:
#   1. gate      — only subjects fMRIPrep finished cleanly are eligible,
#                  detected by the subject-level sub-<id>.html report on CORE
#                  (dataset-scoped, array-agnostic — no SLURM job id needed; see
#                  completed_subjects()). --jobid narrows to one run if wanted.
#   2. pull      — rsync COPY that subject's fMRIPrep dir CORE -> Fritz
#                  (source on CORE is never moved/removed)
#   3. QC        — qc_motion_gate.py: mean-FD > threshold sessions are excluded
#   4. denoise   — apptainer/singularity run postprocessing.sif (per subject)
#   5. reorient  — final_reorient.py: *_bold.nii.gz -> *_bold_reoriented.nii.gz
#   6. flatten   — copy QC-passing reoriented BOLD into the flat product:
#                  DATA/<COHORT>/__fmri_wholebrain_sch200_flat__/fmri/sub-*/
#
# Resumable one-shot: it processes everything currently eligible, then exits.
# Rerun it (or wrap in `watch -n 600 bash postprocess_local.sh ...` / a cron
# entry) to pick up subjects that finished on CORE since the last run. Already
# flattened subjects are skipped, so reruns are cheap and idempotent.
#
# --flatten-only: skip steps 1/2/4 (CORE gate, pull, denoise) entirely — for
# subjects whose denoised output already exists locally (e.g. pulled via the
# CORE-side batch path + pull_derivatives_from_core.sh, which rsyncs raw
# output but never QC/reorient/flattens it). The subject list is instead every
# sub-* already present in DATA/<COHORT>/derivatives/postprocessed/, and only
# QC (step 3) + reorient + flatten (5+6) run, reading confounds from the
# already-local derivatives/fmriprep. No CORE SSH/credentials/apptainer needed
# in this mode.
#
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the repo-root
# .env. Key-based SSH auth is the default; pass --use-password to use sshpass.
# Not needed at all with --flatten-only.
#
# The postprocessing image must be staged onto Fritz once (it is only read from
# CORE otherwise). See --stage-sif below, or set POSTPROC_SIMG to its path.
# Not needed with --flatten-only (no denoising happens in that mode).
#
# Usage:
#   bash postprocess_local.sh [--dataset oasis3|adni|both] [--jobid N]
#        [--max-parallel N] [--fd-threshold MM] [--limit N]
#        [--stage-sif] [--overwrite] [--dry-run] [--use-password] [--flatten-only]
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

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/postprocess_local"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/postprocess_local_$(date +%Y%m%d_%H%M%S).log}"

# CORE-side roots. These are the LIVE deployment paths (flakhal owns
# /data2/core-rad-fni/flakhal/, no per-user quota) — the same tree the running
# fMRIPrep array writes to and that pull_derivatives_from_core.sh reads.
CORE_PREP_ROOT="${CORE_PREP_ROOT:-/data2/core-rad-fni/flakhal/preprocessing}"

# Postprocessing image, staged onto Fritz (see --stage-sif). Override with
# POSTPROC_SIMG=/path/to/postprocessing.sif.
LOCAL_SIMG="${POSTPROC_SIMG:-${REPO_ROOT}/DATA/PREPROCESSING/images/postprocessing.sif}"
CORE_SIMG="${CORE_SIMG:-${CORE_PREP_ROOT}/images/postprocessing.sif}"

# Container denoising parameters — must mirror postprocessing_array_oasis_adni.slurm.
STRATEGY="${STRATEGY:-ICAAROMA2Phys1GS}"
DUMMY="${DUMMY:-10}"
FWHM="${FWHM:-6}"
LPF="${LPF:-0.1}"
HPF="${HPF:-0.01}"

# ─── Parse arguments ────────────────────────────────────────────────────────
DATASET="both"
JOBID=""
MAX_PARALLEL=3
FD_THRESHOLD=0.5
LIMIT=0
STAGE_SIF=false
OVERWRITE=false
DRY_RUN=false
USE_PASSWORD=false
FLATTEN_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="${2,,}"; shift 2 ;;
        --jobid)        JOBID="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --fd-threshold) FD_THRESHOLD="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --stage-sif)    STAGE_SIF=true; shift ;;
        --overwrite)    OVERWRITE=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --use-password) USE_PASSWORD=true; shift ;;
        --flatten-only) FLATTEN_ONLY=true; shift ;;
        -h|--help)      sed -n '2,50p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

if $FLATTEN_ONLY && $STAGE_SIF; then
    die "--flatten-only and --stage-sif are mutually exclusive (flatten-only never touches the SIF)."
fi

[[ "$DATASET" =~ ^(oasis3|adni|both)$ ]] || die "--dataset must be oasis3, adni, or both"
[[ "$MAX_PARALLEL" =~ ^[0-9]+$ && "$MAX_PARALLEL" -ge 1 ]] || die "--max-parallel must be a positive integer"

# ─── Runtime + interpreter ──────────────────────────────────────────────────
# apptainer/singularity is only needed to run the denoise container, which
# --flatten-only never does.
RUNTIME="$(command -v apptainer || command -v singularity || true)"
if ! $FLATTEN_ONLY; then
    [[ -n "$RUNTIME" ]] || die "neither apptainer nor singularity found on Fritz — cannot run the postprocessing container."
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -n "$PYTHON" ]] || die "no python interpreter found (need the project .venv for pandas/nibabel)."

# ─── Credentials ────────────────────────────────────────────────────────────
# --flatten-only never talks to CORE, so it needs neither the .env file nor
# SSH/rsync setup.
if $FLATTEN_ONLY; then
    SSH_CMD=()
    RSYNC_RSH=""
else
    [[ -f "${REPO_ROOT}/.env" ]] || die "${REPO_ROOT}/.env not found (need CORE_USER/CORE_HOST)."
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    [[ -n "${CORE_USER:-}" && -n "${CORE_HOST:-}" ]] || die "CORE_USER and CORE_HOST must be set in ${REPO_ROOT}/.env."

    SSH_CMD=(ssh -o BatchMode=yes "${CORE_USER}@${CORE_HOST}")
    RSYNC_RSH="ssh"
    if $USE_PASSWORD; then
        [[ -n "${CORE_PASSWORD:-}" ]] || die "--use-password passed but CORE_PASSWORD unset in .env."
        command -v sshpass &>/dev/null || die "--use-password needs sshpass installed."
        SSH_CMD=(sshpass -p "${CORE_PASSWORD}" ssh "${CORE_USER}@${CORE_HOST}")
        RSYNC_RSH="sshpass -p ${CORE_PASSWORD} ssh"
    fi
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

folder_name() { case "$1" in oasis3) echo "OASIS3" ;; adni) echo "ADNI" ;; esac; }

# ─── Stage the SIF onto Fritz (once) ────────────────────────────────────────
stage_sif() {
    if [[ -f "$LOCAL_SIMG" ]]; then
        info "SIF already staged at $LOCAL_SIMG — nothing to do."
        return
    fi
    info "Staging postprocessing.sif from CORE -> Fritz ($LOCAL_SIMG) ..."
    mkdir -p "$(dirname "$LOCAL_SIMG")"
    if $DRY_RUN; then
        info "(--dry-run) would rsync ${CORE_USER}@${CORE_HOST}:${CORE_SIMG} -> ${LOCAL_SIMG}"
        return
    fi
    rsync -avh --progress -e "$RSYNC_RSH" "${CORE_USER}@${CORE_HOST}:${CORE_SIMG}" "$LOCAL_SIMG" \
        || die "failed to stage SIF from ${CORE_SIMG}."
    success "SIF staged."
}

# ─── fMRIPrep-complete subjects for a dataset (mapped to subject ids) ────────
# Prints one subject id per line (WITHOUT sub- prefix). Runs entirely on CORE.
#
# Ground truth for "fMRIPrep finished cleanly for this subject" is the
# subject-level report OUTPUT_BASE/sub-<id>.html, which fMRIPrep writes ONLY at
# the end of a successful subject run. This is deliberately NOT a SLURM job-id
# gate:
#   * The job name (fmriprep_oasis_adni) is shared by BOTH cohorts, so no single
#     array id distinguishes ADNI from OASIS3 — an id-based gate can map a
#     completed OASIS3 task index onto an ADNI subject at the same index.
#   * Reruns are PARTIAL arrays (e.g. the 31 timed-out ADNI subjects resubmitted
#     under a new id), so the "completed set" is a union across several arrays,
#     not one job.
#   * A timed-out subject leaves a PARTIAL output dir, so a mere "-d output dir"
#     check would wrongly pass it. The html marker does not exist until the run
#     actually finishes, so partial/timed-out subjects are excluded for free.
# The html gate is dataset-scoped (reads only this dataset's outputs) and
# array-agnostic, so it needs no job id and picks up rerun subjects the moment
# their new array finishes them.
#
# --jobid (optional) narrows the result to subjects whose task COMPLETED under
# the given job(s) — a comma-separated list is allowed. Kept as an override for
# reproducing a specific run; the default (no --jobid) is the correct choice.
completed_subjects() {
    local ds_value="$1" jobid="$2"
    "${SSH_CMD[@]}" bash -s -- "$ds_value" "$jobid" "$CORE_PREP_ROOT" <<'REMOTE'
set -uo pipefail
DS="$1"; JOBID="$2"; PREP_ROOT="$3"
INPUT_BASE="${PREP_ROOT}/data/${DS}"
OUTPUT_BASE="${PREP_ROOT}/outputs/${DS}/fmriprep"

# fMRIPrep-complete subjects: those with a subject-level sub-<id>.html report.
declare -A DONE=()
while IFS= read -r html; do
    b="$(basename "$html")"; b="${b%.html}"     # sub-<id>
    DONE["${b#sub-}"]=1                          # BARE id (no sub- prefix)
done < <(find "$OUTPUT_BASE" -maxdepth 1 -type f -name 'sub-*.html' 2>/dev/null)
[[ ${#DONE[@]} -gt 0 ]] || exit 0

if [[ -z "$JOBID" ]]; then
    # Default: every fMRIPrep-complete subject in this dataset. Emit the BARE id
    # (no sub- prefix): the container's --subject_id and qc_motion_gate.py's
    # --subject both expect it prefix-less.
    for s in "${!DONE[@]}"; do echo "$s"; done
    exit 0
fi

# --jobid override: intersect the html-complete set with the COMPLETED array
# tasks of the given job(s). Map each COMPLETED task index to a subject using
# the SAME sorted INPUT_BASE/sub-* order the fMRIPrep array used (INDEX =
# task_id - 1). JobID (not JobIDRaw) is the array form "4179125_147"; the index
# follows the underscore.
mapfile -t SUBS < <(ls -d "${INPUT_BASE}"/sub-* 2>/dev/null | sort | xargs -r -n1 basename)
[[ ${#SUBS[@]} -gt 0 ]] || exit 0
while read -r raw state; do
    [[ "$state" == "COMPLETED" ]] || continue
    idx="${raw##*_}"
    [[ "$idx" =~ ^[0-9]+$ ]] || continue
    sub="${SUBS[$((idx - 1))]:-}"
    [[ -n "$sub" ]] || continue
    [[ -n "${DONE[${sub#sub-}]:-}" ]] && echo "${sub#sub-}"
done < <(sacct -j "$JOBID" -X -n -P --format=JobID,State 2>/dev/null | awk -F'|' '{print $1, $2}')
REMOTE
}

# ─── Subjects already denoised locally (for --flatten-only) ────────────────
# Prints one BARE subject id per line, sourced from
# DATA/<COHORT>/derivatives/postprocessed/sub-* — no CORE round-trip.
# -type d matters: sibling fMRIPrep output dirs place subject-level
# sub-<ID>.html REPORT FILES next to sub-<ID>/ dirs (see flatten_fmriprep.sh's
# local_fmriprep_subjects() for a confirmed instance of this); restricting to
# directories keeps this immune even though postprocessed/ has none today.
local_postprocessed_subjects() {
    local ds_name="$1"
    find "${REPO_ROOT}/DATA/${ds_name}/derivatives/postprocessed" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' 2>/dev/null \
        | xargs -r -n1 basename \
        | sed 's/^sub-//'
}

# ─── Process one subject end-to-end ─────────────────────────────────────────
# Runs in a subshell (backgrounded by the pool). Logs are prefixed with the sub.
process_subject() {
    local ds_key="$1" ds_name="$2" ds_value="$3" sub="$4"
    local tag="[sub-${sub}]"

    local core_fmriprep="${CORE_PREP_ROOT}/outputs/${ds_value}/fmriprep/sub-${sub}"
    local local_deriv="${REPO_ROOT}/DATA/${ds_name}/derivatives"
    local local_fmriprep_root="${local_deriv}/fmriprep"
    local local_postproc_root="${local_deriv}/postprocessed"
    local flat_root="${REPO_ROOT}/DATA/${ds_name}/__fmri_wholebrain_sch200_flat__/fmri"
    local flat_sub="${flat_root}/sub-${sub}"
    local qc_csv="${REPO_ROOT}/DATA/${ds_name}/__fmri_wholebrain_sch200_flat__/__artifacts__/qc_motion.csv"

    # Idempotency: already-flattened subject is skipped unless --overwrite.
    if ! $OVERWRITE && compgen -G "${flat_sub}/*_bold_reoriented.nii.gz" > /dev/null; then
        info "${tag} already flattened — skipping (use --overwrite to redo)."
        return 0
    fi

    if $DRY_RUN; then
        if $FLATTEN_ONLY; then
            info "${tag} (--dry-run, --flatten-only) would: QC (local) ; reorient (already-denoised, local) ; flatten -> ${flat_sub}"
        else
            info "${tag} (--dry-run) would: rsync <- ${core_fmriprep} ; QC ; ${RUNTIME##*/} run ${STRATEGY} ; reorient ; flatten -> ${flat_sub}"
        fi
        return 0
    fi

    if ! $FLATTEN_ONLY; then
        # 2. Pull (COPY) this subject's fMRIPrep dir.
        info "${tag} pulling fMRIPrep output from CORE ..."
        mkdir -p "${local_fmriprep_root}/sub-${sub}"
        rsync -az -e "$RSYNC_RSH" \
            "${CORE_USER}@${CORE_HOST}:${core_fmriprep}/" \
            "${local_fmriprep_root}/sub-${sub}/" \
            || { error "${tag} rsync failed — skipping."; return 1; }
    fi

    # 3. QC gate → passing sessions (stdout of qc_motion_gate.py).
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
        warn "${tag} all sessions excluded by motion QC — not postprocessing."
        return 0
    fi
    info "${tag} QC-passing sessions: $(echo "$passing" | tr '\n' ' ')"

    if ! $FLATTEN_ONLY; then
        # 4. Denoise (container processes ALL sessions of the subject).
        info "${tag} running postprocessing container (${STRATEGY}) ..."
        mkdir -p "$local_postproc_root"
        if ! "$RUNTIME" run --contain --cleanenv \
                -B "${local_fmriprep_root}":/input \
                -B "${local_postproc_root}":/out \
                "$LOCAL_SIMG" \
                --bids_dir /input --out_dir /out --subject_id "$sub" \
                --strategy "$STRATEGY" --dummy "$DUMMY" --FWHM "$FWHM" --LPF "$LPF" --HPF "$HPF"; then
            error "${tag} postprocessing container failed — skipping."
            return 1
        fi
    fi

    # 5 + 6. Reorient QC-passing sessions and flatten into the product.
    mkdir -p "$flat_sub"
    local ses n_flat=0
    while IFS= read -r ses; do
        [[ -n "$ses" ]] || continue
        local bold
        for bold in "${local_postproc_root}/sub-${sub}/${ses}/"*"desc-${STRATEGY}_bold.nii.gz"; do
            [[ -e "$bold" ]] || { warn "${tag} ${ses}: no *desc-${STRATEGY}_bold.nii.gz produced."; continue; }
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
# Track only OUR background PIDs explicitly. `exec > >(tee …)` below registers
# the tee subshell as a job too, and a bare `wait`/`jobs` with no filter would
# include it — that subshell only exits once stdout closes (i.e. once this
# script exits), so waiting on it here would deadlock. Always wait/count by
# explicit PID, never bare `wait` or unfiltered `jobs`.
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
info " Continuous Fritz-side postprocessing"
info " Dataset      : $DATASET"
info " Max parallel : $MAX_PARALLEL   FD threshold: ${FD_THRESHOLD}mm   strategy: $STRATEGY"
info " Dry-run      : $DRY_RUN        Overwrite: $OVERWRITE"
info " Flatten-only : $FLATTEN_ONLY"
info " Log          : $LOG_FILE"
info "================================================================"

if ! $FLATTEN_ONLY; then
    $STAGE_SIF && stage_sif
    if [[ ! -f "$LOCAL_SIMG" ]] && ! $DRY_RUN; then
        die "postprocessing SIF not found at $LOCAL_SIMG. Stage it once with --stage-sif (or set POSTPROC_SIMG)."
    fi
fi

TOTAL_ELIGIBLE=0
for ds_key in oasis3 adni; do
    [[ "$DATASET" == "$ds_key" || "$DATASET" == "both" ]] || continue
    ds_name="$(folder_name "$ds_key")"
    ds_value="$ds_key"   # literal value exported to the fMRIPrep job

    if $FLATTEN_ONLY; then
        info "${ds_name}: gating on already-local derivatives/postprocessed/sub-* (no CORE round-trip) ..."
        mapfile -t subs < <(local_postprocessed_subjects "$ds_name" | sort -u)
    else
        if [[ -n "$JOBID" ]]; then
            info "${ds_name}: gating on fMRIPrep completion (sub-*.html), restricted to job(s) ${JOBID} ..."
        else
            info "${ds_name}: gating on fMRIPrep completion (sub-*.html reports under outputs/${ds_value}/fmriprep) ..."
        fi
        mapfile -t subs < <(completed_subjects "$ds_value" "$JOBID" | sort -u)
    fi
    if [[ ${#subs[@]} -eq 0 ]]; then
        warn "${ds_name}: no eligible subjects found — nothing to do."
        continue
    fi
    info "${ds_name}: ${#subs[@]} eligible subject(s)."

    count=0
    for sub in "${subs[@]}"; do
        (( LIMIT > 0 && count >= LIMIT )) && { info "${ds_name}: --limit ${LIMIT} reached."; break; }
        count=$((count + 1)); TOTAL_ELIGIBLE=$((TOTAL_ELIGIBLE + 1))
        pool_wait_slot
        process_subject "$ds_key" "$ds_name" "$ds_value" "$sub" &
        PIDS+=("$!")
    done
done

wait "${PIDS[@]}" 2>/dev/null || true
success "================================================================"
success " All eligible subjects processed this pass (${TOTAL_ELIGIBLE} attempted)."
success " Rerun to pick up subjects CORE finishes later."
success "================================================================"
