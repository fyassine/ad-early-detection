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
#   1. gate      — only subjects whose fMRIPrep SLURM task is COMPLETED
#                  (queried live via sacct on CORE) are eligible
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
# Credentials: CORE_USER / CORE_HOST / CORE_PASSWORD are read from the repo-root
# .env. Key-based SSH auth is the default; pass --use-password to use sshpass.
#
# The postprocessing image must be staged onto Fritz once (it is only read from
# CORE otherwise). See --stage-sif below, or set POSTPROC_SIMG to its path.
#
# Usage:
#   bash postprocess_local.sh [--dataset oasis3|adni|both] [--jobid N]
#        [--max-parallel N] [--fd-threshold MM] [--limit N]
#        [--stage-sif] [--overwrite] [--dry-run] [--use-password]
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

# fMRIPrep SLURM job name, used to auto-resolve the array job id if --jobid is
# omitted. Matches fmriprep_array_oasis_adni.slurm's #SBATCH --job-name.
FMRIPREP_JOB_NAME="${FMRIPREP_JOB_NAME:-fmriprep_oasis_adni}"

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
        -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$DATASET" =~ ^(oasis3|adni|both)$ ]] || die "--dataset must be oasis3, adni, or both"
[[ "$MAX_PARALLEL" =~ ^[0-9]+$ && "$MAX_PARALLEL" -ge 1 ]] || die "--max-parallel must be a positive integer"

# ─── Runtime + interpreter ──────────────────────────────────────────────────
RUNTIME="$(command -v apptainer || command -v singularity || true)"
[[ -n "$RUNTIME" ]] || die "neither apptainer nor singularity found on Fritz — cannot run the postprocessing container."

PYTHON="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -n "$PYTHON" ]] || die "no python interpreter found (need the project .venv for pandas/nibabel)."

# ─── Credentials ────────────────────────────────────────────────────────────
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

# ─── Resolve the fMRIPrep array job id (if not given) ───────────────────────
resolve_jobid() {
    if [[ -n "$JOBID" ]]; then echo "$JOBID"; return; fi
    # Prefer a currently-queued/running array (ArrayJobId), else the most recent
    # job of this name sacct knows about (today's window by default).
    local jid
    jid="$("${SSH_CMD[@]}" "squeue -h -u \$USER -n '${FMRIPREP_JOB_NAME}' -o '%A' 2>/dev/null | head -1" || true)"
    if [[ -z "$jid" ]]; then
        jid="$("${SSH_CMD[@]}" "sacct -n -X --name '${FMRIPREP_JOB_NAME}' --format JobIDRaw -S \$(date -d '14 days ago' +%F) 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | tail -1" || true)"
    fi
    echo "$jid"
}

# ─── COMPLETED subjects for a dataset (sacct gate, mapped to subject ids) ────
# Prints one subject id per line (WITHOUT sub- prefix). Runs entirely on CORE:
# maps each COMPLETED array-task index to a subject using the SAME sorted
# INPUT_BASE/sub-* ordering the fMRIPrep array used (INDEX = task_id - 1), then
# keeps only those whose fMRIPrep OUTPUT subject dir actually exists.
completed_subjects() {
    local ds_value="$1" jobid="$2"
    "${SSH_CMD[@]}" bash -s -- "$ds_value" "$jobid" "$CORE_PREP_ROOT" <<'REMOTE'
set -uo pipefail
DS="$1"; JOBID="$2"; PREP_ROOT="$3"
INPUT_BASE="${PREP_ROOT}/data/${DS}"
OUTPUT_BASE="${PREP_ROOT}/outputs/${DS}/fmriprep"

# Subject order the array indexed into (full-path sort, then basename).
mapfile -t SUBS < <(ls -d "${INPUT_BASE}"/sub-* 2>/dev/null | sort | xargs -r -n1 basename)
[[ ${#SUBS[@]} -gt 0 ]] || exit 0

# COMPLETED array task indices for this job.
while read -r raw state; do
    [[ "$state" == "COMPLETED" ]] || continue
    # JobID (not JobIDRaw) is the array form "4179125_147"; the index follows the
    # underscore. JobIDRaw would be the underlying allocation id (no underscore).
    idx="${raw##*_}"
    [[ "$idx" =~ ^[0-9]+$ ]] || continue
    sub="${SUBS[$((idx - 1))]:-}"
    [[ -n "$sub" ]] || continue
    # Belt-and-suspenders: only emit if fMRIPrep output for the subject exists.
    # Emit the BARE id (no sub- prefix): the container's --subject_id and
    # qc_motion_gate.py's --subject both expect it prefix-less.
    [[ -d "${OUTPUT_BASE}/${sub}" ]] && echo "${sub#sub-}"
done < <(sacct -j "$JOBID" -X -n -P --format=JobID,State 2>/dev/null | awk -F'|' '{print $1, $2}')
REMOTE
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
        info "${tag} (--dry-run) would: rsync <- ${core_fmriprep} ; QC ; ${RUNTIME##*/} run ${STRATEGY} ; reorient ; flatten -> ${flat_sub}"
        return 0
    fi

    # 2. Pull (COPY) this subject's fMRIPrep dir.
    info "${tag} pulling fMRIPrep output from CORE ..."
    mkdir -p "${local_fmriprep_root}/sub-${sub}"
    rsync -az -e "$RSYNC_RSH" \
        "${CORE_USER}@${CORE_HOST}:${core_fmriprep}/" \
        "${local_fmriprep_root}/sub-${sub}/" \
        || { error "${tag} rsync failed — skipping."; return 1; }

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

    # 5 + 6. Reorient QC-passing sessions and flatten into the product.
    mkdir -p "$flat_sub"
    local ses n_flat=0
    while IFS= read -r ses; do
        [[ -n "$ses" ]] || continue
        local bold
        for bold in "${local_postproc_root}/sub-${sub}/${ses}/"*"desc-${STRATEGY}_bold.nii.gz"; do
            [[ -e "$bold" ]] || { warn "${tag} ${ses}: no *desc-${STRATEGY}_bold.nii.gz produced."; continue; }
            local reoriented="${bold%.nii.gz}_reoriented.nii.gz"
            "$PYTHON" "${SCRIPT_DIR}/final_reorient.py" "$bold" "$reoriented" \
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
info " Log          : $LOG_FILE"
info "================================================================"

$STAGE_SIF && stage_sif
if [[ ! -f "$LOCAL_SIMG" ]] && ! $DRY_RUN; then
    die "postprocessing SIF not found at $LOCAL_SIMG. Stage it once with --stage-sif (or set POSTPROC_SIMG)."
fi

TOTAL_ELIGIBLE=0
for ds_key in oasis3 adni; do
    [[ "$DATASET" == "$ds_key" || "$DATASET" == "both" ]] || continue
    ds_name="$(folder_name "$ds_key")"
    ds_value="$ds_key"   # literal value exported to the fMRIPrep job

    jobid="$(resolve_jobid)"
    if [[ -z "$jobid" ]]; then
        warn "${ds_name}: could not resolve an fMRIPrep job id (pass --jobid) — skipping."
        continue
    fi
    info "${ds_name}: gating on sacct COMPLETED for job ${jobid} ..."

    mapfile -t subs < <(completed_subjects "$ds_value" "$jobid")
    if [[ ${#subs[@]} -eq 0 ]]; then
        warn "${ds_name}: no COMPLETED subjects with output yet — nothing to do."
        continue
    fi
    info "${ds_name}: ${#subs[@]} COMPLETED subject(s) eligible."

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
