#!/usr/bin/env bash
# =============================================================================
# submit_array.sh — RUNS ON CORE, not on Fritz.
#
# Sizes and submits one stage of the OASIS3/ADNI SLURM array. Get it onto CORE
# with `bash src/fritz/push_scripts_to_core.sh` from Fritz, then SSH in and run
# it from ${PREP_ROOT}/scripts/core/. See
# DATA/PREPROCESSING/pipeline-Fritz-CORE.md for the full step sequence.
#
# The array size is counted from CORE's own filesystem — whatever subjects are
# actually there — rather than being guessed from a local count on Fritz and
# exported over SSH. Run the push (src/fritz/push_bids_to_core.sh) to
# completion first: the array jobs fail loudly on a missing subject, they no
# longer wait for one to arrive.
#
#   --stage fmriprep       counts ${PREP_ROOT}/data/${DATASET}/sub-*
#   --stage postprocessing counts ${PREP_ROOT}/outputs/${DATASET}/fmriprep/sub-*
#                          (submit once fMRIPrep has produced output)
#
# Usage (on CORE):
#   bash submit_array.sh --dataset oasis3 --stage fmriprep
#   bash submit_array.sh --dataset oasis3 --stage postprocessing
#   bash submit_array.sh --dataset oasis3_smoketest --stage fmriprep --dry-run
#
# --dataset takes the literal value exported to the job (oasis3, adni, or a
# "_smoketest"-suffixed variant such as oasis3_smoketest) — it is not a cohort
# key, so no suffix logic is applied to it here.
#
# Monitor submitted jobs with:
#   squeue -u "$USER" -n fmriprep_oasis_adni
#   sacct  -u "$USER" --name fmriprep_oasis_adni --format=JobID,State,Elapsed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Root of flakhal's preprocessing tree on CORE. flakhal owns
# /data2/core-rad-fni/flakhal/ (no per-user quota, unlike /home — see
# DATA/PREPROCESSING/src/logs/probe_report.txt for the earlier 2026-07-06 probe
# against a colleague's tree).
#
# NOTE: the .slurm files hardcode this same root in their #SBATCH
# --output/--error directives, which cannot read shell variables — SBATCH
# directives are parsed before the script runs. The duplication is unavoidable;
# if you move PREP_ROOT, update both .slurm files' #SBATCH paths to match.
PREP_ROOT="${PREP_ROOT:-/data2/core-rad-fni/flakhal/preprocessing}"

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

# ─── Parse arguments ────────────────────────────────────────────────────
DATASET=""
STAGE=""
CONCURRENCY=10
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2,,}"; shift 2 ;;
        --stage) STAGE="${2,,}"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# Guard clauses — no defaults for --dataset/--stage. Submitting the wrong
# dataset silently is far worse than an error message.
[[ -n "$DATASET" ]] || die "--dataset is required (e.g. --dataset oasis3). It is exported to the job as DATASET."
[[ -n "$STAGE" ]] || die "--stage is required: fmriprep or postprocessing."
[[ "$STAGE" =~ ^(fmriprep|postprocessing)$ ]] || die "--stage must be fmriprep or postprocessing, got '${STAGE}'."
[[ "$CONCURRENCY" =~ ^[0-9]+$ && "$CONCURRENCY" -gt 0 ]] || die "--concurrency must be a positive integer, got '${CONCURRENCY}'."

# ─── Resolve stage-specific input dir and job script ─────────────────────
if [[ "$STAGE" == "fmriprep" ]]; then
    # BIDS input pushed by src/fritz/push_bids_to_core.sh.
    COUNT_DIR="${PREP_ROOT}/data/${DATASET}"
    SLURM_FILE="${SCRIPT_DIR}/fmriprep_array_oasis_adni.slurm"
    HINT="Run 'bash src/fritz/push_bids_to_core.sh --dataset ${DATASET}' on Fritz first."
else
    # Postprocessing consumes fMRIPrep's output, so it sizes from that.
    COUNT_DIR="${PREP_ROOT}/outputs/${DATASET}/fmriprep"
    SLURM_FILE="${SCRIPT_DIR}/postprocessing_array_oasis_adni.slurm"
    HINT="Run the fmriprep stage first and let it produce output."
fi

[[ -f "$SLURM_FILE" ]] || die "Job script not found: ${SLURM_FILE}. Run 'bash src/fritz/push_scripts_to_core.sh' on Fritz to ship it."
[[ -d "$COUNT_DIR" ]] || die "Input dir not found: ${COUNT_DIR}. ${HINT}"

N=$(find "$COUNT_DIR" -maxdepth 1 -mindepth 1 -type d -name "sub-*" | wc -l)
(( N > 0 )) || die "No sub-* directories under ${COUNT_DIR} — nothing to submit. ${HINT}"

# ─── Submit ─────────────────────────────────────────────────────────────
info "================================================================"
info " Submit ${STAGE} array on CORE"
info " Dataset     : ${DATASET}"
info " Counted from: ${COUNT_DIR}"
info " Array size  : ${N} subject(s), concurrency ${CONCURRENCY}"
info " Job script  : ${SLURM_FILE}"
info "================================================================"

SBATCH_ARGS=(
    --array="1-${N}%${CONCURRENCY}"
    --export="DATASET=${DATASET}"
    "$SLURM_FILE"
)

if $DRY_RUN; then
    warn "(--dry-run) Would run: sbatch ${SBATCH_ARGS[*]}"
    exit 0
fi

sbatch_out=$(sbatch "${SBATCH_ARGS[@]}") || die "sbatch submission failed for ${DATASET}/${STAGE}."
success "${sbatch_out}"
success "Monitor with: squeue -u \"\$USER\" -n ${STAGE}_oasis_adni"
