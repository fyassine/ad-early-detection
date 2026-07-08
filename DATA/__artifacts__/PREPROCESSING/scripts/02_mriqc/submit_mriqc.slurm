#!/bin/bash
# Stage 2: pre-fMRIPrep MRIQC QC pass (per the doc's "before fMRIPrep" checkpoint).
#
# This is tractable to run in the foreground (~10-20 min for one subject), but per HPC
# login-node etiquette it still goes through sbatch rather than running on cm4login1.
#
# Usage: sbatch submit_mriqc.slurm <bids_root> <mriqc_out_dir> <participant_label>
#   e.g. sbatch submit_mriqc.slurm BIDS mriqc_out 03a0a6663

#SBATCH --job-name=mriqc
#SBATCH --partition=cm4_inter
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Raw Slurm bookkeeping (unfiltered safety net). Clean, per-run logs go to logs/02_mriqc/sub-*/<TS>/.
#SBATCH --output=logs/02_mriqc/_slurm/%x_%j.out
#SBATCH --error=logs/02_mriqc/_slurm/%x_%j.err

set -euo pipefail
module load apptainer/1.3.4

BIDS_ROOT="$(realpath "$1")"
OUT_DIR="$(realpath "$2")"
PARTICIPANT_LABEL="$3"
# Slurm copies the batch script to a spool dir, so $0 does NOT point at the repo. Resolve the
# repo root from an absolute default (override with PREPROCESSING_ROOT=... sbatch ... if moved).
REPO_ROOT="${PREPROCESSING_ROOT:-/dss/dsshome1/0A/di54lup/PREPROCESSING}"
SIF="${REPO_ROOT}/containers/mriqc-24.0.2.sif"
WORK_DIR="${OUT_DIR}/work_${PARTICIPANT_LABEL}"
TEMPLATEFLOW_DIR="${REPO_ROOT}/.cache/templateflow"

source "${REPO_ROOT}/scripts/lib/logging.sh"

mkdir -p "${OUT_DIR}" "${WORK_DIR}" "${TEMPLATEFLOW_DIR}"

if [ ! -f "${SIF}" ]; then
    echo "MRIQC container not found at ${SIF}. Run containers/pull_containers.sh first." >&2
    exit 1
fi

RUN_DIR="$(make_run_logdir 02_mriqc "${PARTICIPANT_LABEL}")"
write_run_summary "${RUN_DIR}" \
    "subject     : ${PARTICIPANT_LABEL}" \
    "bids_root   : ${BIDS_ROOT}" \
    "out_dir     : ${OUT_DIR}" \
    "container   : ${SIF##*/}" \
    "flags       : --no-sub --nprocs ${SLURM_CPUS_PER_TASK}"

# Clean tool output -> RUN_DIR/mriqc.{out,err}; tqdm progress spam stripped from stderr.
# Persistent TemplateFlow cache so templates download once across runs.
apptainer run --cleanenv \
    --env TEMPLATEFLOW_HOME=/templateflow \
    --env TQDM_DISABLE=1 \
    --bind "${BIDS_ROOT}:/data:ro" \
    --bind "${OUT_DIR}:/out" \
    --bind "${WORK_DIR}:/work" \
    --bind "${TEMPLATEFLOW_DIR}:/templateflow" \
    "${SIF}" /data /out participant \
    --participant-label "${PARTICIPANT_LABEL}" \
    --work-dir /work \
    --nprocs "${SLURM_CPUS_PER_TASK}" \
    --no-sub \
    > "${RUN_DIR}/mriqc.out" 2> >(progress_filter > "${RUN_DIR}/mriqc.err")
