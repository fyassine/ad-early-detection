#!/bin/bash
# Stage 3: fMRIPrep. NEW script — generated, not copied (the original fmriprep_array_v1.slurm
# lives on the unreachable CORE cluster at /data2/core-rad/smoteval/HGG_Teil3/slurms/). See
# ../../original/README.md and docs/PIPELINE_OVERVIEW.md for what's known about the original.
#
# --use-aroma is required: the institutional confound strategy (postprocess_confounds.py)
# explicitly regresses out AROMA noise components, so AROMA must run inside fMRIPrep.
# --fs-no-reconall: surface reconstruction is skipped. The institutional outputs are all
# volumetric MNI-space (desc-ICAAROMA2Phys1GS in MNI152NLin2009cAsym:res-2, no fsaverage/
# surface outputs anywhere in the original pipeline), so surfaces aren't needed and skipping
# recon-all keeps a single subject well under the cm4_inter 8h cap (~few hours). If surface-
# based parcellation is later required, drop this flag and move back to teramem_inter (10-day).
#
# Usage: sbatch submit_fmriprep.slurm <bids_root> <out_dir> <participant_label> <dummy_scans>
#   e.g. sbatch submit_fmriprep.slurm BIDS fmriprep_out 03a0a6663 0
#
# Prerequisite: env/freesurfer_license/license.txt must exist (see env/freesurfer_license/README.md).

#SBATCH --job-name=fmriprep
#SBATCH --partition=cm4_inter
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Raw Slurm bookkeeping (unfiltered safety net). Clean, per-run logs go to logs/03_fmriprep/sub-*/<TS>/.
#SBATCH --output=logs/03_fmriprep/_slurm/%x_%j.out
#SBATCH --error=logs/03_fmriprep/_slurm/%x_%j.err

set -euo pipefail
module load apptainer/1.3.4

# Slurm copies the batch script to a spool dir, so $0 does NOT point at the repo. Resolve the
# repo root from an absolute default (override with PREPROCESSING_ROOT=... sbatch ... if moved).
REPO_ROOT="${PREPROCESSING_ROOT:-/dss/dsshome1/0A/di54lup/PREPROCESSING}"
BIDS_ROOT="$(realpath "$1")"
OUT_DIR="$(realpath "$2")"
PARTICIPANT_LABEL="$3"
DUMMY_SCANS="${4:-0}"
SIF="${REPO_ROOT}/containers/fmriprep-20.2.7.sif"  # LTS line — see containers/pull_containers.sh for why
FS_LICENSE="${REPO_ROOT}/env/freesurfer_license/license.txt"
WORK_DIR="${OUT_DIR}/work_${PARTICIPANT_LABEL}"
TEMPLATEFLOW_DIR="${REPO_ROOT}/.cache/templateflow"

source "${REPO_ROOT}/scripts/lib/logging.sh"

mkdir -p "${OUT_DIR}" "${WORK_DIR}" "${TEMPLATEFLOW_DIR}"

if [ ! -f "${SIF}" ]; then
    echo "fMRIPrep container not found at ${SIF}. Run containers/pull_containers.sh first." >&2
    exit 1
fi
if [ ! -f "${FS_LICENSE}" ]; then
    echo "FreeSurfer license missing at ${FS_LICENSE}. See env/freesurfer_license/README.md." >&2
    exit 1
fi

RUN_DIR="$(make_run_logdir 03_fmriprep "${PARTICIPANT_LABEL}")"
write_run_summary "${RUN_DIR}" \
    "subject     : ${PARTICIPANT_LABEL}" \
    "bids_root   : ${BIDS_ROOT}" \
    "out_dir     : ${OUT_DIR}" \
    "container   : ${SIF##*/}" \
    "flags       : --use-aroma --fs-no-reconall --dummy-scans ${DUMMY_SCANS} --output-spaces MNI152NLin2009cAsym:res-2"

# Clean tool output -> RUN_DIR/fmriprep.{out,err}; tqdm progress spam stripped from stderr.
# Persistent TemplateFlow cache so MNI templates download once across runs (the biggest source
# of .err spam on first runs).
apptainer run --cleanenv \
    --env TEMPLATEFLOW_HOME=/templateflow \
    --env TQDM_DISABLE=1 \
    --bind "${BIDS_ROOT}:/data:ro" \
    --bind "${OUT_DIR}:/out" \
    --bind "${WORK_DIR}:/work" \
    --bind "${TEMPLATEFLOW_DIR}:/templateflow" \
    --bind "${FS_LICENSE}:/opt/freesurfer_license.txt" \
    "${SIF}" /data /out participant \
    --participant-label "${PARTICIPANT_LABEL}" \
    --use-aroma \
    --fs-no-reconall \
    --dummy-scans "${DUMMY_SCANS}" \
    --fs-license-file /opt/freesurfer_license.txt \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --work-dir /work \
    --nthreads "${SLURM_CPUS_PER_TASK}" \
    --omp-nthreads "${SLURM_CPUS_PER_TASK}" \
    --mem-mb 60000 \
    > "${RUN_DIR}/fmriprep.out" 2> >(progress_filter > "${RUN_DIR}/fmriprep.err")
