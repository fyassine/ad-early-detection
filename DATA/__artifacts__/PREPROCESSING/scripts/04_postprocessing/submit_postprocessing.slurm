#!/bin/bash
# Stage 4 wrapper. NEW script — generated, not copied (original rad_postprocessing2.slurm is
# on the unreachable CORE cluster). Much cheaper than fMRIPrep (single-image filtering, no
# surface reconstruction), but kept as sbatch for consistency and so it can be chained to the
# fMRIPrep array job via --dependency without manual babysitting.
#
# Usage: sbatch --dependency=afterok:<fmriprep_jobid> submit_postprocessing.slurm \
#          <fmriprep_out_dir> <subject_label> <session_label> <space_label>
#   e.g. sbatch submit_postprocessing.slurm fmriprep_out 03a0a6663 1 MNI152NLin2009cAsym_res-2

#SBATCH --job-name=postprocess
#SBATCH --partition=cm4_inter
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Raw Slurm bookkeeping (unfiltered safety net). Clean, per-run logs go to logs/04_postprocessing/sub-*/<TS>/.
#SBATCH --output=logs/04_postprocessing/_slurm/%x_%j.out
#SBATCH --error=logs/04_postprocessing/_slurm/%x_%j.err

set -euo pipefail
source /dss/dsshome1/0A/di54lup/miniconda3/etc/profile.d/conda.sh
conda activate ad-early-detection

FMRIPREP_OUT="$1"
SUB="$2"
SES="$3"
SPACE="$4"
# Slurm copies the batch script to a spool dir, so $0 does NOT point at the repo. Resolve from
# an absolute default (override with PREPROCESSING_ROOT=... sbatch ... if the repo is moved).
REPO_ROOT="${PREPROCESSING_ROOT:-/dss/dsshome1/0A/di54lup/PREPROCESSING}"
SCRIPT_DIR="${REPO_ROOT}/scripts/04_postprocessing"

source "${REPO_ROOT}/scripts/lib/logging.sh"

# fMRIPrep 20.2.x writes derivatives into a `fmriprep/` subdir of the output dir; newer layouts
# write directly. Handle both.
DERIV="${FMRIPREP_OUT}"
[ -d "${FMRIPREP_OUT}/fmriprep/sub-${SUB}" ] && DERIV="${FMRIPREP_OUT}/fmriprep"
FUNC_DIR="${DERIV}/sub-${SUB}/ses-${SES}/func"
PREFIX="sub-${SUB}_ses-${SES}_task-rest"

RUN_DIR="$(make_run_logdir 04_postprocessing "${SUB}")"
write_run_summary "${RUN_DIR}" \
    "subject     : ${SUB}" \
    "session     : ${SES}" \
    "space       : ${SPACE}" \
    "fmriprep_out: ${FMRIPREP_OUT}" \
    "flags       : AROMA-IC + WM/CSF + GSR + non_steady_state outliers, bandpass 0.01-0.1Hz"

# Clean tool output -> RUN_DIR/postprocess.{out,err}; any nilearn progress spam stripped.
{
    python "${SCRIPT_DIR}/postprocess_confounds.py" \
        --preproc-bold "${FUNC_DIR}/${PREFIX}_space-${SPACE}_desc-preproc_bold.nii.gz" \
        --brain-mask   "${FUNC_DIR}/${PREFIX}_space-${SPACE}_desc-brain_mask.nii.gz" \
        --confounds-tsv "${FUNC_DIR}/${PREFIX}_desc-confounds_timeseries.tsv" \
        --aroma-noise-ics "${FUNC_DIR}/${PREFIX}_AROMAnoiseICs.csv" \
        --melodic-mixing "${FUNC_DIR}/${PREFIX}_desc-MELODIC_mixing.tsv" \
        --output "${FUNC_DIR}/${PREFIX}_run-1_space-${SPACE}_desc-ICAAROMA2Phys1GS_bold.nii.gz"

    python "${SCRIPT_DIR}/qc_motion_table.py" \
        --confounds-tsv "${FUNC_DIR}/${PREFIX}_desc-confounds_timeseries.tsv" \
        --subject "${SUB}" --session "${SES}" \
        --append-to "${FMRIPREP_OUT}/qc_summary.tsv"
} > "${RUN_DIR}/postprocess.out" 2> >(progress_filter > "${RUN_DIR}/postprocess.err")
