#!/usr/bin/env bash
# Drives the interactive (non-sbatch-gated) stages of the pipeline end-to-end on the SAMPLE
# subject: dcm2niix -> BIDS build -> validation. Stops there with a clear message — MRIQC and
# fMRIPrep must be submitted separately via sbatch (see scripts/02_mriqc, scripts/03_fmriprep).
#
# Usage: bash tests/test_pipeline_sample.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBJECT_DIR="${REPO_ROOT}/SAMPLE/03a0a6663-M0_T1_01"
SUBJECT_ID="03a0a6663"
STAGING_DIR="${REPO_ROOT}/staging/${SUBJECT_ID}"
BIDS_ROOT="${REPO_ROOT}/BIDS"

source "${REPO_ROOT}/scripts/lib/logging.sh"
source /dss/dsshome1/0A/di54lup/miniconda3/etc/profile.d/conda.sh
conda activate ad-early-detection
cd "${REPO_ROOT}/scripts/01_dicom_to_bids"

# Per-run log dir (interactive -> no job id suffix); every substep is tee'd to the terminal AND
# to its own .log here plus a combined stage1.log.
RUN_DIR="$(make_run_logdir 01_dicom_to_bids "${SUBJECT_ID}")"
write_run_summary "${RUN_DIR}" \
    "subject     : ${SUBJECT_ID}" \
    "staging_dir : ${STAGING_DIR}" \
    "bids_root   : ${BIDS_ROOT}"
STAGE1_LOG="${RUN_DIR}/stage1.log"

# run_step <substep_name> <command...>: tee combined output to terminal + per-substep log + stage1.log
run_step() {
    local name="$1"; shift
    echo "== ${name} ==" | tee -a "${STAGE1_LOG}"
    # tee to terminal, per-substep file, and the combined log (stderr merged into stdout)
    "$@" 2>&1 | tee "${RUN_DIR}/${name}.log" | tee -a "${STAGE1_LOG}"
    return "${PIPESTATUS[0]}"
}

run_step "1.1_dcm2niix"            python run_dcm2niix.py "${SUBJECT_DIR}/SCANS" "${STAGING_DIR}"
run_step "1.3_dataset_description" python make_dataset_description.py "${BIDS_ROOT}" --name "Glioma Resting-State fMRI (SAMPLE test)"
run_step "1.3_build_bids"          python build_bids.py "${STAGING_DIR}" "${BIDS_ROOT}" "${SUBJECT_ID}" --session 1
run_step "1.4_cleanup_empty_anat"  python cleanup_empty_anat.py "${BIDS_ROOT}"
run_step "1.5_bids_validation"     python run_bids_validator.py "${BIDS_ROOT}"

cat <<EOF

Interactive stages complete. BIDS tree at: ${BIDS_ROOT}
Stage-1 logs:          ${RUN_DIR}

Next steps (sbatch-gated, not run by this script). Each writes clean per-run logs under
logs/<stage>/sub-${SUBJECT_ID}/<timestamp>/ (raw Slurm logs in logs/<stage>/_slurm/):
  1. containers/pull_containers.sh                          (one-time, pulls MRIQC + fMRIPrep .sif)
  2. sbatch scripts/02_mriqc/submit_mriqc.slurm ${BIDS_ROOT} mriqc_out ${SUBJECT_ID}
  3. (after obtaining env/freesurfer_license/license.txt)
     sbatch scripts/03_fmriprep/submit_fmriprep.slurm ${BIDS_ROOT} fmriprep_out ${SUBJECT_ID} 0
  4. sbatch --dependency=afterok:<fmriprep_jobid> scripts/04_postprocessing/submit_postprocessing.slurm \\
       fmriprep_out ${SUBJECT_ID} 1 MNI152NLin2009cAsym_res-2
EOF
