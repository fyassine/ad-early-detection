#!/bin/bash
#SBATCH --account=core-rad-fni
#SBATCH --job-name=postprocessing_oasis_adni
#SBATCH --nodes=1
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output /data2/core-rad-fni/flakhal/preprocessing/logs/postprocessing/%x-%A_%a.log
#SBATCH --error /data2/core-rad-fni/flakhal/preprocessing/logs/postprocessing/%x-%A_%a.err
#
# Generalized postprocessing array job for OASIS3/ADNI. Unlike
# postprocessing_array_v1.slurm (DELCODE non-converter, per-session
# INPUT_BASE/<session>/sub-* layout), fMRIPrep output for OASIS3/ADNI is flat
# (INPUT_BASE/sub-*, one level — see fmriprep_array_oasis_adni.slurm), so no
# SESSION parsing is needed. DATASET must be exported at submit time, e.g.:
#   sbatch --array=1-N%10 --export=DATASET=oasis3 postprocessing_array_oasis_adni.slurm
#
# Prefer submitting via src/core/submit_array.sh (run ON CORE), which sizes
# --array=1-N from the fMRIPrep output subject count and sets --export for you:
#   bash submit_array.sh --dataset oasis3 --stage postprocessing
# Submit this once fMRIPrep has produced enough output — it depends on upstream
# fMRIPrep results existing. See DATA/PREPROCESSING/pipeline-Fritz-CORE.md for
# the full step sequence.

set -uo pipefail

: "${DATASET:?DATASET must be exported, e.g. sbatch --export=DATASET=oasis3 ...}"

# Must match src/core/submit_array.sh's PREP_ROOT and
# src/fritz/pull_derivatives_from_core.sh's CORE_OUTPUTS_ROOT —
# flakhal owns /data2/core-rad-fni/flakhal/ (no per-user quota, unlike
# /home — see DATA/PREPROCESSING/src/logs/probe_report.txt for the earlier
# 2026-07-06 probe against a colleague's tree).
export INPUT_BASE=/data2/core-rad-fni/flakhal/preprocessing/outputs/${DATASET}/fmriprep
export OUTPUT_BASE=/data2/core-rad-fni/flakhal/preprocessing/outputs/${DATASET}/postprocessed

# Path to the postprocessing Singularity image. Staged into flakhal's OWN tree
# (copied once from flbrandl's readable /data2/core-rad-fni/ copy) so
# postprocessing owns its dependency and does not break if a colleague's tree
# becomes unreadable — which is exactly what happened when this pointed at
# /data2/core-rad/flbrandl/ (bare core-rad/, permission-denied for flakhal;
# same class of failure the fMRIPrep SIF path hit — see
# fmriprep_array_oasis_adni.slurm's FMRIPREP_SIMG).
POSTPROC_SIMG=/data2/core-rad-fni/flakhal/preprocessing/images/postprocessing.sif

mkdir -p "${OUTPUT_BASE}"

# Get the list of subject directories (flat — no per-session subdir)
ALL_SUBJECT_PATHS=($(find "${INPUT_BASE}" -maxdepth 1 -type d -name "sub-*" | sort))

if [ ${#ALL_SUBJECT_PATHS[@]} -eq 0 ]; then
    echo "No subjects found under ${INPUT_BASE}. Exiting." >&2
    exit 1
fi

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_PATH=${ALL_SUBJECT_PATHS[$INDEX]:-}

if [[ -z "$CURRENT_PATH" ]]; then
    echo "ERROR: no subject at array index ${SLURM_ARRAY_TASK_ID} (only ${#ALL_SUBJECT_PATHS[@]} subjects found)." >&2
    exit 1
fi

SUBJECT_ID=$(basename "${CURRENT_PATH}" | cut -d'-' -f2)
OUTPUT_DIR=${OUTPUT_BASE}

echo "*************** Postprocessing ${DATASET} subject ${SUBJECT_ID} ***************"

singularity run --contain --cleanenv \
    -B "${INPUT_BASE}":/input \
    -B "${OUTPUT_DIR}":/out \
    "${POSTPROC_SIMG}" \
    --bids_dir /input \
    --out_dir /out \
    --subject_id "${SUBJECT_ID}" \
    --strategy ICAAROMA2Phys1GS \
    --dummy 10 \
    --FWHM 6 \
    --LPF 0.1 \
    --HPF 0.01
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "ERROR: postprocessing container failed (exit ${rc}) for ${DATASET} subject ${SUBJECT_ID}." >&2
    exit "$rc"
fi

echo "Postprocessing completed for ${DATASET} subject ${SUBJECT_ID}"
