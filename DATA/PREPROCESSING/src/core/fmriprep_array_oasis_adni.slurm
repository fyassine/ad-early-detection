#!/bin/bash
#SBATCH --account=core-rad-fni
#SBATCH --job-name=fmriprep_oasis_adni
#SBATCH --nodes=1
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=05:00:00
#SBATCH --output /home/flakhal/preprocessing/logs/fmriprep/%x-%A_%a.log
#SBATCH --error /home/flakhal/preprocessing/logs/fmriprep/%x-%A_%a.err
#
# Generalized fMRIPrep array job for OASIS3/ADNI (flat sub-*/ses-* BIDS
# layout, no per-timepoint top folder — unlike fmriprep_array_v1.slurm's
# DELCODE non-converter M00/M06 layout). DATASET must be exported at submit
# time, e.g.:
#   sbatch --array=1-272%10 --export=DATASET=oasis3 fmriprep_array_oasis_adni.slurm
# See src/fritz/submit_fmriprep_array_core.sh, which sizes --array=1-N from
# the local (already-organized) BIDS subject count and sets --export for you.

set -uo pipefail

: "${DATASET:?DATASET must be exported, e.g. sbatch --export=DATASET=oasis3 ...}"

# Set up environment variables. Must match src/fritz/submit_fmriprep_array_core.sh's
# CORE_DEST_ROOT/CORE_OUTPUTS_ROOT — flakhal has no write access to
# /data2/core-rad-fni/Delcode_faschmit/ (probed 2026-07-06, see
# DATA/PREPROCESSING/src/logs/probe_report.txt), everything lives under the
# CORE home dir instead.
INPUT_BASE=/home/flakhal/preprocessing/data/${DATASET}
OUTPUT_BASE=/home/flakhal/preprocessing/outputs/${DATASET}/fmriprep
WORK_DIR=/home/flakhal/preprocessing/data/workdir/${DATASET}
# Shared read-only resource (colleague's home dir, group-readable) — leave as-is.
FS_LICENSE=/data2/core-rad-fni/swunderl/freesurfer/license.txt

# Path to the fmriprep Singularity image (shared read-only resource — leave as-is)
FMRIPREP_SIMG=/data2/core-rad-fni/smoteval/images/fmriprep-23.0.2.simg

mkdir -p "${WORK_DIR}" "${OUTPUT_BASE}"

INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# The push (rsync to CORE) may still be in progress when this task starts —
# that's the whole point of submitting early. Poll for this task's subject
# to show up rather than failing immediately if the transfer hasn't reached
# it yet.
WAIT_TIMEOUT=1800
WAIT_INTERVAL=30
waited=0

ALL_SUBJECT_PATHS=($(ls -d "${INPUT_BASE}"/sub-* 2>/dev/null | sort))
CURRENT_PATH=${ALL_SUBJECT_PATHS[$INDEX]:-}

while [[ -z "$CURRENT_PATH" || ! -d "$CURRENT_PATH" ]]; do
    if (( waited >= WAIT_TIMEOUT )); then
        echo "ERROR: subject for array index ${SLURM_ARRAY_TASK_ID} (dataset=${DATASET}) never arrived under ${INPUT_BASE} after ${WAIT_TIMEOUT}s. Is the push rsync still running / did it fail?" >&2
        exit 1
    fi
    echo "Subject for array index ${SLURM_ARRAY_TASK_ID} not present yet under ${INPUT_BASE} — waited ${waited}s, retrying in ${WAIT_INTERVAL}s..."
    sleep "$WAIT_INTERVAL"
    waited=$((waited + WAIT_INTERVAL))
    ALL_SUBJECT_PATHS=($(ls -d "${INPUT_BASE}"/sub-* 2>/dev/null | sort))
    CURRENT_PATH=${ALL_SUBJECT_PATHS[$INDEX]:-}
done

SUBJECT_ID=$(basename "${CURRENT_PATH}" | cut -d'-' -f2)
OUTPUT_DIR=${OUTPUT_BASE}
mkdir -p "${OUTPUT_DIR}"

echo "*************** Processing ${DATASET} subject sub-${SUBJECT_ID} ***************"

singularity run \
    --cleanenv \
    -B "${INPUT_BASE}":/data \
    -B "${OUTPUT_DIR}":/out \
    -B "${FS_LICENSE}":/freesurfer/license.txt \
    -B "${WORK_DIR}":/workdir \
    "${FMRIPREP_SIMG}" \
    /data \
    /out \
    participant \
    --participant-label "${SUBJECT_ID}" \
    --fs-license-file /freesurfer/license.txt \
    --skip-bids-validation \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --bold2t1w-init register \
    --bold2t1w-dof 12 \
    --force-bbr \
    --skull-strip-t1w force \
    --medial-surface-nan \
    --dummy-scans 10 \
    --use-aroma \
    --verbose \
    --n_cpus 16 \
    --mem 32000 \
    --debug all \
    --output-layout bids \
    --fs-no-reconall \
    -w /workdir/

echo "Fmriprep completed for ${DATASET} subject sub-${SUBJECT_ID}"
