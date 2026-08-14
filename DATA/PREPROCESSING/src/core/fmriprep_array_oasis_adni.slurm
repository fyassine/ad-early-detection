#!/bin/bash
#SBATCH --account=core-rad-fni
#SBATCH --job-name=fmriprep_oasis_adni
#SBATCH --nodes=1
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
# 8h is the jobs-cpu partition ceiling (sinfo TIMELIMIT). 5h was not enough:
# on 2026-07-16 sub-OAS30001 (6 sessions) and sub-ADNI002S1261 (498MB BOLD)
# both hit TIMEOUT at 5h, while sub-OAS30002 (4 sessions) only just finished
# at 4h55m. Subjects exceeding 8h will need chunking by session instead.
#SBATCH --time=08:00:00
#SBATCH --output /data2/core-rad-fni/flakhal/preprocessing/logs/fmriprep/%x-%A_%a.log
#SBATCH --error /data2/core-rad-fni/flakhal/preprocessing/logs/fmriprep/%x-%A_%a.err
#
# Generalized fMRIPrep array job for OASIS3/ADNI (flat sub-*/ses-* BIDS
# layout, no per-timepoint top folder — unlike fmriprep_array_v1.slurm's
# DELCODE non-converter M00/M06 layout). DATASET must be exported at submit
# time, e.g.:
#   sbatch --array=1-272%10 --export=DATASET=oasis3 fmriprep_array_oasis_adni.slurm
#
# Prefer submitting via src/core/submit_array.sh (run ON CORE), which sizes
# --array=1-N from the subject count actually present under INPUT_BASE and sets
# --export for you:
#   bash submit_array.sh --dataset oasis3 --stage fmriprep
# The BIDS push must have completed first — this job fails loudly rather than
# waiting for subjects to arrive. See
# DATA/PREPROCESSING/pipeline-Fritz-CORE.md for the full step sequence.

set -uo pipefail

: "${DATASET:?DATASET must be exported, e.g. sbatch --export=DATASET=oasis3 ...}"

# Set up environment variables. Must match src/core/submit_array.sh's PREP_ROOT
# and src/fritz/push_bids_to_core.sh's CORE_DEST — flakhal owns /data2/core-rad-fni/flakhal/
# (no per-user quota, unlike /home — see
# DATA/PREPROCESSING/src/logs/probe_report.txt for the earlier 2026-07-06
# probe against a colleague's tree).
INPUT_BASE=/data2/core-rad-fni/flakhal/preprocessing/data/${DATASET}
OUTPUT_BASE=/data2/core-rad-fni/flakhal/preprocessing/outputs/${DATASET}/fmriprep
WORK_DIR=/data2/core-rad-fni/flakhal/preprocessing/data/workdir/${DATASET}
# Shared read-only resource (colleague's home dir, group-readable) — leave as-is.
FS_LICENSE=/data2/core-rad-fni/swunderl/freesurfer/license.txt

# Path to the fmriprep Singularity image (shared read-only resource — leave as-is)
FMRIPREP_SIMG=/data2/core-rad-fni/smoteval/images/fmriprep-23.0.2.simg

# TemplateFlow reference-template cache. Must live off /home — this is what
# blew the /home quota on 2026-07-10 (fMRIPrep downloading MNI/OASIS30ANTs
# templates into the default $HOME/.cache/templateflow mid-run). Bound
# explicitly into the container and exported via --env below because
# --cleanenv strips the host environment (and singularity still auto-binds
# $HOME, which resolves to /home/flakhal inside the container otherwise).
TEMPLATEFLOW_HOME=/data2/core-rad-fni/flakhal/templateflow

mkdir -p "${WORK_DIR}" "${OUTPUT_BASE}" "${TEMPLATEFLOW_HOME}"

# Fail fast on a missing subject. The push (src/fritz/push_bids_to_core.sh)
# always completes before this job is submitted — submit_array.sh sizes the
# array from what is already on CORE — so a subject that isn't here is a real
# error, not a transfer still in flight. This used to poll for 30 minutes,
# which turned a failed push into a half-hour hang.
ALL_SUBJECT_PATHS=($(find "${INPUT_BASE}" -maxdepth 1 -type d -name "sub-*" | sort))

if [ ${#ALL_SUBJECT_PATHS[@]} -eq 0 ]; then
    echo "ERROR: no subjects found under ${INPUT_BASE} (dataset=${DATASET}). Did src/fritz/push_bids_to_core.sh run?" >&2
    exit 1
fi

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_PATH=${ALL_SUBJECT_PATHS[$INDEX]:-}

if [[ -z "$CURRENT_PATH" ]]; then
    echo "ERROR: no subject at array index ${SLURM_ARRAY_TASK_ID} (dataset=${DATASET}, only ${#ALL_SUBJECT_PATHS[@]} subjects under ${INPUT_BASE}). The array was sized larger than the data — resubmit with src/core/submit_array.sh, which sizes it from CORE." >&2
    exit 1
fi

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
    -B "${TEMPLATEFLOW_HOME}":/templateflow \
    --env TEMPLATEFLOW_HOME=/templateflow \
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
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "ERROR: fMRIPrep container failed (exit ${rc}) for ${DATASET} subject sub-${SUBJECT_ID}." >&2
    exit "$rc"
fi

echo "Fmriprep completed for ${DATASET} subject sub-${SUBJECT_ID}"
