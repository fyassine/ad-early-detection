#!/bin/bash
#SBATCH --account=core-rad
#SBATCH --job-name=fmriprep_non-converter_all
#SBATCH --nodes=1
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=05:00:00
#SBATCH --output /data2/core-rad-fni/Delcode_faschmit/logs/fmriprep/slurm-%A_%a.log
#SBATCH --error /data2/core-rad-fni/Delcode_faschmit/logs/fmriprep/slurm-%A_%a.err
#SBATCH --array=1-272%10

# Set up environment variables
INPUT_BASE=/data2/core-rad-fni/Delcode_faschmit/data/non-converter_BIDS
OUTPUT_BASE=/data2/core-rad-fni/Delcode_faschmit/outputs/non-converter
WORK_DIR=/data2/core-rad-fni/Delcode_faschmit/data/workdir
FS_LICENSE=/data2/core-rad/swunderl/freesurfer/license.txt

# Path to the fmriprep Singularity image
FMRIPREP_SIMG=/data2/core-rad/smoteval/images/fmriprep-23.0.2.simg

# Get the list of subject directories
ALL_SUBJECT_PATHS=($(ls -d ${INPUT_BASE}/M*/sub-*))

# Check if we have any subjects
if [ ${#ALL_SUBJECT_PATHS[@]} -eq 0 ]; then
    echo "No subjects found. Exiting."
    exit 1
fi

# Calculate the index for this job array task
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Get the subject path for this task
CURRENT_PATH=${ALL_SUBJECT_PATHS[$INDEX]}

# Extract timepoint and subject ID
TIMEPOINT=$(echo ${CURRENT_PATH} | rev | cut -d'/' -f2 | rev)
SUBJECT_ID=$(basename ${CURRENT_PATH} | cut -d'-' -f2)

# Set specific directories for this subject
INPUT_DIR=${INPUT_BASE}/${TIMEPOINT}
OUTPUT_DIR=${OUTPUT_BASE}/${TIMEPOINT}

mkdir -p ${OUTPUT_DIR}

echo "*************** Processing subject sub-${SUBJECT_ID} at ${TIMEPOINT} ***************"

singularity run \
    --cleanenv \
    -B ${INPUT_DIR}:/data \
    -B ${OUTPUT_DIR}:/out \
    -B ${FS_LICENSE}:/freesurfer/license.txt \
    -B ${WORK_DIR}:/workdir \
    ${FMRIPREP_SIMG} \
    /data/sub-${SUBJECT_ID} \
    /out \
    participant \
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

echo "Fmriprep completed for subject sub-${SUBJECT_ID} at ${TIMEPOINT}"