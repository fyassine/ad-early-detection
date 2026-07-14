#!/bin/bash
#SBATCH --account=core-rad-fni
#SBATCH --job-name=postprocessing_all_sessions
#SBATCH --nodes=1
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output /data2/core-rad-fni/Delcode_faschmit/logs/postprocessing/slurm-%A_%a.log
#SBATCH --error /data2/core-rad-fni/Delcode_faschmit/logs/postprocessing/slurm-%A_%a.err
#SBATCH --array=1-268%10

# Set up environment variables (Pfade für alle Sessions)
export INPUT_BASE=/data2/core-rad-fni/Delcode_faschmit/outputs/non-converter/fmriprep
export OUTPUT_BASE=/data2/core-rad-fni/Delcode_faschmit/outputs/non-converter/postprocessed

# Path to the postprocessing Singularity image
POSTPROC_SIMG=/data2/core-rad/flbrandl/postprocessing.sif

# Get the list of subject directories (NUR Ordner, keine HTMLs)
ALL_SUBJECT_PATHS=($(find ${INPUT_BASE} -maxdepth 2 -type d -name "sub-*" | sort))

# Check if we have any subjects
if [ ${#ALL_SUBJECT_PATHS[@]} -eq 0 ]; then
    echo "No subjects found. Exiting."
    exit 1
fi

# Calculate the index for this job array task
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Get the current subject path and extract variables
CURRENT_PATH=${ALL_SUBJECT_PATHS[$INDEX]}
SESSION=$(echo ${CURRENT_PATH} | rev | cut -d'/' -f2 | rev)
SUBJECT_ID=$(basename ${CURRENT_PATH} | cut -d'-' -f2)

# Set directories for this specific task
INPUT_DIR=${INPUT_BASE}/${SESSION}
OUTPUT_DIR=${OUTPUT_BASE}/${SESSION}

echo "*************** Postprocessing subject ${SUBJECT_ID} session ${SESSION} ***************"

singularity run --contain --cleanenv \
    -B ${INPUT_DIR}:/input \
    -B ${OUTPUT_DIR}:/out \
    ${POSTPROC_SIMG} \
    --bids_dir /input \
    --out_dir /out \
    --subject_id ${SUBJECT_ID} \
    --strategy ICAAROMA2Phys1GS \
    --dummy 10 \
    --FWHM 6 \
    --LPF 0.1 \
    --HPF 0.01

echo "Postprocessing completed for subject ${SUBJECT_ID}"