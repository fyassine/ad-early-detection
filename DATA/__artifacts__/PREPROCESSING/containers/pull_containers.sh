#!/usr/bin/env bash
# Pull pinned MRIQC / fMRIPrep containers via apptainer.
# Run from this directory: bash pull_containers.sh
#
# Versions are pinned (not :latest) for reproducibility across the cohort. Bump these
# deliberately and re-pull when you need a newer release; record the change in
# docs/PIPELINE_OVERVIEW.md so all subjects in a given analysis run used the same version.
set -euo pipefail

module load apptainer/1.3.4

MRIQC_VERSION="${MRIQC_VERSION:-24.0.2}"
# fMRIPrep deprecated/removed --use-aroma starting at 23.1 (AROMA moved to the separate
# fmripost-aroma BIDS-app). Since the institutional confound strategy requires --use-aroma,
# pin the 20.2.7 LTS line, which still has it built in and is what produces the
# desc-smoothAROMAnonaggr_bold.nii.gz / AROMAnoiseICs.csv naming the original pipeline expects.
FMRIPREP_VERSION="${FMRIPREP_VERSION:-20.2.7}"

cd "$(dirname "$0")"

echo "Pulling MRIQC ${MRIQC_VERSION}..."
apptainer pull --force "mriqc-${MRIQC_VERSION}.sif" "docker://nipreps/mriqc:${MRIQC_VERSION}"

echo "Pulling fMRIPrep ${FMRIPREP_VERSION}..."
apptainer pull --force "fmriprep-${FMRIPREP_VERSION}.sif" "docker://nipreps/fmriprep:${FMRIPREP_VERSION}"

echo "Done. Update scripts/02_mriqc/submit_mriqc.slurm and scripts/03_fmriprep/submit_fmriprep.slurm"
echo "if these version numbers change."
