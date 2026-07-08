Basically this is the same as in “Neuroimaging Basics”

**Pipeline:**

Steps:

0.)  /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/fslmerge3.py

only to delete the ones without .nii.gz in anat but does not work

(BIDS_onlyniigz_files3.py,

0.5) find /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/BIDS3 -type d -exec sh -c 'anat_dir="$1/anat"; if [ -d "$anat_dir" ] && [ -z "$(ls -A "$anat_dir")" ]; then echo "Deleting empty directory: $1"; rm -r "$1"; else echo "Skipping non-empty directory: $1"; fi' _ {} \;

find /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/BIDS3 -type d -exec sh -c 'if [ "$(find "$1" -maxdepth 1 -type f -iname "*.nii" | wc -l)" -eq 1 ]; then echo "Deleting parent directory: $1"; rm -r "$1"; else echo "Skipping directory: $1"; fi' _ {} \;)

1.) /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData

BIDS_og.py (but this sometimes takes the reoriented one and sometimes not)

2.)**/data2/core-rad/swunderl/Glioma_Sophia** Final_change_foldernames_final.sh

2.a) for data in /data2/core-rad/swunderl/Glioma_Sophia/data/*; do cp dataset_description.json "$data"; done

3.) run fmrirep final_fmirprep.slurm /**data2/core-rad/swunderl/Glioma_Sophia (**sbatch --account=core-psy --array=2-55%7 final_fmirprep.slurm)

- Maybe rerun again if not finished correctly etc. check output files!!! there should be “no error” in the end

(3.a) MRIQC gibts ein image auf dem Core)

(3b.) Motor Seed maps DMN maps etc.)

**4.) postprocessing: use** /data2/core-rad/swunderl/Glioma_random/rad_postprocessing2.slurm over chenyangs or other account in mine there is sth wrong:

CAVE: no „sub-„ in this code rad_postprocessing2.slurm, no ses in the folder structure! —> run **/data2/core-rad/swunderl/Glioma_random**$ sh delete_ses.sh

ODER

in LRZ:

/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_random

4b) transfer to LRZ: rsync -avuzh final_data/ di76gez@lxlogin4.lrz.de://dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_random

**5.) reorient (LRZ) e.g.** /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_random/final_reorient.py

(https://nipy.org/nibabel/neuro_radio_conventions.html)

—>This script needds fsl which can has to be initialised like that in my account:

(paths need to be adjusted with your account)

FSLDIR=/dss/dsshome1/lxc0C/di76gez/fsl

PATH=${FSLDIR}/share/fsl/bin:${PATH}

export FSLDIR PATH

. ${FSLDIR}/etc/fslconf/fsl.sh

—>This script needs freesurfer which can has to be initialised like that in my account:

export FREESURFER_HOME=/dss/dsshome1/lxc0C/di76gez/freesurfer

source $FREESURFER_HOME/SetUpFreeSurfer.sh

(5a) Motor Seed maps DMN maps etc.)

FOR THESE STEPS BETTER USE THE SCRIPTS IN “Neuroimaging Basics”