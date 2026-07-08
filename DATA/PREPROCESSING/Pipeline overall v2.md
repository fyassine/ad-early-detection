# Pipeline

Note that the following scripts are given as examples for the required steps in the main pipeline (Neuroimaging basics). They are not meant to be run directly, but to illustrate some the computations and goal outputs.
Always make sure the code is adjusted for the correct paths, directory structures, filenames and accounts.

Steps:

1. **Prepare data format.**

**Goal**: All data is in Nifti format and each recording fMRI session has a single image.

1.1)  Convert the data from DICOM to Nifti format using dcm2niix package.

1.2)  If the resting state of a session is separated into two runs, merge into one image.

ex.)  /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/fslmerge3.py

1.3)  If needed, remove directories with empty anat folders:

ex.) find /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/BIDS3 -type d -exec sh -c 'anat_dir="$1/anat"; if [ -d "$anat_dir" ] && [ -z "$(ls -A "$anat_dir")" ]; then echo "Deleting empty directory: $1"; rm -r "$1"; else echo "Skipping non-empty directory: $1"; fi' _ {} \;

1. **Organise the directories to BIDS.** More information about the structure: https://bids.neuroimaging.io/tools/validator.html.

ex.)  Refer to the functions in the file below, but adjust to match the structure and filenames of your raw data.

/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/BIDS_og.py

2.1)  If a change to the folder names is needed:

ex.)   **/data2/core-rad-fni/swunderl/Glioma_Sophia/**Final_change_foldernames_final.sh

2.2)  Ensure the file ‘dataset_description.json’ is present inside each subject folder.

ex.)  for data in /data2/core-rad-fni/path-to-your-project/data/*; do cp dataset_description.json "$data"; done

1. **Preprocess using fMRIPrep (CORE):**

Note: Ensure the correct fMRIPrep version is used.

Ex.) `/data2/core-rad/smoteval/HGG_Teil3/slurms/fmriprep_array_v1.slurm`

- Ensure preprocessing finishes correctly, check output **.log* and **.err* files.

**4. Postprocessing (CORE):** 

ex.)   ****/data2/core-rad-fni/swunderl/Glioma_random/rad_postprocessing2.slurm 

5.  Transfer to LRZ and proceed with analysis.

ex.)  rsync -avuzh final_data/ di76gez@@cool.hpc.lrz.de://dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_random

6. **Reorient (LRZ):**

**Ex.)**  /dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_random/final_reorient.py

(https://nipy.org/nibabel/neuro_radio_conventions.html)

Requirements:

- FSL: https://fsl.fmrib.ox.ac.uk/fsl/docs/install/linux.html

Note: After installation paths need to be set (adjusted for your account):

`FSLDIR=/dss/dsshome1/lxc0C/di76gez/fsl`

`PATH=${FSLDIR}/share/fsl/bin:${PATH}`

`export FSLDIR PATH`