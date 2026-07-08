# Parcellation — out of scope here

Parcellation is handled separately by the project owner. This stage is not implemented.

**Expected input**: the reoriented final BOLD images produced by stage 5, e.g.
`sub-<ID>_ses-<N>_task-rest_run-<N>_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz`.

Since the data lives in `MNI152NLin2009cAsym` space and `nilearn` is already in the env
(`ad-early-detection`), an atlas-based volumetric approach via
`nilearn.maskers.NiftiLabelsMasker` would be a natural fit if recon-all surfaces end up not
being used — but this is the project owner's call, not implemented here.
