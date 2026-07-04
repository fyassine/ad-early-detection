# Origin of DELCODE fMRI NIfTI Files

## 1. Original Source (LRZ Cluster)
All processed fMRI functional images (`_bold_reoriented.nii.gz`) originated from the Leibniz Supercomputing Centre (LRZ) HPC cluster:

* **Remote Host:** `cool.hpc.lrz.de`
* **User account:** `di54lup`
* **Remote Source Directories:**
  * **For converters/MCI/SCD:** `/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Delcode/data/Converter_newcriteria/postprocessed`
  * **For non-converters:** `/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Delcode/data/non-converter/postprocessed`

The files were transferred using SSH master connections and rsync tunnels, routed via the intermediate `wunderlich` server (`138.245.113.9`).

---

## 2. Copying Missing fMRI Scans & File Origins
Missing fMRI scans are copied from the LRZ cluster using the transfer scripts (such as [transfer_resting_state_from_lrz.py](file:///mnt/e/fyassine/ad-early-detection/DATA/DELCODE/src/transfering/transfer_resting_state_from_lrz.py)). Based on the transfer script, the dataset pipeline, and file timestamps, here is the origin of each file:

### A. Preprocessed Scan Files (from LRZ Cluster)
All `.nii.gz` functional images originated from the Leibniz Supercomputing Centre (LRZ) HPC cluster (`cool.hpc.lrz.de`) under user `di54lup`'s project space. They were preprocessed using a BIDS-compliant fmriprep-like pipeline.

* **The March 6, 2025 version** (no visit tag `_M0_` in filename):
  This was preprocessed on the cluster on March 6, 2025 and transferred to `/mnt/e/fyassine/_ad-early-detection/` as part of the initial dataset collection.
* **The M0 baseline version (`_M0_`):**
  * **File:** `sub-073b63746_ses-01_M0...nii.gz`
  * **Origin:** Preprocessed on the LRZ cluster on March 17, 2026, 00:33:43 CET (using updated naming conventions to include `_M0_`). It was copied locally today (July 2, 2026, 15:08:19 CEST).
