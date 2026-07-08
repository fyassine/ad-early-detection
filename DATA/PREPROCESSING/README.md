# Preprocessing Pipeline — AD Early Detection

> **Reference pipeline:** [Pipeline overall v2.md](Pipeline%20overall%20v2.md)

---

## Where each step runs

| Step | Description | Machine | Notes |
|------|-------------|---------|-------|
| **1.1** | DICOM → NIfTI (`dcm2niix`) | **Fritz** | Install `dcm2niix`; only needed for ADNI (OASIS3 bold is already NIfTI/BIDS-structured) |
| **1.2** | Merge fMRI runs (`fslmerge`) | **Fritz** | FSL 6.0.7 is installed; applies to OASIS3 multi-run sessions |
| **1.3** | Remove empty `anat/` dirs (`find`) | **Fritz** | Pure bash |
| **2** | BIDS organisation | **Fritz** | stdlib only; restructures data into BIDS hierarchy |
| **2.1** | Rename BIDS folders | **Fritz** | Pure bash |
| **2.2** | Copy `dataset_description.json` | **Fritz** | Trivial bash |
| **→** | **rsync BIDS output → CORE** | Fritz → CORE | `rsync -avuzh` to `/home/flakhal/` on CORE |
| **3** | fMRIPrep (`fmriprep-23.0.2.simg`) | **CORE** | Singularity array job; `smoteval`'s image at `/data2/core-rad/smoteval/images/` |
| **4** | Postprocessing (`postprocessing.sif`) | **CORE** | Singularity array job; `flbrandl`'s image at `/data2/core-rad/flbrandl/` |
| **5** | Transfer to LRZ | CORE → LRZ | `rsync -avuzh` |
| **6** | Reorient | LRZ | `final_reorient.py`; requires FSL + FreeSurfer |

---

## Datasets

| Dataset | Raw format on Fritz | BIDS output location |
|---------|---------------------|----------------------|
| **OASIS3** | Already NIfTI, semi-BIDS in `DATA/OASIS3/__bold_and_smri__/sub-*/ses-*/func/` | Merge multi-run → full BIDS in `DATA/OASIS3/BIDS/` |
| **ADNI** | DICOM `.zip` files in `DATA/ADNI/__dicom_zips_flat__/` | dcm2niix → BIDS in `DATA/ADNI/BIDS/` |

---

## Fritz steps (1–2): quick-start

The orchestrator script for Fritz is:

```
DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh
```

It runs Steps 1–2 for **both** datasets (or a single one), then rsyncs the
BIDS output to CORE.

Usage:

```bash
# Run pipeline for both datasets and rsync to CORE
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both

# Run for OASIS3 only
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset oasis3

# Run for ADNI only
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset adni

# Dry-run (skip rsync)
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both --dry-run
```

---

## CORE steps (3–4): SLURM array jobs

Reference scripts (adapt paths before submitting):

| Script | Purpose |
|--------|---------|
| [`src/core/fmriprep_array_v1.slurm`](src/core/fmriprep_array_v1.slurm) | fMRIPrep array job |
| [`src/core/postprocessing_array_v1.slurm`](src/core/postprocessing_array_v1.slurm) | Postprocessing array job |

Submit example:

```bash
# On CORE — after BIDS data has arrived via rsync
sbatch --array=1-272%10 src/core/fmriprep_array_v1.slurm
```

---

## Requirements

| Tool | Where | Notes |
|------|-------|-------|
| `dcm2niix` | Fritz | `sudo apt install dcm2niix` or conda |
| FSL 6.0.7 | Fritz | Already installed; ensure `$FSLDIR` is set |
| Singularity / Apptainer | CORE | Available system-wide |
| FreeSurfer license | CORE | `/data2/core-rad/swunderl/freesurfer/license.txt` |

---

## Logs

Runtime logs from the Fritz script are written to `DATA/PREPROCESSING/logs/`.
