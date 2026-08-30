# Processing Scripts

All scripts live in `CLASSIFIER/src/processing/`. They generate functional connectivity (FC) matrices for each network experiment. Run from the **repository root**.

---

## Quick Start — Run Everything

```bash
# Schaefer-only experiments (no fMRI files needed, fast):
python -m CLASSIFIER.src.processing.run_all_processing

# All experiments including hippocampus (requires Tian atlas NIfTI):
python -m CLASSIFIER.src.processing.run_all_processing \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt
```

---

## Scripts

### `run_all_processing.py`

Master orchestrator. Runs all Schaefer-only subset experiments sequentially, then prints the commands for Tian-atlas experiments (or runs them if `--tian-atlas` is provided).

**Arguments:**
| Flag | Description | Default |
|---|---|---|
| `--tian-atlas` | Path to Tian Scale II atlas NIfTI | None (skip Tian jobs) |
| `--tian-labels` | Path to Tian label text file | None |
| `--skip-schaefer` | Skip the fast Schaefer subset jobs | False |

---

### `subset_schaefer_networks.py`

Slices the existing whole-brain 200×200 correlation matrices (`__v3__`) to extract a subset of Schaefer network parcels. **Does not access fMRI files** — operates on pre-computed `.npz` matrices only.

**When to use:** Any experiment that uses only Schaefer cortical networks (no hippocampus).

**Generates:** `__v6__` (Limbic), `__v7__` (DAN), `__v9__` (DMN+Limbic)

```bash
# Limbic only → __v6__
python -m CLASSIFIER.src.processing.subset_schaefer_networks \
    --networks Limbic \
    --output-version __v6__ \
    --output-suffix limbic

# Dorsal Attention Network → __v7__
python -m CLASSIFIER.src.processing.subset_schaefer_networks \
    --networks DorsAttn \
    --output-version __v7__ \
    --output-suffix dorsal_attention

# DMN + Limbic → __v9__
python -m CLASSIFIER.src.processing.subset_schaefer_networks \
    --networks Default Limbic \
    --output-version __v9__ \
    --output-suffix dmn_limbic
```

**Available Schaefer networks (200 ROI, 7 Yeo):**
| Network | Parcels | Biological role |
|---|---|---|
| `Default` | 46 | Default Mode Network (DMN) |
| `Limbic` | 12 | Limbic cortex (OFC, temporal pole) |
| `DorsAttn` | 26 | Dorsal Attention Network (IPS, FEF) |
| `Cont` | 30 | Frontoparietal control |
| `SomMot` | 35 | Somatomotor |
| `Vis` | 29 | Visual |
| `SalVentAttn` | 22 | Salience / Ventral Attention |

---

### `process_using_schaeffer_atlas.py`

Original whole-brain processing script. Reads resting-state BOLD NIfTI files, applies the full Schaefer 200 ROI atlas, and saves 200×200 Pearson FC matrices. Output goes to `__v3__`.

**Only needs to be run once** (output already exists in `__v3__`).

```bash
python -m CLASSIFIER.src.processing.process_using_schaeffer_atlas
```

---

### `process_using_tian_atlas.py`

Reads resting-state BOLD NIfTI files and extracts hippocampal time series using the **Tian subcortical atlas (Scale II)**. Saves 4×4 FC matrices (bilateral hippocampus, 2 sub-regions per hemisphere) to `__v5__`.

**Requires:** Tian atlas NIfTI + label file. Download from https://github.com/yetianmed/subcortex

```bash
python -m CLASSIFIER.src.processing.process_using_tian_atlas \
    --atlas-path /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --labels-path /path/to/Tian_Subcortex_S2_3T_label.txt
```

**Output:** `DATA/DELCODE/__v5__/matrices/*_hippocampus_correlation_matrix[_z_transformed].npz`

---

### `process_combined_schaefer_tian.py`

Produces **joint FC matrices** by running both a Schaefer masker (cortical network subset) and the Tian hippocampus masker on the same BOLD file, concatenating the time series before computing FC. This is the correct approach for any experiment involving hippocampus + cortical networks, because it preserves cross-region (e.g. DMN ↔ Hippocampus) connectivity.

**Requires:** BOLD NIfTI files + Tian atlas. Generates `__v8__`, `__v10__`, `__v11__`.

```bash
# DMN + Hippocampus → __v8__ (50 ROIs)
python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \
    --networks Default \
    --output-version __v8__ \
    --output-suffix dmn_hippo \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt

# DMN + Limbic + Hippocampus → __v10__ (62 ROIs)
python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \
    --networks Default Limbic \
    --output-version __v10__ \
    --output-suffix dmn_limbic_hippo \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt

# All combined (DMN + Limbic + DAN + Hippocampus) → __v11__ (88 ROIs)
python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \
    --networks Default Limbic DorsAttn \
    --output-version __v11__ \
    --output-suffix all_combined \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt
```

---

## Data Version Summary

| Version | Description | ROIs | Script | Source |
|---|---|---|---|---|
| `__v3__` | Whole brain | 200 | `process_using_schaeffer_atlas.py` | Raw fMRI |
| `__v4__` | DMN only | 46 | *(existing)* | Raw fMRI |
| `__v5__` | Hippocampus only | 4 | `process_using_tian_atlas.py` | Raw fMRI |
| `__v6__` | Limbic only | 12 | `subset_schaefer_networks.py` | `__v3__` |
| `__v7__` | Dorsal Attention only | 26 | `subset_schaefer_networks.py` | `__v3__` |
| `__v8__` | DMN + Hippocampus | 50 | `process_combined_schaefer_tian.py` | Raw fMRI |
| `__v9__` | DMN + Limbic | 58 | `subset_schaefer_networks.py` | `__v3__` |
| `__v10__` | DMN + Hippo + Limbic | 62 | `process_combined_schaefer_tian.py` | Raw fMRI |
| `__v11__` | All combined | 88 | `process_combined_schaefer_tian.py` | Raw fMRI |

Each version directory contains a `README.md` with its biological rationale and exact generation command.
