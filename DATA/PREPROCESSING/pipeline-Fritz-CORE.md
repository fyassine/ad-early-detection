# Fritz → CORE Preprocessing Pipeline Plan
_Last updated: 2026-07-06_

## Status

| Section | Status |
|---------|--------|
| Steps 1–2 Fritz script | ✅ Written (`src/fritz/run_fritz_pipeline.sh`) |
| README updated | ✅ Done |
| ADNI series heuristics | ✅ **Verified** — full scan of all 732 zips complete |
| ADNI T1w availability | ❌ **No T1w in current download** — fMRI only |
| OASIS3 T1w availability | ⏳ **Pending** — check OASIS3 portal for paired T1w |
| Steps 3–4 CORE SLURM scripts | ✅ Reference scripts in `src/core/` |

---

## Execution split

| Step | Description | Machine |
|------|-------------|---------|
| 1.1 | DICOM → NIfTI (`dcm2niix`) | **Fritz** (ADNI only) |
| 1.2 | Merge fMRI runs (`fslmerge`) | **Fritz** (OASIS3 multi-run) |
| 1.3 | Remove empty `anat/` dirs | **Fritz** |
| 2 | BIDS organisation | **Fritz** |
| 2.1 | Rename BIDS folders | **Fritz** |
| 2.2 | Copy `dataset_description.json` | **Fritz** |
| → | **rsync BIDS → CORE** | Fritz → CORE |
| 3 | fMRIPrep (Singularity) | **CORE** |
| 4 | Postprocessing (Singularity) | **CORE** |

---

## Dataset notes

### OASIS3

- Raw data: `DATA/OASIS3/__bold_and_smri__/sub-*/ses-*/func/`
- Already in NIfTI format with BIDS-like naming
- Some sessions have multiple runs (`run-01`, `run-02`, `run-03`)
  - `run-01` is often very short (scout/localiser); runs < 50 TRs are filtered out
  - Remaining runs are merged with `fslmerge -t`
- **No `anat/` folder present in current download**

> ⏳ **Action needed**: Verify whether paired T1w (MPRAGE) data can be
> downloaded from the OASIS3 portal for these subjects. If yes, add them to
> the BIDS `anat/` folder. If no, fMRIPrep must run with `--fs-no-reconall`.

---

### ADNI

- Raw data: `DATA/ADNI/__dicom_zips_flat__/` — flat folder of 732 `.zip` files
- Naming convention: `<SiteID>_S_<SubjectID>_<SeriesID>.zip`
  - e.g. `002_S_1261_831069.zip` → subject `002_S_1261`, series `831069`
- Each zip contains exactly **one series** (one scan per zip)
- **BIDS subject ID mapping**: `002_S_1261` → `sub-ADNI002S1261` ✅

#### Confirmed series types (full scan of all 732 zips, 2026-07-06)

| Series name | Type |
|-------------|------|
| `Axial_rsfMRI__Eyes_Open_` | resting-state BOLD |
| `Axial_rsfMRI__EYES_OPEN_` | resting-state BOLD |
| `Axial_rsFMRI_Eyes_Open` | resting-state BOLD |
| `Axial_rsfMRI__Eyes_Open__-phase_P_to_A` | resting-state BOLD (PA phase enc.) |
| `Axial_MB_rsfMRI__Eyes_Open_` | multiband resting-state BOLD |
| `Axial_MB_rsfMRI_AP` | multiband BOLD (AP phase enc.) |
| `Axial_MB_rsfMRI__Eyes_Open____straight_no_angle` | multiband BOLD |
| `Axial_fcMRI__Eyes_Open_` | functional connectivity MRI |
| `Axial_fcMRI__EYES_OPEN_` | functional connectivity MRI |
| `Axial_fcMRI_0_angle__EYES_OPEN_` | fcMRI |
| `Axial_fcMRI` | fcMRI |
| `Axial_RESTING_fcMRI__EYES_OPEN_` | fcMRI |
| `Resting_State_fMRI` | resting-state BOLD (older protocol) |
| `Extended_Resting_State_fMRI` | resting-state BOLD (extended) |

> ❌ **No T1w / sMRI in current download.** All 732 zips are fMRI only.
> fMRIPrep will need to run with `--fs-no-reconall` for ADNI, or a separate
> T1w download is required from ADNI (search for MPRAGE/3D T1 series).

**BOLD detection regex** (in `run_fritz_pipeline.sh`):
```
grep -iE "rsfmri|fcmri|fmri|resting|bold|rest"
```
This covers all confirmed series names above.

---

## Fritz script usage

```bash
# Both datasets + rsync to CORE
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both

# Single dataset
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset oasis3
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset adni

# Dry-run (no rsync)
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both --dry-run
```

SSH config required on Fritz (`~/.ssh/config`):
```
Host HOST
    HostName srvcorem2.med.uni-muenchen.de
    User flakhal
```
Ensure key-based SSH auth: `ssh-copy-id flakhal@HOST`

---

## CORE SLURM scripts

| Script | Step | Key flags to adapt |
|--------|------|--------------------|
| `src/core/fmriprep_array_v1.slurm` | Step 3 | `INPUT_BASE`, `OUTPUT_BASE`, `FS_LICENSE`, `--fs-no-reconall` if no T1w |
| `src/core/postprocessing_array_v1.slurm` | Step 4 | `INPUT_BASE`, `OUTPUT_BASE` |

---

## Open items

- [ ] Verify OASIS3 T1w — download paired MPRAGE from OASIS3 portal, or confirm `--fs-no-reconall`
- [ ] Decide whether to download ADNI T1w separately, or run fMRIPrep without structural
- [ ] Confirm CORE destination path (`/home/flakhal/` vs `/data2/` project space)
