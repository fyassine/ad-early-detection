# Open questions (non-blocking)

- **Multiple fieldmaps per subject (RESOLVED)**: this SAMPLE has two GRE fieldmaps
  (`delcode_Fieldmap_3.5iso` at 64×64 and `delcode_Fieldmap_3iso` at 80×80). An early
  `build_bids.py` bug paired magnitude images across the two acquisitions, which crashed
  fMRIPrep's magnitude-merge (`shape (80,80,48) not compatible with (64,64,47)`). Fixed:
  `build_bids.py` now groups magnitudes/phasediffs by voxel geometry and emits only the single
  fieldmap matching the BOLD's resolution (the 3.5iso one here); other fieldmaps are dropped and
  logged. If a future subject legitimately needs the non-matching fieldmap, revisit the
  selection heuristic.

- **recon-all (RESOLVED)**: running with `--fs-no-reconall`. Your documentation didn't specify
  recon-all, and the original institutional outputs are all volumetric MNI-space with no
  surface outputs — so surfaces are skipped for speed. Revisit only if surface-based
  parcellation is later required (see `scripts/03_fmriprep/README.md`).

- **`--dummy-scans` value**: exact N for the dzne_RestingState_3.5iso protocol (TR=2.58s) is
  not yet confirmed — need the acquisition protocol PDF or scanner console settings. Default
  used in examples is `0`; verify before a real multi-subject run.
- **Bandpass band**: defaulting to 0.01–0.1 Hz (standard rs-fMRI), not yet confirmed against
  what the original Glioma analyses actually used.
- **`Final_change_foldernames_final.sh` equivalent**: not recreated. `build_bids.py`'s output
  naming is correct BIDS from the start, so this step is likely unnecessary — only write a new
  version if a concrete downstream consumer still expects the old institutional foldername
  scheme.
- **Container version pins**: MRIQC pinned to 24.0.2 in `containers/pull_containers.sh` — bump
  deliberately and document the change here if a different version is needed.
- **fMRIPrep 20.2.7 (LTS) vs. fmripost-aroma migration**: `--use-aroma` was removed from
  fMRIPrep >=23.1, so the pipeline currently pins the 20.2.7 LTS line specifically to keep
  AROMA support. If 20.2.7 becomes impractical (e.g. missing newer fixes), the alternative is
  running a current fMRIPrep without `--use-aroma` and adding `fmripost-aroma` as a separate
  step — not implemented, would need its own Slurm script and confound-matrix wiring.
- **T1w ND vs non-ND**: `series_classification.py` defaults to promoting the non-ND
  (distortion-corrected, confirmed via DIS3D/DIS2D ImageType tags in the SAMPLE subject) MPRAGE
  as the canonical `T1w`, keeping ND as a `rec-ND` variant. Same pattern applied to FLAIR.
  Revisit if FreeSurfer surface quality from the distortion-corrected variant is unexpectedly
  poor.
- **Unmapped delcode series** (FLASH, IR-EPI, T2, and a 3iso fieldmap pair with no geometry
  match to any func/dwi run in the SAMPLE subject) are left out of the BIDS tree by
  `series_classification.py` and logged by `build_bids.py` rather than guessed at. If any of
  these turn out to be needed, add a rule in `series_classification.py`.
- **Reorientation byte-parity with FSL**: `scripts/05_reorientation/final_reorient_nibabel.py`
  has not been validated against real `fslswapdim`/`fslorient -forceradiological` output, since
  FSL isn't installed on this system. Sanity-checked via before/after `aff2axcodes` only.
- **Session/visit labeling**: the SAMPLE subject directory name `03a0a6663-M0_T1_01` hints at a
  longitudinal naming scheme (`M0` = baseline visit?). Pipeline currently defaults every
  subject to `ses-1`; if this is really a multi-timepoint cohort, `build_bids.py`'s
  `--session` argument should be driven by a parsed visit label instead of a hardcoded default.
