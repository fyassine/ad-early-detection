# Glioma Resting-State fMRI Preprocessing Pipeline

dcm2niix → BIDS → MRIQC → fMRIPrep (`--use-aroma`, `--dummy-scans`) → confound regression +
bandpass → late reorientation → parcellation (out of scope, separate).

See `docs/PIPELINE_OVERVIEW.md` for the full stage diagram and `docs/OPEN_QUESTIONS.md` for
unresolved items. `original/README.md` documents the 3 institutional scripts this reuses/fixes
and which 3 scripts were unreachable (CORE cluster) and rewritten from scratch.

Quick start: `bash tests/test_pipeline_sample.sh` runs the interactive stages end-to-end on
`SAMPLE/03a0a6663-M0_T1_01`.
