# __v5__ — Hippocampus Only (Tian Subcortical Atlas)

**Condition:** Hippocampus only  
**Hypothesis:** H2 baseline — does hippocampal connectivity alone carry AD signal?

## Atlas
- **Source:** Tian Subcortical Atlas, Scale II (Tian et al. 2020, Nature Neuroscience)
- **ROIs:** 4 parcels — bilateral hippocampus (2 subdivisions per hemisphere)
- **Matrix shape:** 4×4 Pearson FC (raw + Fisher z-transformed)

## Files
- `matrices/*_hippocampus_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_hippocampus_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — Tian parcel names for the selected hippocampal ROIs

## Generation
```
python -m CLASSIFIER.src.processing.process_using_tian_atlas \
    --atlas-path /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --labels-path /path/to/Tian_Subcortex_S2_3T_label.txt
```

## Biological rationale
The hippocampus is the primary site of early tau accumulation in AD (Braak stage I-II)
and is not parcellated in the Schaefer cortical-only atlas. The Tian Scale II atlas
provides finer hippocampal subdivisions than a single bilateral ROI. Tested in isolation
to establish the standalone signal before combining with DMN.
