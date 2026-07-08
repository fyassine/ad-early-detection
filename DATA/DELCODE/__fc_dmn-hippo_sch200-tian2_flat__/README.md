# __v8__ — DMN + Hippocampus (Schaefer 200 + Tian Scale II)

**Condition:** Default Mode Network + Hippocampus  
**Hypothesis:** H2 — DMN extended with hippocampal memory circuitry

## Atlas
- **Cortical:** Schaefer 2018, 200 ROIs — Default network (46 parcels)
- **Subcortical:** Tian Scale II — bilateral hippocampus (4 parcels)
- **ROIs:** 50 total (46 DMN + 4 hippocampus)
- **Matrix shape:** 50×50 Pearson FC (raw + Fisher z-transformed)

## Files
- `matrices/*_dmn_hippo_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_dmn_hippo_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — ordered parcel names [DMN parcels, then Tian hippocampus]

## Generation
```
python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \
    --networks Default \
    --output-version __v8__ \
    --output-suffix dmn_hippo \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt
```

**Note:** Time series from both maskers are concatenated before FC computation,
preserving DMN↔Hippocampus cross-region connectivity in the off-diagonal blocks.

## Biological rationale
The hippocampus is anatomically and functionally coupled with the DMN (PCC, mPFC,
angular gyrus). AD-related tau spread follows the entorhinal→hippocampus→DMN
trajectory. This version tests whether adding hippocampal nodes to the DMN graph
improves classification over DMN alone.
