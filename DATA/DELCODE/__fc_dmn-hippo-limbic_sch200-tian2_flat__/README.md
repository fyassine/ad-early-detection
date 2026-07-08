# __v10__ — DMN + Hippocampus + Limbic (Schaefer 200 + Tian Scale II)

**Condition:** Default Mode Network + Hippocampus + Limbic network  
**Hypothesis:** H2 full — complete DMN/memory system extension

## Atlas
- **Cortical:** Schaefer 2018, 200 ROIs — Default (46) + Limbic (12) = 58 parcels
- **Subcortical:** Tian Scale II — bilateral hippocampus (4 parcels)
- **ROIs:** 62 total (58 cortical + 4 hippocampus)
- **Matrix shape:** 62×62 Pearson FC (raw + Fisher z-transformed)

## Files
- `matrices/*_dmn_limbic_hippo_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_dmn_limbic_hippo_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — ordered parcel names [DMN, Limbic, then Tian hippocampus]

## Generation
```
python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \
    --networks Default Limbic \
    --output-version __v10__ \
    --output-suffix dmn_limbic_hippo \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt
```

## Biological rationale
Full H2 hypothesis: DMN + the complete AD memory circuit (limbic cortex + hippocampus).
If this outperforms __v8__ (DMN+Hippo) and __v9__ (DMN+Limbic) individually,
it confirms that both extensions contribute independent signal.
