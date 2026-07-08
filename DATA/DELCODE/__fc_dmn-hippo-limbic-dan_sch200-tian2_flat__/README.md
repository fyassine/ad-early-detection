# __v11__ — All Combined: DMN + Hippocampus + Limbic + DAN (Schaefer 200 + Tian Scale II)

**Condition:** All networks combined  
**Hypothesis:** H1+H2+H3 — does adding DAN to the memory system help or add noise?

## Atlas
- **Cortical:** Schaefer 2018, 200 ROIs — Default (46) + Limbic (12) + DorsAttn (26) = 84 parcels
- **Subcortical:** Tian Scale II — bilateral hippocampus (4 parcels)
- **ROIs:** 88 total (84 cortical + 4 hippocampus)
- **Matrix shape:** 88×88 Pearson FC (raw + Fisher z-transformed)

## Files
- `matrices/*_all_combined_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_all_combined_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — ordered parcel names [DMN, Limbic, DorsAttn, then Tian hippocampus]

## Generation
```
python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \
    --networks Default Limbic DorsAttn \
    --output-version __v11__ \
    --output-suffix all_combined \
    --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \
    --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt
```

## Biological rationale
Includes the antagonistic H3 network (DAN) alongside the full H2 memory circuit.
Compared to __v10__, any AUC improvement attributable to DAN suggests the
attentional system carries independent AD classification signal; no improvement
(or degradation) confirms DAN is noise in this context.
