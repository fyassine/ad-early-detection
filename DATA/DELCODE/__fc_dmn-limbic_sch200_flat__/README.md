# __v9__ — DMN + Limbic (Schaefer 200)

**Condition:** Default Mode Network + Limbic network  
**Hypothesis:** H2 — DMN extended with limbic cortex

## Atlas
- **Source:** Schaefer 2018, 200 ROIs, 7 Yeo networks
- **ROIs:** 58 parcels — Default (46) + Limbic (12)
- **Matrix shape:** 58×58 Pearson FC (raw + Fisher z-transformed)
- **Parent version:** Extracted from `__v3__` (200×200 whole-brain matrices)

## Files
- `matrices/*_dmn_limbic_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_dmn_limbic_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — Schaefer parcel names for the 58 selected ROIs

## Generation
```
python -m CLASSIFIER.src.processing.subset_schaefer_networks \
    --networks Default Limbic \
    --output-version __v9__ \
    --output-suffix dmn_limbic
```

## Biological rationale
Limbic cortex (entorhinal, parahippocampal, OFC, temporal pole) forms the cortical
gateway between the DMN and medial temporal memory structures. This version tests
whether adding Schaefer Limbic parcels improves over DMN alone, as a purely
cortical extension of the AD hypothesis.
