# __v6__ — Limbic Only (Schaefer 200)

**Condition:** Limbic network only  
**Hypothesis:** H2 baseline — does limbic cortex alone carry AD signal?

## Atlas
- **Source:** Schaefer 2018, 200 ROIs, 7 Yeo networks
- **ROIs:** 12 parcels — bilateral limbic cortex (orbitofrontal + temporal pole)
- **Matrix shape:** 12×12 Pearson FC (raw + Fisher z-transformed)
- **Parent version:** Extracted from `__v3__` (200×200 whole-brain matrices)

## Files
- `matrices/*_limbic_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_limbic_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — Schaefer parcel names for the 12 Limbic ROIs

## Generation
```
python -m CLASSIFIER.src.processing.subset_schaefer_networks \
    --networks Limbic \
    --output-version __v6__ \
    --output-suffix limbic
```

## Biological rationale
Schaefer Limbic parcels (entorhinal/parahippocampal cortex, temporal pole, OFC) are
anatomically adjacent to the hippocampus and consistently show early atrophy in AD.
Tested in isolation before combining with DMN to isolate incremental contribution.
