# __v7__ — Dorsal Attention Network Only (Schaefer 200)

**Condition:** Dorsal Attention Network (DAN) only  
**Hypothesis:** H3 — competing/antagonistic network as alternative to DMN

## Atlas
- **Source:** Schaefer 2018, 200 ROIs, 7 Yeo networks
- **ROIs:** 26 parcels — bilateral DAN (IPS, FEF, MT+)
- **Matrix shape:** 26×26 Pearson FC (raw + Fisher z-transformed)
- **Parent version:** Extracted from `__v3__` (200×200 whole-brain matrices)

## Files
- `matrices/*_dorsal_attention_correlation_matrix.npz` — raw Pearson correlation
- `matrices/*_dorsal_attention_correlation_matrix_z_transformed.npz` — Fisher z-transformed
- `parcel_labels.txt` — Schaefer parcel names for the 26 DorsAttn ROIs

## Generation
```
python -m CLASSIFIER.src.processing.subset_schaefer_networks \
    --networks DorsAttn \
    --output-version __v7__ \
    --output-suffix dorsal_attention
```

## Biological rationale
The DAN (intraparietal sulcus, frontal eye fields) shows an inverse relationship with
the DMN during attention-demanding states ("task-positive network"). Tested separately
as an antagonistic hypothesis to H1/H2 — DAN disruption may reflect compensatory
mechanisms or independent disease processes rather than the DMN signature.
