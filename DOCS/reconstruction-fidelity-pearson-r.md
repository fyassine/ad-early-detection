# Reconstruction-fidelity Pearson r

## Where it's computed

`CLASSIFIER/model/GAAE/explain.py::reconstruction_quality(x, x_recon)` is the single
canonical implementation. It is shared by both encoder adapters — the GAAE adapter
calls it directly, the VGAE adapter (`CLASSIFIER/adapters/explain.py::diagnostics`,
calling into `CLASSIFIER/model/VGAE/explain.py::reconstruct_adjacency`) calls the same
function on a different pair of matrices. There is no separate "VGAE version" of the
formula.

## The formula

Given the input matrix `x` and its reconstruction `x_recon` (same shape), both are
flattened to 1-D vectors `flat_a`, `flat_b` and the Pearson product-moment correlation
coefficient is computed between them:

```python
flat_a, flat_b = a.ravel(), b.ravel()
pearson_r = np.corrcoef(flat_a, flat_b)[0, 1]   # = cov(flat_a, flat_b) / (std(flat_a) * std(flat_b))
```

Equivalently:

```
        Σ (aᵢ - ā)(bᵢ - b̄)
r = ───────────────────────────────
     √(Σ (aᵢ - ā)²) · √(Σ (bᵢ - b̄)²)
```

where the sum runs over **every element of the flattened matrix** (all nodes × all
feature columns for GAAE; all N×N adjacency entries for VGAE) — it is not averaged
per-row first. `r` is undefined (`NaN`) if either matrix is constant (zero variance).

This is reported instead of, or alongside, absolute-error metrics (`mse`, `rmse`, `mae`,
`nrmse = rmse / std(x)`) because the FC z-score scale is dataset-specific; correlation
gives a scale-free read on whether the reconstruction tracks the input's pattern.

`reconstruction_quality` also reports `r2 = 1 - SS_res/SS_tot` (coefficient of
determination against the input's own mean) and bins `pearson_r` into a qualitative
label: `≥0.90 excellent · 0.80–0.90 good · 0.60–0.80 fair · <0.60 poor` — a rule of
thumb borrowed from parcellated-fMRI autoencoder literature, not a project-specific
threshold.

## What `x` / `x_recon` actually are — this differs by model

This is the detail that matters when comparing GAAE and VGAE numbers:

- **GAAE** (`reconstruct_features` in `model/GAAE/explain.py`): `x` is the input
  node-feature matrix — Fisher z-transformed FC rows, shape `(N_ROI, F)`, continuous,
  roughly zero-mean. `x_recon` is the GAT decoder's feature reconstruction, same shape
  and scale. Pearson r here asks "does the decoder reproduce the *connectivity
  profile* per ROI?"
- **VGAE** (`reconstruct_adjacency` in `model/VGAE/explain.py`): `x` is `adj_true`, the
  **binary** adjacency matrix (0/1, derived from `edge_index` via `to_dense_adj`),
  shape `(N_ROI, N_ROI)`. `x_recon` is `adj_hat = sigmoid(z @ zᵀ)`, a continuous
  edge-probability matrix in `[0, 1]`. Pearson r here asks "does the decoder's edge
  probability rank-track which ROI pairs are actually connected?"

  Note this is true even though the VGAE **model itself** also reconstructs node
  features when trained with `feature_decoder=True` (it does for the anticollapse
  configs below — `feature_loss_weight: 0.5`, i.e. half the training loss comes from
  `decode_features(z)` vs. the input FC rows, `model/VGAE/models.py:199-206`,
  `forward()` at `models.py:208-215`). But the **explain/diagnostics path never calls
  `decode_features`**: `VGAEAdapter.diagnostics` (`adapters/explain.py:441-472`) and
  `model/VGAE/explain.py::reconstruct_adjacency` / `trace_forward` only ever score
  `decode_all` (the adjacency head). So the VGAE fidelity number below tells you
  nothing about how well the model's own feature decoder reconstructs FC — that head
  is trained but never measured by this notebook.

These are **different reconstruction targets** (continuous FC features vs. binary
adjacency) computed with the identical correlation formula. The two `r` values are not
measuring the same thing and are not directly interchangeable as a "model A is better
than model B" signal.

## Cohort-level reporting

`adapter.diagnostics(bundle)` (`CLASSIFIER/adapters/explain.py`) calls
`reconstruction_quality` once per subject and aggregates `pearson_r` across the
cohort into `fidelity_median` and `fidelity_iqr` (25th/75th percentile), e.g.:

```python
fidelity_median = float(np.nanmedian(fidelity_r))
fidelity_iqr = [np.nanpercentile(fidelity_r, 25), np.nanpercentile(fidelity_r, 75)]
```

The `DATA_JOURNEY_*` notebooks run this on `CV_BUNDLE` (pooled train+val — the data the
encoder was fit on) and again on `TEST_BUNDLE` "for context," printing both.

## VGAE feature fidelity now measured directly (2026-06-24 update)

`reconstruction_quality` is identical either way, but originally `VGAEExplainAdapter
.diagnostics()` only ever called `reconstruct_adjacency` — it never called the VGAE's
own `decode_features(z)`, even for configs trained with `feature_decoder=True`. That
meant there was no way to compare VGAE's actual feature-reconstruction fidelity
against GAAE's; the 0.34 adjacency number above was not a fair stand-in for it (see
previous section).

Fixed by adding `model/VGAE/explain.py::reconstruct_features` (mirrors GAAE's
`reconstruct_features`, returns `None` if `model.has_feature_decoder` is `False`) and
wiring it into `diagnostics()`: when the encoder has a feature decoder, the dict now
also carries `feature_fidelity_r` / `feature_fidelity_median` / `feature_fidelity_iqr`,
computed by running `decode_features(z)` output through the same
`reconstruction_quality` call GAAE uses. The `DATA_JOURNEY` notebook prints and plots
this alongside the adjacency number when present.

## GAAE vs. VGAE — 2026-06-24 data-journey re-runs (DELCODE whole-brain, anticollapse configs)

| | n | Train+Val adjacency r | Test adjacency r | **Train+Val feature r** | **Test feature r** |
|---|---|---|---|---|---|
| GAAE (feature recon only) | 133 / 34 | — | — | **0.061** [0.014, 0.109] | **0.046** [0.026, 0.109] |
| VGAE-GCN-anticollapse | 133 / 34 | 0.341 [0.316, 0.357] | 0.319 [0.302, 0.338] | **-0.088** [-0.181, 0.014] | **-0.008** [-0.090, 0.121] |
| VGAE-GAT-anticollapse | 133 / 34 | 0.336 [0.303, 0.364] | 0.331 [0.306, 0.347] | **-0.028** [-0.127, 0.066] | **0.028** [-0.072, 0.103] |

The **feature** columns are the genuinely comparable ones (same target as GAAE:
continuous FC rows via `decode_features`/the GAT decoder). The **adjacency** columns
are VGAE-only and not comparable to GAAE at all (different decoder head, different,
easier target — see above).

Reading the comparable (feature) numbers: **VGAE's own feature decoder reconstructs
DELCODE FC worse than GAAE's**, not better. Train+val median feature r is essentially
zero or negative for both VGAE variants (GCN: -0.088, GAT: -0.028) versus GAAE's
positive 0.061 — i.e. the VGAE feature decoder's output is uncorrelated with (or
slightly anti-correlated with) the true FC rows, while GAAE's decoder at least tracks
the input direction weakly. This directly confirms the earlier hypothesis: the high
adjacency r (~0.34) was an artifact of adjacency being an easy target for a decoder
that mostly predicts "no edge," not evidence of better FC reconstruction. By the
metric that's actually apples-to-apples, GAAE's feature reconstruction is the
stronger (if still weak, `quality="poor"`) one.

Other notes from the re-run:

- Both train+val and test fidelity stay close within each model and metric (e.g. GAAE
  feature: 0.061 vs 0.046; VGAE-GCN feature: -0.088 vs -0.008) — no train/test
  collapse, so this isn't an overfitting artifact either.
- VGAE's train+val reconstruction-error AUC (converter vs. stable, computed from the
  adjacency MSE) is ~0.46–0.51 across both variants — chance level; reconstruction
  error doesn't separate classes for the anticollapse configs.
- Bottom line: if "which encoder reconstructs FC better" is the question, the answer
  from this run is GAAE, by the only metric that measures the same thing for both. For
  a more conclusive comparison, also check downstream signal (e.g. GEC/GEP classifier
  AUC built on each encoder's embeddings), since reconstruction fidelity and
  downstream discriminative power don't have to agree.
