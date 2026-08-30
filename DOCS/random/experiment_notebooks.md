# Experiment Notebooks

Two parameterized notebooks cover the full training pipeline for each network combination. They live in `CLASSIFIER/notebooks/`.

---

## Overview

Each network experiment follows a two-stage pipeline:

```
1. NETWORK_GAAE_RUNNER.ipynb  →  encoder checkpoint
2. NETWORK_GEC_RUNNER.ipynb   →  classification results (AUC, sensitivity, specificity, F1)
```

The GAAE stage trains an unsupervised graph autoencoder on all four cohorts (AD, MCI Stable, Healthy, Converter). Its encoder is then loaded as a pretrained feature extractor for the GEC classification stage.

---

## Experiment Table

Run the notebooks in this order to reproduce the full network framing comparison:

| # | Hypothesis | `network_combo` | `data_version` | `n_rois` | `file_suffix` |
|---|---|---|---|---|---|
| 1 | DMN baseline | `dmn` | `__v4__` | 46 | `_dmn_correlation_matrix_z_transformed.npz` |
| 2 | Hippocampus alone | `hippo` | `__v5__` | 4 | `_hippocampus_correlation_matrix_z_transformed.npz` |
| 3 | Limbic alone | `limbic` | `__v6__` | 12 | `_limbic_correlation_matrix_z_transformed.npz` |
| 4 | DAN alone (H3) | `dan` | `__v7__` | 26 | `_dorsal_attention_correlation_matrix_z_transformed.npz` |
| 5 | DMN + Hippo | `dmn_hippo` | `__v8__` | 50 | `_dmn_hippo_correlation_matrix_z_transformed.npz` |
| 6 | DMN + Limbic | `dmn_limbic` | `__v9__` | 58 | `_dmn_limbic_correlation_matrix_z_transformed.npz` |
| 7 | DMN + Hippo + Limbic | `dmn_limbic_hippo` | `__v10__` | 62 | `_dmn_limbic_hippo_correlation_matrix_z_transformed.npz` |
| 8 | All combined | `all_combined` | `__v11__` | 88 | `_all_combined_correlation_matrix_z_transformed.npz` |

---

## Stage 1 — NETWORK_GAAE_RUNNER.ipynb

**Purpose:** Pretrain a Graph Attention Autoencoder on all cohorts for the chosen network combination. The trained encoder is used as an initializer for the GEC classifier.

### How to run

1. Open `CLASSIFIER/notebooks/NETWORK_GAAE_RUNNER.ipynb`
2. Edit the `EXPERIMENT` config cell at the top:

```python
EXPERIMENT = {
    "network_combo": "dmn_hippo",    # label for WandB and checkpoint dir name
    "data_version": "__v8__",         # which DATA/DELCODE/__vN__/matrices to use
    "n_rois": 50,                     # number of ROIs (must match the data version)
    "file_suffix": "_dmn_hippo_correlation_matrix_z_transformed.npz",
    "knn_k": 8,                       # k for kNN adjacency graph
}
```

3. Run all cells. Training logs to WandB project `gaae-network-{network_combo}`.

### Outputs

Saved to `CLASSIFIER/notebooks/checkpoints_gaae_{network_combo}/{run_name}/`:
- `model_{run_name}.pth` — full GAAE model (encoder + decoder)
- `run_config.json` — model dimensions and training config (read by GEC runner to auto-configure)
- `loss_curves.png` — train/val reconstruction loss

---

## Stage 2 — NETWORK_GEC_RUNNER.ipynb

**Purpose:** Train the Cost-Weighted GEC classifier using 5-fold stratified cross-validation. Loads the pretrained GAAE encoder and fine-tunes a classification head.

### How to run

1. Open `CLASSIFIER/notebooks/NETWORK_GEC_RUNNER.ipynb`
2. Edit the `EXPERIMENT` config cell (use the **same values** as stage 1):

```python
EXPERIMENT = {
    "network_combo": "dmn_hippo",
    "data_version": "__v8__",
    "n_rois": 50,
    "file_suffix": "_dmn_hippo_correlation_matrix_z_transformed.npz",
    "knn_k": 8,
    "use_pretrained_encoder": True,   # set False to train end-to-end (no GAAE)
}
```

3. Run all cells. When prompted:
   - **GAAE checkpoint selection:** choose which GAAE run to load (most recent is pre-selected)
   - **Threshold method:** choose between Youden's J-statistic or best F1 for the final decision threshold

### Interactive prompts

| Prompt | Options | Default |
|---|---|---|
| GAAE checkpoint index | Integer index of available runs | Last (most recent) |
| Threshold method | `[1] Youden's J` or `[2] Best F1` | `[1] Youden's J` |

### Outputs

Saved to `CLASSIFIER/notebooks/checkpoints_gec_{network_combo}/{run_name}/`:
- `best_model_fold{N}.pth` — best fold model weights
- `run_summary.json` — CV metrics, thresholds, full config
- `cv_results.json` — per-fold AUC, sensitivity, specificity, F1

---

## Comparing Results

After running all 8 experiments, compare OOF AUC from each `run_summary.json`:

```python
import json, glob
results = []
for f in glob.glob('checkpoints_gec_*/*/run_summary.json'):
    with open(f) as fp:
        s = json.load(fp)
    results.append({
        'combo': s['experiment']['network_combo'],
        'n_rois': s['experiment']['n_rois'],
        'best_auc': s['best_val_auc'],
        'threshold_method': s['threshold_method'],
    })
results.sort(key=lambda x: x['best_auc'], reverse=True)
for r in results:
    print(f"{r['combo']:20s}  n_rois={r['n_rois']:3d}  AUC={r['best_auc']:.4f}")
```

Expected narrative:
- **Single networks** (exps 1–4) establish per-region baseline signal
- **DMN + memory extensions** (exps 5–7) should outperform DMN alone if H2 holds
- **All combined** (exp 8) vs exp 7 shows whether DAN adds signal or noise

---

## Edge Weighting (Optional Comparison)

To test weighted FC edges instead of binary kNN adjacency, change `adjacency_function` in the dataset loading cell:

```python
from common.utils import knn_weighted_adjacency_matrix_no_diag

# in the ClassificationDataset call:
adjacency_function=knn_weighted_adjacency_matrix_no_diag,
```

Weighted edges use `|correlation|` as edge weight instead of 0/1. GATv2Conv already accepts `edge_attr` so no model changes are needed.

---

## Tips

- **Data not found?** Make sure `DATA/DELCODE/{data_version}/matrices/` exists and contains `.npz` files. Run the processing scripts first (see [processing_scripts.md](processing_scripts.md)).
- **Mismatched dimensions?** The `run_config.json` from GAAE stage is read automatically to configure `IN_FEATURES`, `HIDDEN_DIM`, `LATENT_DIM`. If you change model dims between GAAE and GEC, they must match.
- **Skip pretraining?** Set `"use_pretrained_encoder": False` for an end-to-end GEC baseline without GAAE.
- **WandB offline?** Set `WANDB_MODE=offline` in the environment or call `wandb.init(mode="disabled")` manually in the config cell.
