# Disease Axis Projection in the GAAE Latent Space

**Notebook**: `CLASSIFIER/notebooks/GAAE_LATENT_SPACE_VISUALIZER.ipynb`  
**Model**: Graph Autoencoder + Encoder (GAAE), trained on whole-brain FC matrices

---

## Overview

The disease axis projection is a post-hoc interpretability technique that extracts a **single linear direction** in the GAAE's 64-dimensional latent space that best separates MCI subjects who later convert to Alzheimer's disease from those who remain stable. It answers the question:

> *"In what direction does a subject's brain representation need to move to look more like a converter?"*

---

## Step-by-Step Construction

### 1. Encode All Scans into the Latent Space

Every FC matrix is passed through the trained GAAE encoder (weights frozen) to produce a node-level embedding, which is then mean-pooled into a single graph-level vector:

```
z_i = mean_pool(GAAE_encoder(F_i))    z_i ∈ ℝ⁶⁴
```

This yields an embedding matrix `Z ∈ ℝ^{N×64}` over all training+validation scans, and a label vector `y ∈ {0,1}^N` (0 = stable MCI, 1 = converter).

The embeddings are **z-standardised** per dimension before any further analysis:

```
Z_std = (Z - μ) / σ       (column-wise)
```

---

### 2. Fit a Logistic Regression on the Latent Space

A L2-regularised logistic regression is trained directly on `Z_std`:

```
LR.fit(Z_std_train, y_train)
```

The fitted weight vector `w̃ ∈ ℝ⁶⁴` is the **normal to the decision hyperplane** — it points in the direction that maximally separates converters from stable MCI in latent space.

The unit-normalised version is the **disease axis**:

```
w̃_hat = w̃ / ‖w̃‖₂
```

---

### 3. Compute Per-Scan Disease Scores

Each scan embedding is projected onto the disease axis via a dot product:

```
s_i = z_i · w̃_hat        s_i ∈ ℝ
```

This scalar `s` is the **disease score**:

| Value | Interpretation |
|---|---|
| `s > 0` | Embedding on the **converter side** of the decision boundary |
| `s < 0` | Embedding on the **stable MCI side** |
| `s = 0` | Exactly on the **decision boundary** |

At the subject level, the mean `s` across all of a subject's scans is used as a summary disease score.

---

### 4. Residual Decomposition (for 3D Visualisation)

The disease score explains one dimension of the latent space variation. The remaining 63 dimensions contain information orthogonal to the conversion direction (e.g., age, sex, noise, unrelated connectivity patterns). To make this residual variation visible in 3D, PCA is applied:

```
R_i = z_i - s_i · w̃_hat       ← residual (subtract disease component)
[PC1, PC2] = PCA(R, n_components=2)
```

The final 3D coordinate of each scan is:

```
(x, y, z) = (s_i,  PC1_i,  PC2_i)
```

---

## The 3D Visualisation

The interactive Plotly 3D plot shows the entire training+validation scan population projected into this 3-axis space:

| Axis | Meaning |
|---|---|
| **x — Disease Score** | `s = z · w̃_hat` — pure MCI→AD conversion direction |
| **y — Residual PC1** | Largest source of within-class FC variation |
| **z — Residual PC2** | Second largest source of within-class variation |

Additional visual elements:
- **Coloured dots**: blue = stable MCI, red = converter
- **Lines**: chronological visit-to-visit trajectories per subject
- **Diamond markers**: class centroids in this latent-space decomposition
- **Grey plane at x=0**: the logistic regression decision boundary

A converter whose trajectory line moves **rightward (+x)** across visits is progressing toward AD in the latent representation. Stable MCI subjects that drift along y/z but remain left of the boundary are not converting.

---

## Disease Axis Steering Experiment

To validate that the disease axis encodes meaningful, decodable FC structure (rather than being a pure discriminative artefact), the notebook performs a **latent space steering experiment**:

1. Select a representative MCI subject as a **probe point** `z₀`.
2. Traverse the disease axis: `z(α) = z₀ + α · w̃_hat` for `α ∈ {−3σ, −2σ, −1σ, 0, +1σ, +2σ, +3σ}`.
3. Decode each steered embedding back to an FC matrix via the GAAE decoder.
4. Compute the **difference** `FC(α) − FC(0)` to highlight which connections change.

If the disease axis is clinically meaningful, the difference maps should show systematic changes in known AD-relevant networks (e.g., default-mode network hypo-connectivity, hippocampal network disruption) as `α` increases.

---

## Relationship to Other Components

```
GAAE encoder  ──→  Z (latent)  ──→  LogReg (w̃)  ──→  Disease score s
                        │                                     │
                        │  residual R = Z - s⊗w̃_hat          │
                        │                                     │
                        ▼                                     ▼
                  PCA(R) → y,z axes            GELSTM input / ABI comparison
```

- The **GELSTM** uses the full `z` embedding (not the projected `s`) as its sequential input — it learns its own temporal weighting of the latent dimensions.
- The **ABI Longitudinal** model operates entirely in FC space and does not use the disease axis.
- The disease axis provides a direct, human-interpretable **compression** of the LogReg classifier's decision function into a single number per scan, comparable to a biomarker score.

---

## Key Variables in the Notebook

| Variable | Shape | Description |
|---|---|---|
| `Z_std` | `(N_scans, 64)` | Standardised scan embeddings |
| `w_hat` | `(64,)` | Unit-normalised disease axis vector |
| `s` | `(N_scans,)` | Per-scan disease scores |
| `R_tv` | `(N_scans, 64)` | Residuals orthogonal to disease axis |
| `pca2_coords` | `(N_scans, 2)` | Top-2 residual PCs for 3D plot |
| `sorted_subjects` | list | Subjects sorted by mean disease score |
| `steered_fcs` | list of matrices | Decoded FC matrices along `w̃_hat` |
