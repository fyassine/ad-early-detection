# Noise Perturbation Methods for Robustness Evaluation

## Overview

Robustness of the graph-based classifier is evaluated by perturbing held-out test samples and measuring whether the model's reconstruction error — and hence its cohort prediction — remains stable under noise. The sweep uses **four distinct perturbation strategies**, each targeting a different aspect of the brain-connectivity graph representation. All methods are implemented in [`CLASSIFIER/common/robustness.py`](../CLASSIFIER/common/robustness.py) via `perturb_graph()`, with the `matrix_noise_rebuild` variant handled separately inside [`CLASSIFIER/common/reconstruction_eval.py`](../CLASSIFIER/common/reconstruction_eval.py).

---

## Experimental Setup

The robustness analysis is run on the **top-5 most confidently classified subjects per cohort** (Healthy, AD, MCI, Converter), selected from the test set based on a *selection margin* — how far their reconstruction error sits beyond the Youden-optimal one-vs-rest threshold $\tau_c$ derived from the validation set:

$$
\text{margin}_i =
\begin{cases}
e_i - \tau_c & \text{if direction is "high"} \\
\tau_c - e_i & \text{if direction is "low"}
\end{cases}
$$

where $e_i$ is the total reconstruction error of subject $i$ and $\tau_c$ is the per-cohort Youden threshold.

| Parameter | Value |
|---|---|
| Noise levels | 0 %, 5 %, 10 %, 20 %, 30 % |
| Trials per (subject, method, noise level) | 10 |
| Top-k subjects per cohort | 5 |
| Cohorts | healthy, ad, mci, converter |

Each trial applies a fresh random draw from the same `numpy.random.Generator`, ensuring reproducibility while still capturing variance across seeds within a fixed noise level.

---

## Method 1 — `feature_noise`

**What it perturbs:** Node feature matrix **X** (shape `[N_nodes × N_features]`).

**Mechanism:**

$$
\boldsymbol{\epsilon} \sim \mathcal{N}\bigl(\mathbf{0},\; (\sigma_{\mathbf{X}} \cdot \lambda)^2 \, \mathbf{I}\bigr)
$$

$$
\tilde{\mathbf{X}} = \mathbf{X} + \boldsymbol{\epsilon}
$$

where $\sigma_{\mathbf{X}} = \operatorname{std}(\mathbf{X})$ is the empirical standard deviation over all elements of $\mathbf{X}$, and $\lambda \in [0, 1]$ is the `noise_level`.

where `σ_X` is the standard deviation of all elements in **X** (falls back to 1.0 for degenerate single-element tensors). The perturbation is drawn from a zero-mean Gaussian scaled by the empirical feature spread, meaning a `noise_level` of 0.1 adds noise whose amplitude is 10 % of the typical feature variation.

**Graph structure:** Unchanged — the same `edge_index` is used.

**Purpose:** Measures sensitivity to small corruptions in brain region activity / connectivity features. A model that generalises well should maintain its reconstruction error ranking (and thus its cohort decision) under mild feature noise.

**Source:** [`robustness.py` L37–47](../CLASSIFIER/common/robustness.py)

---

## Method 2 — `matrix_noise_rebuild`

**What it perturbs:** Node features **and** the adjacency matrix, jointly.

**Mechanism:**
1. Apply `feature_noise` (same Gaussian as above) to **X** → **X̃**.
2. Recompute the **k-NN binary adjacency matrix** from **X̃** using `knn_binary_adjacency_matrix_no_diag(**adjacency_args)`.
3. Replace `edge_index` and `edge_attr` with the edges of the new graph.

This is the most structurally disruptive method because the graph topology changes as a consequence of the feature perturbation — nodes that were previously close in feature space may no longer be connected, and new edges may be introduced.

**Purpose:** Simulates a realistic scenario where noisy neuroimaging measurements would cause a different parcellation/connectivity graph to be constructed, testing whether the model is robust to such compounded input uncertainty.

**Source:** [`reconstruction_eval.py` L72–81](../CLASSIFIER/common/reconstruction_eval.py) (requires `adjacency_args` — the same `ADJACENCY_ARGS` dict used to build the original dataset graph).

---

## Method 3 — `edge_perturbation`

**What it perturbs:** Graph topology — edges are removed and replaced by random edges.

**Mechanism:**

$$
k_{\text{keep}} = \operatorname{round}\bigl(|\mathcal{E}| \cdot (1 - \lambda)\bigr), \qquad
k_{\text{add}}  = \operatorname{round}\bigl(|\mathcal{E}| \cdot \lambda\bigr)
$$

where $|\mathcal{E}|$ is the original number of directed edges and $\lambda$ is the `noise_level`. The perturbed edge set is:

$$
\mathcal{E}' = \operatorname{unique}\bigl(\mathcal{E}_{\text{kept}} \cup \mathcal{E}_{\text{random}}\bigr), \quad \mathcal{E}_{\text{random}} \subseteq \{(u,v) \mid u,v \in \mathcal{V},\; u \neq v\}
$$

1. A random subset of `keep` existing edges is retained.
2. `add` new random (src, dst) pairs are sampled uniformly over `[0, N_nodes)`.
   - Self-loops are discarded (`src == dst` filtered out).
3. The retained and new edges are concatenated and **deduplicated** via `torch.unique(..., dim=1)` to prevent duplicate `(src, dst)` columns from inflating the dense adjacency matrix above 1 (which would violate the BCE loss's `target ∈ [0,1]` assumption).
4. `edge_attr` is reset to all-ones for the surviving edges.

**Node features:** Unchanged.

**Purpose:** Simulates uncertainty in the brain connectivity graph itself — e.g., thresholding artefacts or parcellation boundary effects. Tests whether the model's decision is stable when the graph scaffold changes without any feature modification.

**Test coverage:** Two unit tests in [`tests/test_robustness.py`](../CLASSIFIER/tests/test_robustness.py) verify (a) no dense-adjacency value exceeds 1.0 and (b) no duplicate edge columns remain after perturbation.

**Source:** [`robustness.py` L49–84](../CLASSIFIER/common/robustness.py)

---

## Method 4 — `conditioning_noise`

**What it perturbs:** Subject-level conditioning variables (`patient_age`, `patient_sex`).

**Mechanism:**

$$
\tilde{a} = a + \delta_a, \quad \delta_a \sim \mathcal{N}(0,\;(0.05\,\lambda)^2)
$$

$$
\tilde{s} = \operatorname{clip}\bigl(s + \delta_s,\;0,\;1\bigr), \quad \delta_s \sim \mathcal{N}(0,\;(0.10\,\lambda)^2)
$$

where $a$ is the normalised patient age, $s \in [0,1]$ is the encoded patient sex, and $\lambda$ is the `noise_level`.

The scale factors (0.05 for age, 0.1 for sex) are deliberately small because these are normalised scalar conditioning signals, not high-dimensional features. If a field is absent from the data object the perturbation is silently skipped.

**Graph structure & features:** Unchanged.

**Purpose:** Tests robustness to demographic metadata uncertainty — relevant in clinical settings where age or sex may be incorrectly recorded. Measures whether the cohort decision is driven by the graph signal or by demographic conditioning.

**Source:** [`robustness.py` L86–99](../CLASSIFIER/common/robustness.py)

---

## Summary Table

| Method | Feature X | Edge topology | Conditioning | Typical use case |
|---|---|---|---|---|
| `feature_noise` | ✅ Gaussian | ❌ | ❌ | Measurement noise in brain features |
| `matrix_noise_rebuild` | ✅ Gaussian | ✅ Rebuilt from noisy X | ❌ | Compound input + graph uncertainty |
| `edge_perturbation` | ❌ | ✅ Drop & random add | ❌ | Connectivity graph structural noise |
| `conditioning_noise` | ❌ | ❌ | ✅ Age & sex | Demographic metadata uncertainty |

---

## Output Metrics

For each `(subject, method, noise_level, trial)` combination the sweep records:

- **`Total Error`** — reconstruction error of the model on the perturbed sample.
- **`CohortStable`** — binary flag: `1` if the error still crosses the cohort's one-vs-rest threshold in the correct direction (i.e., the original prediction is preserved), `0` otherwise.

These are aggregated across trials to produce:

| Metric | Description |
|---|---|
| `MeanTotalError` | Mean reconstruction error across 10 trials |
| `StdTotalError` | Standard deviation of reconstruction error |
| `CohortStabilityRate` | Fraction of trials where the cohort decision is preserved |

Results are visualised by `plot_robustness_sweep()` as two panels per cohort: **error drift** (mean error vs. noise level) and **decision stability** (stability rate vs. noise level), with the cohort threshold drawn as a horizontal reference line.

---

## File Locations

| File | Role |
|---|---|
| [`CLASSIFIER/common/robustness.py`](../CLASSIFIER/common/robustness.py) | Core `perturb_graph()` implementation |
| [`CLASSIFIER/common/reconstruction_eval.py`](../CLASSIFIER/common/reconstruction_eval.py) | Model-agnostic `compute_errors_for_dataset()` with `matrix_noise_rebuild` logic |
| [`CLASSIFIER/model/GAAE/evaluation.py`](../CLASSIFIER/model/GAAE/evaluation.py) | GAAE-specific variant + threshold/stability utilities |
| [`CLASSIFIER/tests/test_robustness.py`](../CLASSIFIER/tests/test_robustness.py) | Unit tests for edge-perturbation correctness |
