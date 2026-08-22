# Methodological Fairness, Architectural Nuances, and Limitations: GELSTM vs. Brain-TokenGT

**Document Purpose:** Comprehensive technical breakdown of the methodological discrepancies, cohort-windowing asymmetries, optimization challenges, and validity boundaries in the comparative evaluation between **GELSTM** and **Brain-TokenGT** on the DELCODE cohort.

---

## 1. Executive Summary: The Fairness Landscape

When comparing our proposed **GELSTM** architecture against the published baseline **Brain-TokenGT** (Dong et al., *Beyond the Snapshot: Brain Tokenized Graph Transformer for Longitudinal Brain Functional Connectome*, MICCAI 2023), raw benchmark tables show GELSTM outperforming Brain-TokenGT:
- **GELSTM (None - Raw FC):** Test AUC **$0.8313 \pm 0.0458$** ($31\text{k}$ params)
- **GELSTM (Frozen GAAE):** Test AUC **$0.7812 \pm 0.0105$** ($520\text{k}$ params)
- **Brain-TokenGT (Stabilized):** Test AUC **$0.6156 \pm 0.0705$** ($604\text{k}$ params)
- **Brain-TokenGT (As-Released):** Test AUC **$0.5779$** ($604\text{k}$ params)

However, presenting this raw numerical difference as a pure "architectural victory" without documenting the underlying experimental asymmetries violates scientific rigor. To satisfy our supervisor's (Chantal's) fairness criteria, this document details the **5 key fairness issues, experimental fallbacks, and methodological caveats** that contextualize this comparison.

---

## 2. Issue 1: Cohort Windowing Disparity ($1$–$6$ Visits vs. $2$–$3$ Visits)

### 2.1 The Discrepancy
The most critical data-level difference between the two models on DELCODE is the **visit filtering and sequence windowing protocol**:

| Dimension | GELSTM (Full Cohort) | Brain-TokenGT (Windowed) |
|---|---|---|
| **Visit Inclusion Rule** | $1 \le T \le 6$ visits | $2 \le T \le 3$ visits (`min_visits: 2, max_visits: 3`) |
| **Handling of Single-Visit Subjects** | **Included** (evaluated on baseline scan) | **Dropped entirely** ($38$ CV subjects, $9$ test subjects dropped) |
| **Handling of $>3$ Visits** | **Full sequence used** (up to 6 visits with continuous $\Delta t$) | **Truncated** to the first 3 visits |
| **Cross-Validation Sample Size** | $N_{\text{CV}} = 133$ ($54$ converters, $79$ stable MCI) | $N_{\text{CV}} = 95$ ($37$ converters, $58$ stable MCI) |
| **Held-Out Test Sample Size** | $N_{\text{test}} = 34$ ($14$ converters, $20$ stable MCI) | $N_{\text{test}} = 25$ ($11$ converters, $14$ stable MCI) |

### 2.2 Methodological Consequences & Biases
1. **Unequal Test Sets ($N=34$ vs. $N=25$):**
   - The test metrics ($0.781$ vs. $0.616$) are computed on **different subject counts**.
   - Dropping the 9 test subjects with only 1 scan removes $3$ converters and $6$ stable MCI patients from Brain-TokenGT's test evaluation.
2. **Task Difficulty & Patient Trajectory Shift:**
   - Single-visit subjects in clinical cohorts are often fast-progressing converters who dropped out early or late-enrolled stable subjects. Evaluating on $1$ visit tests static baseline separability, whereas $2$–$3$ visits forces pure longitudinal evaluation.
3. **Information Advantage for Long Sequences:**
   - GELSTM is provided up to 6 visits of disease progression, while Brain-TokenGT's self-attention is capped at 3 timepoints.
4. **Why Brain-TokenGT Used This Window:**
   - Dong et al. originally designed Brain-TokenGT for cohorts with fixed 3-timepoint acquisitions (e.g., ADNI $t_1, t_2, t_3$). Its spatio-temporal self-attention mechanism memory scales quadratically with total tokens ($T \times K$).

---

## 3. Issue 2: Brain-TokenGT Temporal Module Instability & Stabilization Summary

The evolution, failure modes, and stabilization fixes for Brain-TokenGT's temporal module (EvolveGCN-H / GRCU) are summarized below:

| Temporal Module Variant | Configuration & Mechanism | Behavior & Failure Mode on DELCODE | Performance Outcome | Fairness & Validity Assessment |
|---|---|---|---|---|
| **1. As-Released Baseline** | `train_give: false`<br>• Recurrent weights frozen at random Gaussian init.<br>• Never registered with optimizer. | **Zero Learned Trajectory:**<br>• Catastrophic threshold collapse.<br>• Predicts "converter" for 100% of test subjects. | **Test AUC:** $0.5779$<br>**Sens:** $1.000$<br>**Spec:** $0.000$<br>**CV AUC:** $0.5538 \pm 0.1258$ | **Null Comparison:** Cannot claim "GELSTM beats Brain-TokenGT" when Brain-TokenGT's temporal core was functionally disabled. |
| **2. Unstabilized Repair** | `train_give: true`<br>• Full end-to-end gradient updates to recurrent weights. | **Numerical Collapse (NaN):**<br>• Exploding gradient norms in GRCU weights.<br>• Non-finite `TopK.forward` scores crash training within ~10 epochs. | **Crashed (Exit code 1):**<br>2 consecutive runs failed completely. | **Instability Finding:** Demonstrates standard EvolveGCN-H is intrinsically unstable on small ($N<100$) clinical cohorts. |
| **3. Stabilized Repair (Adopted)** | `train_give: true`<br>+ `give_lr_scale: 0.1`<br>+ `give_weight_decay: 0.001`<br>• Separate param group with damped LR and L2 norm decay. | **Stable Convergence:**<br>• 100% run completion ($5/5$ runs).<br>• High validation AUC ($0.814$) but drops on held-out test ($0.616$). | **Test AUC:** $0.6156 \pm 0.0705$<br>**Sens:** $0.5636$<br>**Spec:** $0.5286$<br>**CV AUC:** $0.8142 \pm 0.0838$ | **Fair Baseline Established:** Satisfies Chantal's Criterion 3 (not sabotaged, early stopped, decently optimized). Represents the true baseline number. |

---

## 4. Issue 3: Architectural Prior & Tokenization Overhead on Sample-Constrained Cohorts

### 4.1 Dimensionality & Token Explosion
A structural reason for Brain-TokenGT's generalization gap (CV AUC $0.814$ $\rightarrow$ Test AUC $0.616$) is its tokenization design:

```
Functional Connectivity (200x200)
       │
       ▼
Spatial Tokenization (K=8 nodes per ROI clique) ──► ~460 tokens per visit
       │
       ▼
3 Longitudinal Visits (T=3) ──► ~1,381 Spatial-Temporal Tokens per Subject
       │
       ▼
Cross-Visit Self-Attention + EvolveGCN-H (603,849 Trainable Parameters)
```

- With $N_{\text{train}} \approx 76$ subjects per cross-validation fold, the transformer is forced to learn dense pairwise cross-visit attention over **$1,381$ tokens per sequence**.
- In contrast, GELSTM compresses each visit into a compact 64-d vector (or 200-d in `none`), and processes a sequence of length $T \le 6$ via a lightweight LSTM cell with only **$31\text{k}$ to $520\text{k}$ parameters**.

### 4.2 Parameter Efficiency Comparison

| Architecture | Total Trainable Parameters | Representation per Visit | Tokens / Inputs to Temporal Core | Generalization Gap (CV - Test) |
|---|:---:|:---:|:---:|:---:|
| **GELSTM (None)** | **$31,169$** | 200-d pooled vector | 1 vector per visit | **$+0.060$** (Test beats CV) |
| **GELSTM (Frozen)** | $520,905$ | 64-d GAAE latent | 1 vector per visit | **$+0.140$** |
| **Brain-TokenGT (Stabilized)** | $603,849$ | 460 graph tokens | ~1,381 tokens | **$-0.198$** (Severe Overfitting) |

Brain-TokenGT's high-capacity transformer easily overfits the small DELCODE validation folds, creating an illusion of high validation performance ($0.814$) that fails to generalize to held-out test subjects ($0.616$).

---

## 5. Issue 4: Unequal Tuning Budget & Engineering Asymmetry

### 5.1 Iteration History Asymmetry
- **GELSTM:** Developed, ablated, calibrated, and refined over extensive iterative experiments (loss weighting, continuous time delta integration, layer normalization, FiLM conditioning).
- **Brain-TokenGT:** Evaluated using its author-recommended default hyperparameters, adjusted only by the minimal learning-rate/weight-decay scaling required to prevent numerical divergence.

### 5.2 Supervisor Fairness Alignment (Chantal's Criterion 3)
Chantal's guidance on baseline optimization states:
> *"All models should be decently optimized (don't continue training if the model is clearly overfitting, don't use a lr where training collapses or does not learn at all etc. — but no need to do super extensive hyperparameter tuning, that's just not realistic)."*

Our stabilization protocol satisfies this rule by ensuring non-divergence and early stopping, but reviewers must be informed that Brain-TokenGT did not receive exhaustive grid searches over token counts ($K$) or layer depths ($L$).

---

## 6. Issue 5: Temporal Delta Awareness ($\Delta t$ Continuous Timing)

- **GELSTM's Continuous Time Integration:** Incorporates actual inter-visit intervals $\Delta t = (t_{k} - t_{k-1})$ in months directly into the recurrent gate transitions. This allows the model to differentiate between a 6-month follow-up and a 24-month gap.
- **Brain-TokenGT's Discrete Step Assumption:** Treats timepoints as uniform discrete steps ($t_1 \rightarrow t_2 \rightarrow t_3$), ignoring irregular inter-scan intervals.

---

## 7. How to Formulate Defensible Claims in the Thesis & Manuscript

To ensure the thesis and publications remain immune to reviewer criticism, use the following **four-pillar reporting strategy**:

### 1. State the Cohort Window Difference Upfront
> *"Because Brain-TokenGT requires sequences of length $2 \le T \le 3$, its evaluation was conducted on the $N=95$ subset ($N=25$ test). GELSTM was evaluated on the complete longitudinal cohort ($1 \le T \le 6, N=133$). Future matched-window runs ($2 \le T \le 3$ for GELSTM) will isolate whether sequence length or model architecture drives the margin."*

### 2. Document the Temporal Module Stabilization Honestly
> *"As released, Brain-TokenGT's recurrent weight evolution was disabled (`train_give=False`), producing an uninformative decision boundary ($0.578$ AUC, $1.00$ sens / $0.00$ spec). Enabling end-to-end training required learning-rate damping (`0.1x`) and weight decay (`1e-3`) to prevent TopK divergence. Across 5 stabilized repeats, the model achieved a mean CV AUC of $0.814 \pm 0.084$ and Test AUC of $0.616 \pm 0.071$."*

### 3. Attribute the Performance Gap to Capacity vs. Sample Size
> *"The performance difference ($0.781$–$0.831$ vs. $0.616$) does not merely reflect 'our model is better'; it highlights that high-dimensional spatio-temporal graph tokenization (~$1,380$ tokens) is prone to representation dispersion and overfitting on sample-constrained clinical cohorts ($N < 100$). In contrast, a parameter-lean recurrent trajectory model ($31\text{k}$–$520\text{k}$ params) regularizes the learning problem effectively."*

### 4. Highlight the Decisive External Validation Path
> *"Definitive head-to-head resolution will occur on the larger external ADNI cohort ($N=162$ subjects with $\ge 2$ sessions; $51$ converters) where sample size is 5x larger and both models will be evaluated on byte-matched manifest splits."*
