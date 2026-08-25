<a id="top"></a>

# Temporal-First Graph Network (TFGN): Methods and Results

**Scope.** This document is the self-contained methods + results write-up for the
temporal-first ablation ladder (pooled ADNI+DELCODE training, external OASIS-3 test).
It is a reading version of the pre-registration in `DOCS/temporal-first-ablation.md` and
the execution record in `DOCS/flipped/PLAN.md`; where the two disagree with anything
here, those documents are authoritative.

Status: **ladder complete**. All 76 ladder runs plus the pre-registered escalation arm
have reported; the single Tier-4 held-out read has been spent; no further GPU work is
planned under this plan.

One-page version: [`METHODS_SUMMARY.md`](METHODS_SUMMARY.md).

## Table of Contents

- [1. Methods](#1-methods)
  - [1.1 Motivation and hypothesis](#11-motivation-and-hypothesis)
  - [1.2 Data, preprocessing and labels](#12-data-preprocessing-and-labels)
  - [1.3 Self-supervised pretraining (two runs, both on pooled ADNI+DELCODE only)](#13-self-supervised-pretraining-two-runs-both-on-pooled-adnidelcode-only)
  - [1.4 The TFGN architecture](#14-the-tfgn-architecture)
  - [1.5 Pre-registered design decisions (fixed before any run)](#15-pre-registered-design-decisions-fixed-before-any-run)
  - [1.6 Training and optimisation](#16-training-and-optimisation)
  - [1.7 The ablation ladder — arms](#17-the-ablation-ladder--arms)
  - [1.8 Evaluation protocol](#18-evaluation-protocol)
  - [1.9 Interpretability validation (pre-registered, independent of AUC)](#19-interpretability-validation-pre-registered-independent-of-auc)
- [2. Results](#2-results)
  - [2.1 Scorecard](#21-scorecard)
  - [2.2 Stopping-rule verdicts (every rung against S1)](#22-stopping-rule-verdicts-every-rung-against-s1)
  - [2.3 The winner, and why the win is not capacity](#23-the-winner-and-why-the-win-is-not-capacity)
  - [2.4 S3 is void, not rejected](#24-s3-is-void-not-rejected)
  - [2.5 Sequence length — the SENS decomposition](#25-sequence-length--the-sens-decomposition)
  - [2.6 Matched-window head-to-head — a crossover, not a defeat](#26-matched-window-head-to-head--a-crossover-not-a-defeat)
  - [2.7 Tier-4 held-out reads (one pass, spent once)](#27-tier-4-held-out-reads-one-pass-spent-once)
  - [2.8 The cohort shortcut, and a mitigation that was tried and failed](#28-the-cohort-shortcut-and-a-mitigation-that-was-tried-and-failed)
  - [2.9 Interpretability validation — reproducible, but not DMN-specific](#29-interpretability-validation--reproducible-but-not-dmn-specific)
  - [2.10 Scaling gate: closed](#210-scaling-gate-closed)
- [3. Summary and limitations](#3-summary-and-limitations)
- [4. Reproducibility](#4-reproducibility)

---

<a id="1-methods"></a>

## 1. Methods

<a id="11-motivation-and-hypothesis"></a>

### 1.1 Motivation and hypothesis

Every longitudinal classifier previously built in this project (GELSTM, GEGRU, GEC, GEP)
is **spatial-then-temporal**: each visit's whole-brain functional-connectivity (FC) graph
is pooled to a single 64-d vector by a graph encoder *before* any temporal model sees it
(`GELSTMClassifier.encode_visit`, `CLASSIFIER/model/GELSTM/models.py:191`). Region
identity is destroyed by mean-pooling before the trajectory is modelled at all.

TFGN inverts that order — **temporal-then-spatial**. A node-shared LSTM first encodes each
of the 200 regions' own FC-row trajectory; a learned saliency gate then suppresses regions
that never change; a variational graph encoder (GVAE) propagates the surviving per-node
dynamics over the baseline connectome; a residual skip protects the unsmoothed temporal
features; and a dual-score readout produces a per-region diagnostic map alongside the
classification logit.

**Hypothesis.** If region-resolved temporal dynamics carry conversion-relevant signal that
early pooling destroys, deferring pooling until after per-node temporal encoding should
improve prediction at matched or lower capacity.

**Honest expectation, fixed in advance.** The in-domain test set is n=64 and the external
set is n=60. AUC differences below ≈0.08 are not resolvable at those sizes (≈0.04 on the
248-subject cross-validation pool). The ladder was designed to *rule changes out* cheaply,
not to guarantee a win.

<a id="12-data-preprocessing-and-labels"></a>

### 1.2 Data, preprocessing and labels

All three cohorts pass through one identical pipeline: BIDS conversion → fMRIPrep (motion
correction, susceptibility-distortion correction, coregistration, normalisation) →
denoising (ICA-AROMA + 2 physiological + 1 global-signal regressor, then bandpass) →
motion-QC gate (mean framewise displacement > 0.5 mm excludes the session) → Schaefer-200
parcel time-series extraction → pairwise Pearson correlation + Fisher *z*-transform, giving
one 200×200 matrix per session → linear regression of age effects.

**Label.** A subject is a **converter** if any visit carries a post-conversion diagnosis,
and **stable MCI** otherwise. Visits at which a converter is already demented are excluded
as model inputs — the task is *predicting* conversion, not recognising established
dementia.

**Pooled protocol** (new for this work; no pooled-cohort protocol existed in the repo
before). Built by `DATA/manifest/build_pooled_assets.py`, which harmonises DELCODE's and
ADNI's downstream splits into one schema, constructs ADNI's missing pretrain split under
DELCODE's leakage rule, and materialises a symlink farm over both cohorts' FC `.npz`
files so a single glob root serves the pooled dataset.

| split | composition | n |
|---|---|---|
| CV / training pool | ADNI + DELCODE downstream train+val, `min_visits=2` | **248** (ADNI 153 + DELCODE 95) |
| In-domain test | ADNI + DELCODE downstream test, `min_visits=2` | **64** (ADNI 39 + DELCODE 25; 24 converters / 40 stable) |
| External test | all OASIS-3 downstream subjects (train+val+test concatenated) | **60** (31 converters / 29 stable) |
| Sensitivity pool | same as CV pool with `min_visits=3` | 140 |

OASIS-3 is excluded from *everything* upstream — including autoencoder pretraining — so it
remains a genuinely external cohort. Seeds 42, 43, 44, 45 for every arm.

Cohort-specific visit semantics are preserved: DELCODE encodes protocol months, ADNI and
OASIS-3 encode actual elapsed days. `CLASSIFIER/common/pooled_data.py` builds one
`LongitudinalSubjectDataset` per cohort with that cohort's root and allow-list column and
concatenates the items, dropping the non-native allow-list column and raising if the
retained one is entirely null — otherwise the dataset's first-match column pick would have
silently disabled DELCODE's post-conversion leakage filter.

<a id="13-self-supervised-pretraining-two-runs-both-on-pooled-adnidelcode-only"></a>

### 1.3 Self-supervised pretraining (two runs, both on pooled ADNI+DELCODE only)

- **P1 — pooled GAAE** (`gaae-pretrain-pooled-adni-delcode`). Graph autoencoder on ~3 700
  unlabelled pooled scans; architecture identical to the existing DELCODE-only checkpoint
  (hidden 128, latent 64, 2 GAT heads, kNN k=16, adjacency-loss weight 0.2, 500 epochs,
  final validation loss 0.018795) so every downstream arm remains comparable. Supplies the
  frozen encoder for the spatial-first comparators.
- **P2 — node-LSTM SSL** (`tfgn-nodelstm-ssl-pooled`). Self-supervised next-visit FC-row
  forecasting: the node-shared LSTM and input projection are trained to predict
  `x_i^(t+1)` from history over every ≥2-visit subject in the pooled *pretrain train*
  split (MSE, hidden 64, 1 layer, cosine schedule). A persistence baseline
  (`x(t+1)=x(t)`, MSE 0.0630) confirms the checkpoint learned something real: the
  untrained network started worse (0.0658) and training pulled it 27.2 % below persistence
  (0.0459).

<a id="14-the-tfgn-architecture"></a>

### 1.4 The TFGN architecture

Implemented in `CLASSIFIER/model/TFGN/` (pure logic; no I/O, no path construction), with
every stage switchable from `TFGNTrainConfig` so the entire ladder is one model with knobs.
Per-subject inputs are the visit stack `X ∈ R^{T×200×200}`, cumulative `log Δt`
(standardised with train-fold statistics), the baseline adjacency `A_0` (kNN-sparsified,
k=8, on `|FC|`), covariates `[age, sex]`, and — where an arm needs them — the change-mask
`M`, strength centrality of `|A_0|`, and the drift anchor `d̃`.

1. **Node-shared LSTM.** One LSTM (hidden 64, 1 layer, dropout 0.3) applied to each of the
   200 nodes' own FC-row sequence with shared weights, optionally concatenating `log Δt`.
   Output `h_i^{(T)} ∈ R^{64}` per node.
2. **Temporal saliency gate.** `s_i = σ(wᵀ LeakyReLU(W_s h_i))`, applied as residual
   scaling `(1+s_i) h_i`. Regularised by a sparsity KL, `KL(s̄ ‖ ρ)` with ρ=0.15, plus a
   drift-anchor MSE (§1.5).
3. **GVAE encoder.** GATv2 μ/logσ² heads (hidden 128, latent 64, 2 heads) over `A_0` with
   FiLM conditioning of μ on `[age, sex]`; reparameterised sampling in training. Built
   **only** when `recon_target ≠ none`.
4. **Fusion.** `concat_residual` = `LayerNorm(W_u[h_i ‖ z_i])`, or `z_only`.
5. **Readout.** Mean pooling over nodes, or attentive pooling with exact sparsemax
   (Martins & Astudillo, 2016), which produces true zeros rather than a soft distribution.
6. **Dual-score head.** Classification logit plus a per-node topological saliency score
   `s_topo`, anchored to strength centrality by an MSE term (`lambda_cent`).
7. **Cohort-adversary head** (escalation only). Gradient-reversal layer (identity forward,
   negated-and-scaled gradient backward) into a binary ADNI-vs-DELCODE MLP attached to the
   pooled patient embedding.

<a id="15-pre-registered-design-decisions-fixed-before-any-run"></a>

### 1.5 Pre-registered design decisions (fixed before any run)

**Reconstruction target.** `σ(ZZᵀ) ∈ (0,1)` cannot be trained with BCE against
`ΔA ∈ [−2,2]`. The pre-registered default is `delta_a_topk`: a per-subject binary
change-mask `M_ij = 1[|ΔA_ij| ≥ q_{1−κ}]`, κ=0.10, with `BCEWithLogits(pos_weight=(1−κ)/κ)`.
Construction raises if any subject's positive-edge fraction falls outside [0.01, 0.5]
rather than silently training on a degenerate mask. `delta_a_mse` and `a_last` are the
documented alternatives and are guarded against the Fisher-*z* file variant, whose range
invalidates both.

**Topological anchor.** Eigenvector centrality is ill-defined on signed FC (negative
weights break Perron–Frobenius), so the anchor is **strength centrality on `|A_0|`**,
z-scored with train-fold statistics.

**Drift anchor.** A raw drift magnitude `‖x_i^{(T)} − x_i^{(1)}‖₂` is dense and fights the
sparsity KL. The anchor is instead the within-subject drift *rank* through a sharp sigmoid
centred at the (1−ρ) quantile: `q_i = rank(d_i)/(N−1)`, `d̃_i = σ((q_i − (1−ρ))/τ_d)`,
τ_d = 0.05. By construction `mean(d̃) ≈ ρ`, so the two regularisers agree; ranks also make
the anchor scale-free across cohorts with different FC amplitude. `λ_drift = 0.1·λ_sparse`
so the sparsity prior dominates at the margin.

**Determinism.** GATv2's scatter-based aggregation and sparsemax have nondeterministic
GPU backward kernels; without forcing determinism the seed-level standard error would be
measuring GPU scheduling noise. Every TFGN run sets `strict_determinism: true`
(`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`), verified
by running one arm twice and diffing its metric block.

**Cohort-shift control.** ADNI and DELCODE differ in scanner, protocol and follow-up
rhythm (DELCODE ≈90 % regular 12-month visits; ADNI median gap 371 days, IQR 207–419).
Cohort is deliberately **not** given to the model — a FiLM cohort covariate would make
OASIS-3 an untrained one-hot category and its behaviour unspecified. Instead every pooled
run computes a mandatory **cohort probe**: a logistic regression decoding cohort from the
out-of-fold patient latents, recorded as `cohort_probe_auc`. Pre-registered escalation: if
that probe exceeds **0.75** on the winning arm, re-run it with adversarial (gradient-
reversal) cohort conditioning and report both.

<a id="16-training-and-optimisation"></a>

### 1.6 Training and optimisation

Shared across TFGN arms (`configs/tfgn_pooled.json`): Adam, lr 1e-3, weight decay 0,
batch size 16, gradient clipping 1.0, up to 100 epochs with early stopping (patience 20)
and ReduceLROnPlateau (factor 0.5, patience 5, floor 1e-6); class-cost weighting on the
BCE. Five-fold `StratifiedGroupKFold` cross-validation over the 248 pooled subjects;
per-fold `StandardScaler` on the temporal embeddings, and the `log Δt` and centrality
statistics fitted on the training fold only and carried inside the saved model state.
Decision thresholds are selected on validation/out-of-fold predictions (best-F1) and never
from test metrics.

<a id="17-the-ablation-ladder--arms"></a>

### 1.7 The ablation ladder — arms

Each rung inherits the *kept* knobs of the rungs before it; a rung that fails the stopping
rule has its knob dropped and the next rung branches from the last surviving config.

**S0a — logistic regression on drift** (`logregdrift`). Features `[PCA₃₂(vec(ΔA)),
n_visits, total follow-up months, age, sex]`. The linear floor: is there signal in raw FC
change at all? Implemented as an adapter, not a notebook, so it rides the identical CV,
threshold, external-test and ledger path as every deep arm. A demographics-only variant
(`feature_set: demo`, `[age, sex]`) provides the Tier-1 floor every arm must beat.

**S0b — GELSTM, pretrained frozen encoder.** The current spatial-first model with the
pooled P1 GAAE frozen (LSTM hidden 32, 1 layer, `use_time_delta`, mean graph pooling).
Spatial-first **with** self-supervision — the matched comparator for S1c.

**S0c — GELSTM, random encoder.** Identical, `encoder_init: random`, trained end to end.
Spatial-first **without** self-supervision — the matched floor for S1.

**S0d — BrainTokenGT.** The SOTA competitor (Dong et al., MICCAI 2023), stabilised config.
Reported as a reference point with a caveat, not as a clean baseline: its scatter/gather
ops never engage deterministic algorithms and its same-seed test AUC has been observed to
span 0.357–0.708. It is architecturally capped at `min_visits=2, max_visits=3`.

*Why S0b and S0c both exist:* S0b's encoder has 3 700 unlabelled graphs of reconstruction
pretraining behind it while a naive TFGN GVAE learns only from the classification gradient
on 248 labelled subjects. The two clean contrasts are therefore **S0c ↔ S1** (neither
encoder pretrained) and **S0b ↔ S1c** (both pretrained).

**S1 — the flip alone.** `node_lstm_init: random`, gate off, `recon_target: none`,
`fusion: z_only`, `readout: mean`. No self-supervision anywhere on either side; compared
against S0c this isolates the architectural flip itself.

**S1b — node-LSTM self-supervision.** S1 with `node_lstm_init: pretrained_finetuned` from
the P2 checkpoint. Asks whether node-level forecasting pretraining helps.

**S1c (original run) — invalid, recorded as undecidable.** The registered S1c question is
"both encoders self-supervised, compared to S0b". The first run
(`tfgn-s1c-recon-pooled`) was launched with `node_lstm_init: pretrained_finetuned`,
inheriting the S1b fork decision as it stood *before* that decision was reversed (§2.2).
It therefore tests a configuration — SSL node-LSTM init **and** reconstruction loss on top
of a since-dropped knob — that was never a registered rung. Its artifacts are kept
untouched as the record of the invalid configuration; the id was never repointed.

**S1c (corrected re-run) — `tfgn-s1c-recon-random-pooled`.** Byte-identical to the
original entries except `node_lstm_init: random` (S1's carried-forward config), with
`recon_target: delta_a_topk` and its λ/β/free-bits/warmup schedule unchanged. This is the
protocol-valid arm that answers S0b ↔ S1c.

**S2 — temporal saliency gate.** `use_gate: true`, `lambda_sparse: 0.1`,
`lambda_drift: 0.01`, `gate_rho: 0.15`. Branches from S1. Does suppressing static regions
help, and does the gate produce a valid region map?

**S3 — concat-residual fusion.** `fusion: concat_residual`, branching from S1. Does
preserving the unsmoothed per-node features alongside the graph latent help? *(See §2.4 —
this knob has no gradient path under S1's `recon_target: none`, and the rung is void
rather than rejected.)*

**S4 — attentive pooling.** `readout: attention` (sparsemax). Does learned node weighting
beat mean pooling?

**S5 — dual-score readout / full TFGN.** `dual_score: true`, `lambda_cent: 0.1`. The
interpretability layer: a per-node topological saliency map anchored to strength
centrality, alongside the classification logit. **Pre-registered as kept regardless of
AUC** — the keep/drop rule does not apply to it, because a map that fails to improve
classification can still be a valid (or invalid) interpretability artifact, and that
question must not be quietly dropped if the AUC story turns out negative.

**SENS — sequence-length sensitivity.** Winning config at `min_visits=3` (140 subjects).

**W3 — matched-window comparison** (additive, post-ladder). BrainTokenGT is capped at
T ∈ [2,3] while GELSTM and TFGN consume full trajectories, so a strict head-to-head needs
identical inputs: `tfgn-w3-gelstm-{frozen,random}-pooled` and `tfgn-w3-winner-pooled`, all
at `max_visits: 3`. BrainTokenGT is not re-run — S0d already runs at exactly this window
by construction, and the filter-then-truncate order was verified to leave the subject pool
unchanged (only visits are dropped, never subjects).

<a id="18-evaluation-protocol"></a>

### 1.8 Evaluation protocol

A four-tier protocol, fixed before any ladder decision was read.

**Tier 1 — floor gates.** Every arm must beat the demographics-only floor (`[age, sex]`)
and the static N=1 baseline (each fold's own validation subjects truncated to a single
visit and scored at that fold's threshold, via a `fold_probe` hook in
`common/crossval.py`).

**Tier 2 — selection (the stopping rule).** The independent unit is the **seed** (n=4),
not the fold. For each seed, compute the mean paired per-fold ΔAUC (rung *k* minus its
parent, matched fold for fold); report the mean and standard error of those four
seed-level means. **A rung is kept only if `mean(Δ) > SE(Δ)`.** At n=4 no p-value is
claimed and the SE is itself high-variance — this is a heuristic screen, not a test, and
its two outcomes are asymmetric by construction: a failure means *"undetectable at this
sample size"*, never *"harmful"*; a pass means *"worth carrying forward"*, never
*"significant"*. Where two legitimate readings disagree, a pre-registered one-standard-
error tie-breaker prefers the simplest configuration within one SE of the best.

The rule reads **pooled cross-validation out-of-fold (OOF) AUC**, not test AUC. This was
changed by a documented addendum on 2026-08-24, before any ladder decision was read: OOF
gives n=248 instead of n=64 (shrinking the unresolvable-difference floor from ≈0.08 to
≈0.04) and removes selection-on-test entirely. Every arm persists per-subject OOF
predictions (`oof_predictions.csv`: subject, fold, cohort, label, probability, n_scans,
age, sex) so every reported statistic is recomputable per subject and per fold rather than
read from a run's own summary line.

**Tier 3 — robustness vetoes.** Per-cohort OOF AUC, balanced accuracy, scan-count
correlation, and the cohort probe, with thresholds fixed in advance.

**Tier 4 — estimation.** The 64-subject in-domain test set and the 60-subject OASIS-3 set
are read **exactly once**, on the frozen winner and its designated secondaries, at the
OOF-derived threshold, with no retraining: `common/frozen_read.py` reconstructs each run's
adapter from its own saved `run_summary.json` and checkpoint and scores the split.
Every ladder arm ran with `defer_test_eval: true`; this was verified across all 76 runs
before the read (zero `test_*` and zero `ext_*` keys anywhere). `score_frozen_split` now
refuses to score a split whose result keys already exist unless overwrite is passed
explicitly, so the one-shot read cannot be spent twice by re-executing a cell.

Order of claims, stated in advance: **select on OOF → report on the in-domain test →
generalise on OASIS-3.** OOF-based selection reuses the same 248 subjects across every
ladder decision, so the winner's own OOF AUC is mildly optimistic and is not an unbiased
performance estimate.

**Scope — what was tested where.** Every arm below was trained and cross-validated on the
pooled 248-subject ADNI+DELCODE pool (that is where every stopping-rule verdict in §2 comes
from). The in-domain held-out test (n=64) and OASIS-3 (n=60) are one-shot Tier-4 resources
and were **not** spent on most arms — spending them on every rung would turn external
validation into a selection set, exactly the winner's-curse failure Tier 4 exists to avoid.

| arm | ADNI+DELCODE OOF (CV, n=248) | in-domain test (n=64) | OASIS-3 (n=60) |
|---|---|---|---|
| S0-demo, S0a, S0b, S0c, S0d | yes | no | no |
| **S1 flip (primary)** | yes | **yes** | **yes** |
| **S1b SSL (secondary)** | yes | **yes** | **yes** |
| S1c (original, invalid) / S1c-random | yes | no | no |
| S2 gate, S3 fusion (void), S4 attn-pool | yes | no | no |
| **S5 dual-score (secondary, interpretability)** | yes | **yes** | **yes** |
| SENS (`min_visits=3`) | yes | no | no |
| W3-GELSTM-random, W3-TFGN-winner | yes | no | no |
| W3-GELSTM-frozen | yes | ad hoc, post hoc | ad hoc, post hoc |
| S1+adversarial (rejected escalation) | yes | ad hoc, post hoc | ad hoc, post hoc |

Only **3 of 19 registered arms** (S1, S1b, S5 — the pre-registered primary and its two
designated secondaries) were ever read on the held-out splits, in one single Tier-4 pass.
Two further arms were frozen-read **after the fact, outside the pre-registration**, on
explicit request rather than as part of the ladder's own protocol:

- **W3-GELSTM-frozen** — the matched-window winner in the OOF comparison (§2.6, 0.7500 vs
  TFGN's 0.7318) — had no held-out number because the Tier-4 arms were fixed before the W3
  block reported.
- **S1+adversarial** — the escalation arm that lost on OOF (0.7066 vs 0.7488) and moved its
  own target diagnostic the wrong way (`cohort_probe_auc` 0.86 → 0.94) — was deliberately
  not read at Tier 4 originally, since spending a one-shot estimate on an arm the stopping
  rule already rejects is exactly the discipline every other rejected rung was held to.

Both ad-hoc reads have since been executed (`CLASSIFIER/scripts/frozen_read_w3_advcohort.py`)
and permanently spend that arm's one-shot test/external read
(`score_frozen_split`'s overwrite guard now blocks a second read of either); the resulting
numbers are recorded in `run_summary.json` under each run's own `outputs/` directory but
are **not yet transcribed into this document** — that is a separate, explicit step. Every
remaining arm in the table above (S0a–S0d, S1c both versions, S2/S3/S4, SENS,
W3-random/winner) has **no** in-domain or external number at all, ad hoc or otherwise;
their only performance evidence is the pooled OOF column in §2.1.

<a id="19-interpretability-validation-pre-registered-independent-of-auc"></a>

### 1.9 Interpretability validation (pre-registered, independent of AUC)

Fixed in advance so it could not be dropped if the interpretability rung underperformed:
a permutation null on the region map's DMN overlap (1000 permutations, reported as a
percentile), stability of the map across folds and seeds (Spearman), and both statistics
computed separately per cohort to catch a map that is really tracking cohort rather than
disease. Three deviations were required and are recorded rather than silently substituted:

1. **Cross-seed, not cross-fold, stability.** The adapter persists only the winning fold's
   map per run (`(50, 200)` per seed), so 4 maps exist per rung, not 20. This is a genuine
   loss of power for the stability claim; recovering the full statistic would need a
   per-fold artifact hook and a 4-seed re-run.
2. **DMN only, not DMN/hippocampal.** The atlas every TFGN rung actually consumes
   (Schaefer-200 cortical) contains no hippocampal or other subcortical ROI. The overlap
   statistic is restricted to the Yeo-7 `Default` network, 46 of 200 ROIs.
3. **Network-label spin test, not subject-label permutation.** DMN membership is an
   anatomical label, so no subject-label permutation changes it. The null is instead 1000
   random reassignments of which 46 nodes carry the DMN label (preserving the true count),
   each recomputing overlap against the fixed observed top-30 nodes.

Because the learned gate rung was dropped, the quadrant map's temporal axis is the
model-free rank-sigmoid drift anchor `d̃`, computed offline from each subject's own data
with no checkpoint and no GPU — one learned axis (`s_topo`) against one measured axis.
The dropped gate's map is reported as a supporting panel with its rejection stated, not
promoted to the primary axis.

---

[↑ Back to top](#top)

<a id="2-results"></a>

## 2. Results

All numbers are pooled cross-validation OOF AUC, mean ± **SD** across seeds 42–45, and
were recomputed from per-subject `oof_predictions.csv` rather than from any run's own
summary line.

<a id="21-scorecard"></a>

### 2.1 Scorecard

| arm | N | pooled OOF AUC | ADNI | DELCODE | bal. acc | static N=1 |
|---|---|---|---|---|---|---|
| S0-demo (age+sex) | 248 | 0.5296 ± 0.0000 | 0.5162 | 0.5186 | 0.5139 | 0.5296 |
| S0c GELSTM-random | 248 | 0.5625 ± 0.0292 | 0.5348 | 0.6067 | 0.5631 | 0.5140 |
| S0d BrainTokenGT | 248 | 0.6207 ± 0.0338 | 0.6194 | 0.6217 | 0.5889 | 0.5353 |
| S0a logreg-drift | 248 | 0.7053 ± 0.0000 | 0.5564 | 0.9129 | 0.6776 | 0.5653 |
| S0b GELSTM-frozen | 248 | 0.7186 ± 0.0334 | 0.4971 | 0.9178 | 0.6505 | 0.4699 |
| **S1 flip (winner)** | 248 | **0.7488 ± 0.0033** | 0.6526 | 0.8741 | 0.7093 | 0.4919 |
| S1b SSL (sensitivity) | 248 | 0.7502 ± 0.0125 | 0.6624 | 0.8707 | 0.7037 | 0.5072 |
| S1c recon (original, invalid) | 248 | 0.5507 | — | — | — | — |
| S1c recon-random (corrected) | 248 | 0.5433 ± 0.0311 | 0.5335 | 0.5589 | 0.5477 | 0.5213 |
| S2 gate | 248 | 0.7308 ± 0.0160 | 0.6266 | 0.8700 | 0.6811 | 0.4969 |
| S3 fusion (void) | 248 | 0.7488 ± 0.0033 | 0.6526 | 0.8741 | 0.7093 | 0.4919 |
| S4 attn-pool | 248 | 0.6490 ± 0.0144 | 0.5415 | 0.8022 | 0.6111 | 0.5197 |
| S5 dual-score | 248 | 0.7331 ± 0.0173 | 0.6460 | 0.8530 | 0.6885 | 0.5010 |
| SENS (`min_visits=3`) | 140 | 0.7413 ± 0.0137 | 0.6328 | 0.8986 | 0.6907 | 0.5469 |
| W3 GELSTM-random | 248 | 0.5885 ± 0.0304 | 0.5564 | 0.6385 | 0.5541 | 0.5190 |
| W3 GELSTM-frozen | 248 | 0.7500 ± 0.0138 | 0.5848 | 0.9075 | 0.7000 | 0.5013 |
| W3 TFGN-winner | 248 | 0.7318 ± 0.0348 | 0.6584 | 0.8342 | 0.6756 | 0.4880 |

Two features of this table are worth naming directly. First, **all of the signal is
longitudinal**: S1's static N=1 row is 0.4919 — chance — against 0.7488 on full
trajectories. Second, S1 has the **tightest seed SD of any deep model here** (0.0033),
which is a direct consequence of the strict-determinism work rather than luck.

<a id="22-stopping-rule-verdicts-every-rung-against-s1"></a>

### 2.2 Stopping-rule verdicts (every rung against S1)

| contrast | fold-matched Δ ± SE (ratio) | pooled Δ ± SE (ratio) | verdict |
|---|---|---|---|
| S1b vs S1 | +0.0120 ± 0.0019 (+6.46) | +0.0014 ± 0.0074 (+0.19) | disagree → one-SE tie-breaker → **S1** |
| S1c-random vs S1 | −0.1587 ± 0.0099 (−15.98) | −0.2055 ± 0.0150 (−13.72) | dropped |
| S2 gate vs S1 | −0.0064 ± 0.0075 (−0.85) | −0.0180 ± 0.0086 (−2.08) | dropped |
| S3 fusion vs S1 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | **void — not a result** |
| S4 attn-pool vs S1 | −0.0558 ± 0.0058 (−9.60) | −0.0999 ± 0.0083 (−11.96) | dropped |
| S5 dual-score vs S1 | −0.0046 ± 0.0067 (−0.68) | −0.0158 ± 0.0101 (−1.56) | **kept** (pre-registered) |

**The S1b fork.** The two legitimate Tier-2 statistics disagree: the fold-matched one
(within-seed pairing, 5 paired folds per seed) passes at ratio 6.46, the pooled one
(collapsing each seed to one number before differencing) fails at 0.19. The pre-registered
one-standard-error tie-breaker resolves it without picking a side after the fact — S1's
pooled OOF (0.7488) sits 0.0014 from S1b's (0.7502), well inside one SE (0.0074) under
either statistic — so **S1, the simpler configuration with no SSL-checkpoint dependency,
carries forward**. S1b is retained as a documented sensitivity arm. An earlier read had
selected S1b; that decision was reversed on the OOF artifacts and the reversal is
recorded, not silently applied.

**S1c.** The original run inherited the pre-reversal fork and is recorded as
**undecidable** — it tests an unregistered configuration, and reporting its 0.5507 against
S0b's 0.7186 would misattribute an untested auxiliary-loss interaction to the architecture.
The protocol-valid re-run scores 0.5433 ± 0.0311, ~0.20 AUC below S1: **the reconstruction
objective as configured collapses the model.** The honest reading is that the S0b ↔ S1c
contrast is *lost by the auxiliary objective*, not by the flip; a λ sweep to rescue it was
deliberately not run, since an ad-hoc search triggered by having seen 0.5507 is exactly
what the pre-registration exists to prevent.

**S5 is kept, not rejected.** Its fold-matched Δ is −0.0046 ± 0.0067, i.e. |Δ| < SE —
classification-neutral, which is precisely what "zero risk to the backbone" was
pre-registered to mean. It is kept as the interpretability layer.

<a id="23-the-winner-and-why-the-win-is-not-capacity"></a>

### 2.3 The winner, and why the win is not capacity

The selected model is **S1**: node-shared LSTM → mean-pool → linear head. Under
`recon_target: none` no GVAE is constructed, so `fusion: z_only` in its config string is a
no-op and **the winning TFGN contains no graph-propagation stage at all**. That is a
substantive architectural finding and belongs in the results, not a footnote.

| arm | total params | trainable | OOF AUC |
|---|---|---|---|
| **S1 flip (winner)** | **68,417** | **68,417** | **0.7488** |
| S5 dual-score | 68,482 | 68,482 | 0.7331 |
| S2 gate | 72,642 | 72,642 | 0.7308 |
| S0b GELSTM-frozen | 965,897 | 520,905 | 0.7186 |
| S0c GELSTM-random | 965,897 | 965,897 | 0.5625 |

The winner is **14.1× smaller** than the spatial-first baseline it beats by 0.186 AUC
(S0c) and **7.6× smaller in trainable parameters** than the pretrained one it beats by
0.030 (S0b), so the gain cannot be capacity — it comes from deferring pooling until after
node-level temporal encoding, which is exactly what the flip hypothesis claimed. The same
table also answers the objection from the other side: S2 and S5 *add* parameters to S1 and
both score lower.

<a id="24-s3-is-void-not-rejected"></a>

### 2.4 S3 is void, not rejected

`tfgn-s3-fusion-pooled` sets `fusion: concat_residual` on top of S1's
`recon_target: none`. Under that setting the model never constructs a GVAE
(`model/TFGN/models.py:95-105`), so `fusion_module = None` and the forward pass takes
`h_fused = h_T` regardless of the flag (`models.py:175-183`) — the knob is dead code on
this branch by construction. Verified rather than inferred: S3's `oof_predictions.csv` is
**bit-identical to S1's on all four seeds** (max |Δprob| = 0.0e+00). The fusion question is
**untestable within this ladder**, because its only available parent (S1c-random) failed;
branching S3 from a collapsed parent would answer nothing about fusion.

<a id="25-sequence-length--the-sens-decomposition"></a>

### 2.5 Sequence length — the SENS decomposition

SENS's 140 subjects are a strict subset of S1's 248, which is what makes two effects
separable that SENS alone conflates. Restricting **S1's own** OOF predictions to those same
140 subjects:

| seed | S1 restricted to ≥3-visit subjects | SENS (trained on ≥3-visit only) |
|---|---|---|
| 42 | 0.7762 | 0.7385 |
| 43 | 0.7656 | 0.7499 |
| 44 | 0.7603 | 0.7230 |
| 45 | 0.7825 | 0.7535 |
| mean | **0.7712** | **0.7412** |

- **Subgroup difficulty (+0.022):** S1 scores 0.7712 on the ≥3-visit subgroup vs 0.7488
  overall — longer trajectories are more predictable. This is the pre-registered
  sequence-length signal, and it is positive.
- **Training-pool cost (−0.030):** training *only* on those subjects scores 0.7412 on the
  same 140 — shrinking the pool 248 → 140 costs more than the longer sequences gain.

Both are true and not in tension; both retain the pre-registered "too small to decide on
its own" caveat. SENS's 0.7413 must never be reported beside S1's 0.7488 as a like-for-like
row — different N, different subjects, not fold-matched.

<a id="26-matched-window-head-to-head--a-crossover-not-a-defeat"></a>

### 2.6 Matched-window head-to-head — a crossover, not a defeat

| arm (T ∈ [2,3]) | pooled OOF AUC |
|---|---|
| BrainTokenGT (S0d) | 0.6207 ± 0.0338 |
| W3 GELSTM-random | 0.5885 ± 0.0304 |
| **W3 GELSTM-frozen** | **0.7500 ± 0.0138** |
| W3 TFGN-winner | 0.7318 ± 0.0348 |

Fold-matched, W3-TFGN vs W3-GELSTM-frozen is **−0.0244 ± 0.0027 (ratio −8.94)** — a
consistent loss across all four seeds. W3-TFGN vs BrainTokenGT is **+0.0999 ± 0.0162
(+6.17)** — a clean win over the SOTA competitor. Truncation moves the two architectures in
**opposite directions**:

| arm | full trajectory (T≥2) | matched window (T∈[2,3]) | Δ from truncation |
|---|---|---|---|
| GELSTM-frozen (spatial-first) | 0.7186 | 0.7500 | **+0.0314** |
| TFGN (temporal-first) | 0.7488 | 0.7318 | **−0.0170** |
| GELSTM-random | 0.5625 | 0.5885 | +0.0260 |

Spatial-first **gains** from truncation; temporal-first **pays** for it. The only regime
where temporal-first wins is the one with long sequences, and the only regime where
spatial-first wins is the one where the trajectory has been cut to a difference. **The
flip's gain is a long-sequence gain** — which is precisely what the full-trajectory table
was built to claim, confirmed from the other direction. The matched-window table must
never be presented as the main result: a windowing handicap that discards the visits the
recurrent architectures exist to exploit cannot double as evidence about the thesis.

*Reconciling the two ΔAUC statistics (+0.0999 fold-matched vs +0.1111 raw means):* not
single-class fold exclusion — checked explicitly, all 5 folds carry both classes in both
arms across all 4 seeds, and no fold is dropped from either statistic. Pooled AUC ranks all
248 subjects together and therefore additionally charges for cross-fold score
incomparability (each fold is a separately trained model with its own probability scale).
That penalty is −0.0268 for W3-TFGN and −0.0379 for BrainTokenGT, and the 0.0112 gap
between the penalties is exactly the gap between the statistics — itself a reportable
result: TFGN's per-fold outputs are more mutually comparable.

<a id="27-tier-4-held-out-reads-one-pass-spent-once"></a>

### 2.7 Tier-4 held-out reads (one pass, spent once)

Primary S1, secondaries S1b (sensitivity) and S5 (interpretability layer, never a competing
endpoint). Confirmed unspent immediately beforehand.

| arm | role | in-domain test AUC (n=64) | OASIS-3 AUC (n=60) | OOF (reference) |
|---|---|---|---|---|
| **S1 flip** | **primary** | **0.7909 ± 0.0162** | **0.4892 ± 0.0224** | 0.7488 ± 0.0033 |
| S1b SSL | secondary | 0.7760 ± 0.0568 | 0.4602 ± 0.0274 | 0.7502 ± 0.0125 |
| S5 dual-score | secondary | 0.7870 ± 0.0139 | 0.5070 ± 0.0109 | 0.7331 ± 0.0173 |

The in-domain test AUC landed *above* the OOF-derived prediction interval for S1 and S5.
This is reported as a fact, not as a second win: at n=64 the plan's own noise floor
(≈0.08) covers the gap, and OOF pools 248 subjects, so it remains the better-powered point
estimate. The headline claim is not updated on it; the honest phrasing is "did not degrade
in-domain, contrary to the winner's-curse prior".

**The load-bearing external result is OASIS-3, and it is at chance.** All three arms sit
within noise of 0.5 (per-seed SE at n≈60 is ≈0.075; the four-seed-mean SE is 0.011–0.037,
so every arm's mean is within ~1 SE of chance), tightly and consistently across seeds
(S1 0.4705–0.5217; S1b 0.4416–0.5006; S5 0.4972–0.5217). This is **not** below-chance
failure — it is **no signal transferred** to a cohort never seen in training or
pretraining, against 0.77–0.79 in-domain for the same models.

<a id="28-the-cohort-shortcut-and-a-mitigation-that-was-tried-and-failed"></a>

### 2.8 The cohort shortcut, and a mitigation that was tried and failed

The pre-registered escalation trigger — `cohort_probe_auc > 0.75` — **had been firing
since the first ladder runs and was not read against its own threshold until the Tier-4
pass pulled every arm's numbers together.** The probe was computed and persisted correctly
throughout; nothing was hidden, but the rule was not applied when it should have been.
This is recorded as a genuine process gap.

| arm | `cohort_probe_auc` | escalation (>0.75) |
|---|---|---|
| S1 flip | 0.860 ± 0.010 | **yes** |
| S1b SSL | 0.890 | **yes** |
| S1c recon-random | 0.860 | **yes** |
| S2 gate | 0.869 | **yes** |
| S3 fusion | 0.860 (identical to S1) | **yes** |
| S4 attn-pool | 0.847 | **yes** |
| S5 dual-score | 0.863 | **yes** |
| SENS | 0.742 | no (just under, smaller pool) |

A logistic probe decodes ADNI vs DELCODE from the pooled patient latent at ≈0.86 AUC
despite `cohort_conditioning: none` — the model encodes cohort identity strongly as a side
effect of learning to classify, with nothing telling it to. OASIS-3 is exactly the unseen
category the pre-registration flagged as ill-defined under such a shortcut, so the
near-chance transfer is the *predicted consequence*, not an unrelated failure.

**The pre-registered remedy was then run exactly as specified** — `tfgn-s1-advcohort-pooled`,
a gradient-reversal cohort head attached to the same pooled embedding the probe scores,
everything else identical to S1, all four seeds — **and it failed on both axes:**

| arm | OOF AUC | `cohort_probe_auc` |
|---|---|---|
| S1 (baseline) | 0.7488 ± 0.0033 | 0.8600 ± 0.0074 |
| **S1 + adversarial** | **0.7066 ± 0.0075** | **0.9411 ± 0.0131** |

Tier-2: fold-matched −0.0193 ± 0.0022 (ratio −8.68), pooled −0.0422 ± 0.0051 (ratio −8.31)
— consistent across all four seeds, not noise. Classification dropped by more than any
single rung's loss in the entire ladder, and the diagnostic the remedy exists to suppress
*rose* from 0.86 to 0.94. A 2-epoch smoke test had shown the probe at 0.74 and looked
encouraging; in hindsight that measured "how much has this network encoded anything yet",
not the phenomenon under test — a standing caution for early-stop smoke checks in this
codebase.

The leading hypothesis is that `cohort_adv_lambda: 1.0` scales only the reversed gradient,
with no separate weight on the cohort loss term (textbook-minimal DANN), and that a weak,
losing adversarial game can increase observed cohort separability relative to no
adversarial term at all. **This is a hypothesis, not a diagnosis** — no gradient-magnitude
comparison was run. Deliberately not done: no second Tier-4 read was spent on an arm the
stopping rule already rejects, and no λ sweep was run, since quietly retrying after seeing
the result is the undocumented re-run the pre-registration exists to prevent.

**Reporting rule.** The OASIS-3 line reads: *no evidence of transfer to an unseen cohort
(AUC 0.4892 ± 0.0224, indistinguishable from chance), consistent with an un-escalated
cohort-identity shortcut (`cohort_probe_auc` ≈ 0.86 ≫ 0.75); an adversarial
gradient-reversal mitigation was attempted and did not recover transfer — cohort-invariant
representation learning under this pooling protocol is an open problem, not a solved one.*
It is neither a clean external-validation number nor an unexplored trigger.

<a id="29-interpretability-validation--reproducible-but-not-dmn-specific"></a>

### 2.9 Interpretability validation — reproducible, but not DMN-specific

Computed on `s_topo` (from S5, primary) and the offline drift anchor `d̃` (temporal axis),
with the dropped gate's map as a supporting panel.

| statistic | `s_topo` (S5, primary) | `d̃` (offline) | `gate_scores` (S2, dropped) |
|---|---|---|---|
| DMN overlap (top 30 of 200) | 8/30, pct 77.9, p=0.351 | 6/30, pct 41.6, p=0.739 | 0/30, pct 0.0, p=1.000 |
| Cross-seed Spearman, mean [range] | 0.928 [0.898, 0.968] | 1.000 [1.000, 1.000] | 0.823 [0.676, 0.923] |
| ADNI-only DMN overlap (mean) | 7.0 | 10.0 | 0.25 |
| DELCODE-only DMN overlap (mean) | 9.5 | 3.0 | 0.0 |

Quadrant scatter (`s_topo` vs `d̃`, cross-seed-averaged node maps, median split):
HH=66, HL=34, LH=34, LL=66, Spearman r=0.456 (p=1.2e-11). `d̃`'s cross-seed Spearman is
exactly 1.0 because it is a deterministic function of each subject's own data and three of
four S5 seeds select the same best fold; where a seed selects a different fold, it drops to
0.633, tracking the subject-set change exactly as expected.

**Verdict, stated plainly.** Neither axis clears the DMN spin test — **the pre-registered
"the gate targets DMN/hippocampal regions" claim is not supported** on the atlas actually
available. What *is* supported: `s_topo` is stable across seeds (mean r=0.928, both
cohorts separately above 0.91) and correlates with the independent, model-free drift
anchor (r=0.456, p=1.2e-11). The learned topology score is reproducible and tracks
something coherent about within-subject FC change — just not preferentially the DMN. The
dropped gate's map is markedly less stable (0.823) and shows an odd DMN anti-enrichment
(0/30 in every fold, both cohorts), reported for the record and not leaned on.

Since every performance rung above S1 was dropped, this is the *entire* interpretability
contribution, and it is reported as a negative result on the enrichment claim with a
positive result on reproducibility.

<a id="210-scaling-gate-closed"></a>

### 2.10 Scaling gate: closed

The pre-registered gate for the scaled Block B was a cumulative gain from S1c-random
through S5 exceeding the SE of the seed-level differences. The chain delivered −0.1587
(S1c-random), −0.0064 (S2), void (S3), −0.0558 (S4), −0.0046 (S5); no rung above S1 was
kept. **Block B does not run.** Paired with the parameter counts in §2.3, the conclusion
the gate itself specified is the honest thesis result: **signal quality and sample size,
not capacity, are the bottleneck** — and a winner 14× smaller than the baseline it beats
is the evidence for that sentence, not merely consistent with it.

---

[↑ Back to top](#top)

<a id="3-summary-and-limitations"></a>

## 3. Summary and limitations

**What the ladder establishes.**

1. Flipping the pipeline order helps in-domain: temporal-first S1 (0.7488 OOF, 0.7909 on
   the held-out test) beats matched spatial-first without pretraining (S0c, 0.5625) and
   pretrained spatial-first (S0b, 0.7186), at 14× and 7.6× fewer parameters respectively.
2. The advantage is a **long-sequence** advantage: under the competitor's short window the
   two architectures cross over, spatial-first gaining and temporal-first losing.
3. All of the signal is longitudinal — the single-visit baseline is at chance (0.4919).
4. Nothing above the bare flip survived: gate, fusion (void), attentive pooling, the
   reconstruction objective, and node-LSTM SSL were all dropped or neutral. The winning
   model has no graph-propagation stage.
5. The interpretability map is reproducible across seeds and correlates with an
   independent model-free drift measure, but does not show DMN enrichment.

**Limitations, stated as such.**

- **No external transfer.** OASIS-3 is at chance for every arm. The leading explanation is
  a cohort-identity shortcut (probe ≈0.86 ≫ the 0.75 threshold); the pre-registered
  adversarial mitigation was run and failed on both axes. Cohort-invariant representation
  learning under this pooling protocol is open.
- **A missed escalation.** The probe threshold fired from the first ladder runs and was not
  read against its own rule until the final pass. Every in-domain result is unaffected —
  they are OOF/CV quantities that do not depend on cross-cohort transfer — but the process
  gap is recorded.
- **Sample size.** n=248 CV / 64 test / 60 external. Differences below ≈0.04 (OOF) or
  ≈0.08 (test) are not resolvable, and every keep/drop verdict is a heuristic screen at
  n=4 seeds, never a significance test.
- **S0b's cohort asymmetry.** The pretrained spatial-first baseline is near chance on ADNI
  (0.4971) and very strong on DELCODE (0.9178); the pooled number averages two very
  different regimes, which is why per-cohort columns are reported throughout.
- **Two contrasts unanswered.** S0b ↔ S1c is lost to the auxiliary objective's collapse,
  not to the architecture; the fusion question is untestable because its only parent rung
  failed. Neither was rescued by an undocumented re-run.
- **BrainTokenGT is a caveated reference**, not a clean baseline: its scatter/gather ops
  are nondeterministic and its same-seed AUC has been observed to span 0.357–0.708.
- **Interpretability power was reduced** from 20 maps (5 folds × 4 seeds) to 4 (best fold
  per seed) by an artifact limitation, and the DMN statistic is cortical-only because the
  atlas used carries no subcortical ROI.

---

[↑ Back to top](#top)

<a id="4-reproducibility"></a>

## 4. Reproducibility

- Pre-registration: `DOCS/temporal-first-ablation.md` (every deviation recorded as an
  addendum, never a silent edit).
- Execution record and verified scorecard: `DOCS/flipped/PLAN.md`.
- Registry: `CLASSIFIER/experiments/temporal_first.yaml` (one entry per arm × seed);
  shared config `CLASSIFIER/configs/tfgn_pooled.json`; model `CLASSIFIER/model/TFGN/`;
  adapters `CLASSIFIER/adapters/{tfgn,logreg_drift}.py`.
- Aggregation: `CLASSIFIER/notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb`
  (Tiers 1–3 from `oof_predictions.csv` only; the Tier-4 frozen read in its final section,
  archived as an executed notebook under `_results/`).
- Every arm ran under strict determinism with seeds 42–45; per-subject OOF predictions are
  persisted for every run, so every statistic in §2 is recomputable per subject and per
  fold.

[↑ Back to top](#top)
