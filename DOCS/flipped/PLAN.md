# Temporal-First Graph Network (TFGN) — pooled ADNI+DELCODE → external OASIS-3

## Context

Every longitudinal model in this repo is **spatial-then-temporal**: each visit's
Schaefer-200 FC graph is collapsed to one 64-d vector by the frozen GAAE encoder
(`GELSTMClassifier.encode_visit`, `model/GELSTM/models.py:191`), and only then does an
LSTM see the sequence. Region identity is destroyed before any trajectory is modelled.

This plan builds the inverted architecture — **temporal-then-spatial**: a node-shared
LSTM encodes each region's own FC-row trajectory, a learned saliency gate suppresses
static regions, a GVAE then propagates the surviving dynamics over the baseline topology,
a residual skip protects the unsmoothed temporal features, and an anchored dual-score
readout gives a 2-D regional diagnostic map.

Two things make it worth doing now beyond the architecture itself:

1. **No pooled-cohort protocol exists.** Every `ext-adni-*` / `ext-oasis3-*` entry trains
   and tests *inside* one cohort. `DOCS/__artifacts__/timeline/SECTION_08...md:73-93`
   registers "train DELCODE+ADNI → test held-out OASIS-3" as an unstarted supervisor
   request. This plan builds it, and it is what gives the ladder statistical power
   (248 CV subjects instead of 133).
2. **The current encoder is not earning its place.** On DELCODE, `encoder_init=none`
   (0.83) ≥ `pretrained_frozen` (0.78); on ADNI/OASIS-3 both sit at chance. Either the
   temporal-first flip changes that, or capacity was never the bottleneck — and the
   pre-registered stopping rule below is what decides which, before anything is run.

**Honest expectation, stated up front.** In-domain test is n≈64 and external is n=60.
An AUC difference below ~0.08 is not resolvable at these sizes. The ladder is designed
to *rule changes out* cheaply, not to guarantee a win. Block B (scaling) is deliberately
gated on Block A clearing the noise floor.

### Decisions already fixed

| | |
|---|---|
| Training pool | ADNI + DELCODE `downstream` train+val, `min_visits=2` → **248 subjects** |
| In-domain test | ADNI + DELCODE `downstream` test, `min_visits=2` → **64 subjects**. Drives every ladder decision. |
| External test | **All 60 OASIS-3 subjects** (35+12+13), scored once per arm at the in-domain OOF threshold. Never used for selection. |
| Autoencoder | New GAAE pretrained on pooled **ADNI+DELCODE** unlabelled scans (~3 700 graphs). OASIS-3 excluded entirely so it stays fully external. |
| Cohort covariate | **Held out of the model** by default; made explicit via a mandatory cohort-decoding probe with a pre-registered escalation threshold (§3, §0.1). |
| Scope | Block A ladder now; Block B scaling written but gated. |
| Stage-0 references | logreg-drift, GELSTM `pretrained_frozen`, GELSTM `random`, BrainTokenGT — all re-run on the pooled protocol. |

---

## Phase 0 — Pre-registration and determinism (do first, no GPU)

### 0.1 `DOCS/temporal-first-ablation.md`

Written and committed *before* any run, in the style of
`DOCS/reconstruction-value-ablation.md`. It fixes the arm table, the config knob per arm,
**and the four things below, so none of them can be quietly changed after seeing results.**

**(a) Stopping rule.**

> For ladder step *k* vs *k−1*: the independent unit is the **seed** (n=4), not the fold.
> Compute each seed's mean paired per-fold ΔAUC, then report mean and SE across the 4 seed
> means. A step is kept only if `mean(Δ) > SE(Δ)` on the **in-domain** test AUC. At n=4 no
> p-value is claimed and the SE is itself high-variance — this is a heuristic screen, not
> a test. **A rung that fails means "undetectable at this sample size", never "harmful";
> a rung that passes means "worth carrying forward", never "significant".** External
> OASIS-3 AUC is reported with a bootstrap CI (`common/comparison.paired_bootstrap_ci`)
> and never used to select an arm.

**Superseded (2026-08-24 addendum, `DOCS/temporal-first-ablation.md` "Evaluation &
Comparison Protocol").** The stopping rule now reads **pooled CV out-of-fold (OOF)
AUC**, not in-domain test AUC — more statistical power (n=248 vs n=64), and the
64-subject in-domain test set is read exactly once, on the frozen final winner, never
during the ladder. Formula and machinery are otherwise unchanged. See Phase 4.5 below.

**(b) Reconstruction target.** `σ(ZZᵀ) ∈ (0,1)` cannot be BCE'd against
`ΔA = A^(T) − A^(1) ∈ [−2,2]`. Three well-typed targets, one chosen up front:

| `recon_target` | decoder | target | loss |
|---|---|---|---|
| `delta_a_topk` **(ladder default)** | `σ(ZZᵀ)` | binary change-mask `M_ij = 1[\|ΔA_ij\| ≥ q_{1−κ}]`, κ = 0.10, quantile computed **per subject** | `BCEWithLogits` with `pos_weight = (1−κ)/κ` |
| `delta_a_mse` | `tanh(ZZᵀ/√d)` — no sigmoid | `ΔA/2 ∈ [−1,1]` | MSE |
| `a_last` | `σ(ZZᵀ)` | `(A^(T)+1)/2 ∈ [0,1]` | BCE |

`delta_a_topk` is the ladder default: it is the target that literally encodes "which
edges changed", it keeps the sigmoid decoder valid, and it is the same quantity the
edge-change-ranking objective in Block B uses. `delta_a_mse` is the documented fallback if
the mask degenerates (guard: raise if fewer than 1 % or more than 50 % of edges are
positive for any subject). `a_last` exists only for a Block-B contrast.

**(c) Anchoring quantities.**

- *Topological anchor.* Eigenvector centrality is ill-defined on a signed FC matrix
  (negative weights break Perron–Frobenius). The anchor is **strength centrality on
  `|A_0|`** — row sums of the absolute kNN-sparsified baseline FC — z-scored with
  train-fold statistics. (Eigenvector centrality on `|A_0|` is a valid alternative and is
  recorded as such, but strength is the pre-registered one.)
- *Drift anchor, reconciled with the sparsity prior.* A raw drift target
  `‖x_i^(T) − x_i^(1)‖₂` is dense (mean ≈ 0.5 after any normalisation) and fights
  `KL(s̄ ‖ ρ=0.15)`, which pushes most gates to zero. The anchor target is therefore the
  **drift quantile pushed through a sharp sigmoid centred at the (1−ρ) quantile**:

  ```
  q_i = rank(d_i) / (N−1)                    # within-subject drift quantile, ∈ [0,1]
  d̃_i = σ( (q_i − (1−ρ)) / τ_d ),  τ_d = 0.05
  ```

  so `mean(d̃) ≈ ρ` **by construction** and the two regularisers agree instead of
  cancelling. Ranks also make the anchor scale-free across cohorts with different FC
  amplitude. Additionally `λ_drift = 0.1 · λ_sparse`, so the sparsity prior dominates if
  they ever do disagree.

**(d) Gate-map validation, pre-registered so it cannot be dropped if S5 underperforms.**
Regardless of S5's AUC: permutation null over the gate map `s` (1 000 label permutations,
report the DMN/hippocampal overlap percentile), cross-fold Spearman stability of `s`
across the 5 folds × 4 seeds, and the same two statistics computed **separately per
cohort** (see the cohort probe below).

### 0.2 Strict determinism

`SHARED/seeding.py` gains an opt-in `set_seed(seed, *, strict: bool = False)` that
additionally calls `torch.use_deterministic_algorithms(True)` and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. TFGN uses GATv2 scatter-backward and sparsemax, both
nondeterministic on GPU by default — without this the SE-based stopping rule would be
measuring GPU noise. Threaded through as `strict_determinism: true` in the shared TFGN
config. Existing runs are untouched (`strict` defaults False).

---

## Phase 1 — Pooled-cohort data plumbing

**1.1 `DATA/manifest/build_pooled_assets.py`** (new CLI, mirrors `build_cohort_splits.py`)

Four artefacts, each with a printed report:

- `DATA/POOLED_ADNI_DELCODE/SPLITS/downstream/{train,val,test}.csv` — union of the two
  cohorts' downstream splits, harmonised to
  `subject_id, cohort, converter_status, sex, age, n_scans, allowed_days, allowed_months`.
  DELCODE's `Pseudonym`→`subject_id`, `diagnosis`→`converter_status`. **Rows keep only
  their native allow-list column populated**; the other is empty. `--min-visits N` drops
  subjects below the floor (default 2 → 47 DELCODE subjects dropped; `--min-visits 3` is
  the sensitivity arm).
- `DATA/ADNI/__metadata__/SPLITS/pretrain/{train,val,test}.csv` — ADNI's missing pretrain
  split, built with DELCODE's leakage rule (`create_pretrain_data_splits.py:10-16`):
  ADNI downstream val/test subjects are forced into pretrain val/test, so
  `pretrain train ∩ downstream {val,test} = ∅`.
- `DATA/POOLED_ADNI_DELCODE/SPLITS/pretrain/{train,val,test}.csv` — concatenation of
  the two pretrain splits.
- `DATA/POOLED_ADNI_DELCODE/__fc_wholebrain_sch200_flat__/matrices/` — a **symlink
  farm** into both cohorts' `.npz` files. Subject-id prefixes are disjoint
  (`sub-ADNI002S1261` / `sub-011d501d1` / `sub-OAS30001`), so a single glob root works and
  the GAAE static path needs no code change at all.

Asserts, all fail-loud: no subject in two splits; every symlink resolves; every retained
subject has ≥ `min_visits` resolvable FC files; no DELCODE row carries a populated
`allowed_days` and vice-versa. Tests in `DATA/manifest/tests/test_pooled_assets.py`.

**1.2 `CLASSIFIER/common/pooled_data.py`** (new)

```python
COHORT_ROOTS: dict[str, str]   # 'delcode'|'adni'|'oasis3' -> matrices dir
def build_multicohort_bundle(df, *, cohort_roots, **ds_kwargs) -> Bundle
```

Splits `df` by its `cohort` column, builds one `LongitudinalSubjectDataset`
(`model/GELSTM/dataset.py:53`) per cohort with that cohort's root and `cohort=` tag, and
concatenates the items into one `Bundle`. Per sub-frame it **drops the non-native
allow-list column before constructing** and raises if the retained column is entirely
null — otherwise `LongitudinalSubjectDataset`'s first-match column pick
(`dataset.py:142-147`) would silently disable DELCODE's post-conversion leakage filter.
Each item carries its `cohort` tag forward for the probe. A `df` without a `cohort` column
falls through to today's single-cohort path unchanged.

**1.3 Adapter wiring** — `GELSTMAdapter.prepare_data` (`adapters/gelstm.py:227`) routes
through `build_multicohort_bundle` when the frame has a `cohort` column. Same for the new
TFGN and logreg-drift adapters. Single-cohort behaviour is byte-identical.

**1.4 Notebook — `notebooks/LONGITUDINAL/LONGITUDINAL_COMMON_DELCODE.ipynb`**

Four additive edits (the notebook already drives every longitudinal model through the
adapter contract, so no new classification notebook is needed):

- *Cell 8*: a `POOLED` branch on `DATASET` (alongside `ADNI` / `OASIS`) pointing
  `SPLITS_DIR` at the pooled dir, `cohort_tag='pooled'`, and resolving
  `EXTERNAL_TEST_CSV` + `EXTERNAL_COHORT` when `TRAIN_CONFIG['external_test_cohort']` is set.
- *Cell 9*: `run_full_audit` also runs over the pooled CSVs, plus an assertion that the
  external cohort's subject set is disjoint from the CV pool.
- *After cell 27*: an external-test cell — `EXT_BUNDLE = prepare_data(ext_df)`,
  `eval_split(BEST_MODEL_STATE, EXT_BUNDLE, ACTIVE_THRESHOLD, device=device)`, then
  `record_external_metrics(...)`. Guarded on `EXTERNAL_TEST_CSV is not None`.
- *Cohort probe cell*: logistic regression decoding `cohort` from the OOF patient latents,
  reported as `cohort_probe_auc` (details in §3).

All four are guarded so every existing entry is a no-op.

**1.5 `common/run_artifacts.py`** — new `record_external_metrics(run_dir, metrics, *,
threshold, threshold_method, cohort)` writing into `run_summary["metrics"]` as
`ext_<cohort>_auc` / `_sensitivity` / `_specificity` / `_f1` so `collect_results`
(`common/experiment_utils.py:230`) surfaces them as `metric.ext_oasis3_auc` in
`outputs/RESULTS.csv` with no ledger changes.

---

## Phase 2 — Pretraining (2 runs, must finish before the ladder)

**P1 `gaae-pretrain-pooled-adni-delcode`** — `mode: static`, `adapter: gaae`, notebook
`STATIC/STATIC_COMMON_DELCODE.ipynb`, `dataset: POOLED_WHOLE_BRAIN`. Same architecture as
`ethereal-planet-16` (`configs/gaae_delcode_whole_brain.json`: hidden 128, latent 64,
heads 2, `adjacency_k` 16, `adj_loss_weight` 0.2) so every downstream arm is directly
comparable to the existing DELCODE-only checkpoint. Reads the pooled pretrain split and
the symlink farm from Phase 1. Produces the `checkpoint_path` used by every S0 GELSTM arm.

**P2 `tfgn-nodelstm-ssl-pooled`** — self-supervised per-node next-visit FC forecasting.
New notebook `notebooks/LONGITUDINAL/LONGITUDINAL_TFGN_SSL_POOLED.ipynb` (this one genuinely
does not fit the classification adapter contract — no labels, no CV, no threshold).
Trains only the node-shared LSTM + input projection to predict `x_i^(t+1)` from history
over every ≥2-visit ADNI+DELCODE subject in the **pretrain train** split, MSE loss,
cosine schedule. Saves a full-state checkpoint under `outputs/tfgn-nodelstm-ssl-pooled/`.
Consumed by the `node_lstm_init` knob.

---

## Phase 3 — The TFGN model

New package `CLASSIFIER/model/TFGN/` — pure logic, no I/O, no path construction
(`.claude/rules/architecture.md`).

| file | contents |
|---|---|
| `dataset.py` | `TFGNItem` wrapper over `LongitudinalSubjectDataset`'s dict: stacks `X ∈ R^{T×200×200}`, `log Δt` (from `visit_identity`, **cumulative months**, standardised with train-fold stats), `A_0` edge_index (`graphs[0]`), the per-subject **change-mask** `M` (§0.1b), **strength centrality of `|A_0|`**, the **rank-sigmoid drift anchor** `d̃` (§0.1c), covariates `[age, sex]`, and the `cohort` tag (carried for the probe, **not** fed to the model). All derived quantities cached per subject. |
| `layers.py` | `NodeSharedLSTM`, `TemporalSaliencyGate` (`s_i = σ(wᵀ LeakyReLU(W_s h_i))`, residual scaling `(1+s_i)h_i`), `GVAEEncoder` (GATv2 μ/logσ² heads + FiLM on μ), `ConcatResidualFusion` (`LayerNorm(W_u[h_i‖z_i])`), `AttentivePool`, `sparsemax`, `DualScoreReadout` |
| `models.py` | `TFGNClassifier` — assembles the stages from the config; every stage is switchable so the ladder is *one* model with knobs, exactly as `encoder_init` is for GELSTM |
| `losses.py` | `gate_sparsity_kl(s, rho)`, `drift_anchor_mse(s, d̃)`, `centrality_anchor_mse(s_topo, c)`, `change_mask_bce(logits, M, pos_weight)` / `delta_a_mse(...)`, `free_bits_kl` (reuse `model/VGAE/losses.py::kl_divergence`) |
| `train.py` | `train_epoch(...)`, `evaluate(..., *, eval_cfg)` returning the same key bundle as `model/GELSTM/train.py:112` (`auc, sensitivity, specificity, f1, best_threshold, probs, targets, preds, subject_ids, n_scans`) so all shared notebook cells work untouched; `make_batches(items, bs, shuffle, rng=...)` |

`CLASSIFIER/configs/tfgn.py` — `TFGNTrainConfig` + `TFGNEvalConfig` dataclasses, every
hyperparameter a field with a default (`.claude/rules/configs.md`). Ladder knobs:

```
node_lstm_init: "random" | "pretrained_frozen" | "pretrained_finetuned"   # P2 checkpoint
use_gate: bool                lambda_sparse / lambda_drift: float         gate_rho: float
recon_target: "none" | "delta_a_topk" | "delta_a_mse" | "a_last"
lambda_recon / beta_kl / free_bits / beta_warmup_epochs: float   change_mask_kappa: float
fusion: "z_only" | "concat_residual"
readout: "mean" | "attention"     dual_score: bool   lambda_cent: float   tau: float
cohort_conditioning: "none" | "film" | "adversarial"     # default "none" — see below
encoder_init: <reuse configs/encoder.py's four-arm enum for the GVAE>
```

**Cohort-shift control — the explicit decision.** ADNI and DELCODE differ in scanner,
protocol and follow-up rhythm (median interval 371 d vs a 90 %-at-12-months DELCODE
protocol), so a pooled model can learn cohort identity as a shortcut. Feeding cohort as a
FiLM covariate would make external transfer ill-defined — OASIS-3 is an unseen category,
and the model's behaviour on an untrained one-hot slot is unpredictable. So the
pre-registered default is `cohort_conditioning: "none"`, paired with a **mandatory probe**:
a logistic regression decoding cohort from the OOF patient latents, reported as
`cohort_probe_auc` in every pooled run's `run_summary`. Pre-registered escalation: if
`cohort_probe_auc > 0.75` on the winning arm, run `cohort_conditioning: "adversarial"`
(gradient-reversal cohort head) as an additional arm and report both. The aggregation
notebook additionally reports the gate map and quadrant scatter **split by cohort**.

`CLASSIFIER/adapters/tfgn.py` — `TFGNAdapter(LongitudinalAdapter)` implementing the six
hooks + descriptors, modelled directly on `adapters/gelstm.py`. Per-fold `StandardScaler`
on the temporal embeddings, the `log Δt` statistics and the centrality z-scoring all ride
inside the composite `state` so the winning fold's statistics survive into the test /
early-detection / trajectory hooks. `extra_artifacts` writes `gate_scores.npy`,
`dual_scores.npy` and `cohort_tags.npy`. Registered as `"tfgn"` in
`adapters/__init__.py:49`.

`CLASSIFIER/adapters/logreg_drift.py` — `LogRegDriftAdapter` (key `logregdrift`) for
Stage 0: features `[PCA₃₂(vec(ΔA)), n_visits, total follow-up months, age, sex]` into
`model/classification/logreg_cv.py::train_logreg_cv`. Making it an adapter rather than a
notebook means the baseline goes through the identical CV, threshold, external-test and
ledger path as every other arm.

Tests: `CLASSIFIER/tests/test_tfgn.py` (shape contracts per stage; gate bounded in (0,1);
sparsemax produces exact zeros; **change-mask is in {0,1} with density within [0.01,0.5]
or raises**; **`mean(d̃) ≈ gate_rho` within tolerance**; **centrality is finite and
non-negative on a signed FC input**; `recon_target="none"` really zeroes that gradient
path; two identical-seed forward passes are bit-identical under `strict=True`),
`test_pooled_data.py` (cohort dispatch; the all-null-allow-column guard raises),
`test_logreg_drift.py`.

---

## Phase 4 — Block A: the ladder (~50 runs)

Registry: **`CLASSIFIER/experiments/temporal_first.yaml`**, one shared
`configs/tfgn_pooled.json`, arms differing only in `hyperparams:` — the
`experiments/ablation.yaml` pattern. Every entry:
`dataset: POOLED_WHOLE_BRAIN`, `threshold_mode: best-f1`,
`notebook: notebooks/LONGITUDINAL/LONGITUDINAL_COMMON_DELCODE.ipynb`,
`hyperparams.external_test_cohort: oasis3`, `hyperparams.min_visits: 2`,
seeds 42/43/44/45.

| Rung | id prefix | knob change vs previous rung | Question |
|---|---|---|---|
| **S0a** | `tfgn-s0-logreg-drift-pooled` | `adapter: logregdrift` | Linear floor on ΔA |
| **S0b** | `tfgn-s0-gelstm-frozen-pooled` | `adapter: gelstm`, `encoder_init: pretrained_frozen` | Spatial-first **with** a self-supervised encoder |
| **S0c** | `tfgn-s0-gelstm-random-pooled` | `adapter: gelstm`, `encoder_init: random` | Spatial-first **without** one — the matched floor for S1 |
| **S0d** | `tfgn-s0-braintokengt-pooled` | `adapter: braintokengt` | Competitor reference |
| **S1** | `tfgn-s1-flip-pooled` | `node_lstm_init: random`, gate off, `recon_target: none`, `fusion: z_only`, `readout: mean` | **Flip alone, no self-supervision anywhere → compare to S0c** |
| **S1b** | `tfgn-s1b-ssl-pooled` | `node_lstm_init: pretrained_finetuned` (P2) | Does node-LSTM SSL forecasting help? |
| **S1c** | `tfgn-s1c-recon-pooled` | `recon_target: delta_a_topk`, `lambda_recon`, `beta_kl` + free bits + warmup | **Headline arm: both encoders self-supervised → compare to S0b** |
| **S2** | `tfgn-s2-gate-pooled` | `use_gate: true`, `lambda_sparse`, `lambda_drift = 0.1·λ_sparse`, `gate_rho: 0.15` | Does suppressing static regions help? |
| **S3** | `tfgn-s3-fusion-pooled` | `fusion: concat_residual` | Does preserving unsmoothed H help? |
| **S4** | `tfgn-s4-attnpool-pooled` | `readout: attention` | Does attentive pooling beat mean pooling? |
| **S5** | `tfgn-s5-dualscore-pooled` | `dual_score: true`, `lambda_cent` | Interpretability at zero risk to the backbone |
| **SENS** | `tfgn-sens-minvisits3-pooled` | winning config, `min_visits: 3` | Does the flip's advantage grow with sequence length? |

**Why S0c and S1c exist.** Without them the flip is handicapped: S0b's GAAE is
reconstruction-pretrained on 3 700 graphs while S1's GVAE learns only from the
classification gradient on 248 subjects, so an S1 loss would confound "temporal-first is
worse" with "this encoder never got self-supervision". The two clean comparisons are
therefore **S0c ↔ S1** (neither encoder pretrained) and **S0b ↔ S1c** (both pretrained).
S1c is the headline number for the thesis.

**Why SENS exists.** At `min_visits=2` an LSTM over T=2 encodes a *difference*, not a
trajectory — and 2-visit subjects are the plurality of the pool. `min_visits=3` shrinks
the CV pool to ~140 and the in-domain test to ~37 (OASIS-3 external to ~24), too small to
decide anything on its own, but it lets the thesis state whether the effect direction
strengthens with sequence length instead of leaving the question open.

Each rung inherits the *kept* knobs of the rungs before it. If a rung fails the stopping
rule its knob is dropped and the next rung branches from the last surviving config —
`temporal_first.yaml` carries the full chain, but the `hyperparams` of S2+ are finalised
after S1c reports. **S1b is the only fork**: its winner sets `node_lstm_init` for S1c–S5.

**Run order and dispatch** (both boxes are currently idle; >2 ids per call uses both):

```bash
cd /mnt/e/fyassine/ad-early-detection
.venv/bin/python scripts/gpus.py

# Phase 2 — pretraining, must complete first
scripts/dispatch.sh --pkg CLASSIFIER --id gaae-pretrain-pooled-adni-delcode
scripts/dispatch.sh --pkg CLASSIFIER --id tfgn-nodelstm-ssl-pooled

# Always dry-run a new arm before spending GPU
cd CLASSIFIER && python run_experiment.py --dry-run --id tfgn-s1-flip-pooled-seed42

# Stage 0 (16 runs), then each rung as a 4-seed block
scripts/dispatch.sh --pkg CLASSIFIER --id tfgn-s0-logreg-drift-pooled-seed42 ... (16 ids)
scripts/dispatch.sh --pkg CLASSIFIER --id tfgn-s1-flip-pooled-seed{42..45}
scripts/dispatch.sh --pkg CLASSIFIER --id tfgn-s1b-ssl-pooled-seed{42..45}
# ... one block per rung, reading the stopping rule between blocks

cd CLASSIFIER && python run_experiment.py --status --watch
python run_experiment.py --collect
```

Never launch the same id on both boxes — `outputs/<id>/latest` has no locking
(`.claude/rules/gpu-dispatch.md`). Budget: ~50 GPU runs, 30–90 min each once the
reconstruction loss and dual-score heads are active, two boxes → **20–24 h wall clock**.

**Aggregation** — `notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb`
(`source_experiment` chain over the rung ids): the rung-by-rung table (CV OOF AUC,
in-domain test AUC, external OASIS-3 AUC, `cohort_probe_auc`, each ± seed SD); the paired
seed-level ΔAUC with its SE against the stopping rule; DeLong and bootstrap CIs via
`common/comparison.py`; the S0c↔S1 and S0b↔S1c headline contrasts called out separately;
and the §0.1d gate-map validation (permutation null, cross-fold Spearman, per-cohort
split) plus the 2×2 quadrant scatter.

---

## Phase 4.5 — Evaluation & comparison protocol (addendum, 2026-08-24)

`DOCS/temporal-first-ablation.md`'s 2026-08-24 addendum moves the ladder onto a
four-tier evaluation protocol — floor gates, the OOF stopping rule, robustness vetoes,
and a single frozen test read — and requires **re-running the 24 already-complete
arms** (S0a/S0b/S0c/S0d/S1/S1b × 4 seeds) under the artifact contract this phase adds,
since none of them persisted per-subject OOF predictions.

**4.5.1 The OOF artifact contract.** `common/crossval.py::run_kfold_cv` gains an
optional `fold_probe(bundle_va, fold_out) -> {name: {subject_id: value}}` hook, called
once per fold; `CVResult` gains `oof_folds` (always populated) and `oof_extras`
(populated only when a probe is supplied). `common/oof.py` (new) — `build_oof_frame`
(tidy per-subject frame: `subject_id, fold, cohort, label, prob, n_scans, age, sex` +
any probe extras) and `oof_metrics` (pooled + per-cohort AUC, PR-AUC, balanced
accuracy, the static-baseline AUC when a `prob_n1` extra is present, scan-count
Spearman). `common/run_artifacts.py::record_oof_artifacts` writes
`oof_predictions.csv` and patches `run_summary["oof"]`; `collect_results`
(`common/experiment_utils.py`) surfaces it as `oof.*` columns in `RESULTS.csv`.

**4.5.2 Deferred test reads.** Every ladder arm sets `defer_test_eval: true`.
`LONGITUDINAL_COMMON_DELCODE.ipynb`'s Configuration cell resolves `DEFER_TEST_EVAL`
from it; the Test-Set / External Test-Set / ROC / Early-Detection / Trajectory cells
each no-op under it (additive `if not DEFER_TEST_EVAL:` guards — every pre-addendum,
non-deferred entry is unaffected). The CV-run cell passes a `fold_probe` that scores
each fold's own N=1-truncated validation subjects at that fold's own threshold, for
the Tier-1 static-baseline floor.

**4.5.3 Tier-1 floors.** `adapters/logreg_drift.py` gains `feature_set: "drift" |
"demo"` — `"demo"` restricts features to `[age, sex]`, no PCA/ΔA (four new registry
entries, `tfgn-s0-demo-pooled-seed{42..45}`). The static baseline is the OOF N=1 row
(no extra runs). The SSL persistence baseline was already computed by
`LONGITUDINAL_TFGN_SSL_POOLED.ipynb` itself and is already in
`tfgn-nodelstm-ssl-pooled`'s `run_summary.json["persistence_baseline"]` — nothing new
needed there.

**4.5.4 Tier-4 frozen reads.** `common/frozen_read.py` (new) —
`score_frozen_split(run_dir, df, ...)` reconstructs a run's adapter from its own saved
`run_summary.json` + checkpoint (`adapter.load_state`, no retraining), scores at the
run's OOF-derived threshold (`adapters.read_run_threshold`), and records through the
existing `record_test_metrics` / `record_external_metrics` — so `RESULTS.csv`'s schema
is identical whether the read happened inline (pre-addendum runs) or here. Fixed a
latent bug in `LogRegDriftAdapter.load_state` while wiring this: it returned the raw
checkpoint dict instead of unwrapping `model_state_dict` (the pattern every sibling
adapter already follows) — meant `eval_split(state, ...)`'s `state["pca"]` lookup
would have failed on any reload; unexercised until this phase's frozen-read path.

**4.5.5 Comparison notebook.** `notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb`
(new) — one section per tier, reading only `oof.*` / `oof_predictions.csv` through
Tier 3, and calling `common.frozen_read.score_frozen_split` exactly twice (in-domain,
OASIS-3) in its final section only.

**4.5.6 Re-run ledger.** All 24 completed arms re-run under the new contract
(`defer_test_eval: true` added to each registry entry) before S1c launches, so the S1b
fork decision is re-verified on OOF numbers computed the same way as every later rung.
S0d (BrainTokenGT) will not reproduce bit-for-bit (`.claude/rules/…` BrainTokenGT
determinism caveat, restated in `DOCS/temporal-first-ablation.md`) — its re-run number
replaces the old one with the caveat attached, not silently.

---

## Addendum (2026-08-24) — Matched-window SOTA comparison (additive, post-ladder)

**Motivation.** BrainTokenGT is architecturally capped at `min_visits=2, max_visits=3`
(Dong et al., MICCAI 2023 — the model was designed for short fixed-length sequences).
GELSTM and TFGN consume full trajectories (T up to ~10 in ADNI). Block A already carries
the S0d caveat, but a strict head-to-head against the SOTA competitor requires identical
inputs. This block adds it without altering any rung definition, the stopping rule, or
the two headline contrasts (S0c↔S1, S0b↔S1c).

**Scope.** Three new arms, 4 seeds each (42–45); everything else identical to Block A —
same pooled splits, seeds, OOF-artifact contract, and evaluation protocol addendum:

| id prefix | config | question |
|---|---|---|
| `tfgn-w3-gelstm-frozen-pooled` | `adapter: gelstm`, `encoder_init: pretrained_frozen`, `min_visits: 2`, `max_visits: 3` | spatial-first matched-window reference |
| `tfgn-w3-gelstm-random-pooled` | `adapter: gelstm`, `encoder_init: random`, `min_visits: 2`, `max_visits: 3` | matched floor |
| `tfgn-w3-winner-pooled` | `adapter: tfgn`, winning ladder config, `min_visits: 2`, `max_visits: 3` | our best model under the competitor's input constraint |

BrainTokenGT is NOT re-run: S0d already runs at exactly this window by construction. Its
Block A numbers are reused verbatim, with the reproducibility caveat attached. This reuse
was checked, not assumed: `BRAINTOKENGT/adapter.py:184-186` filters `n_scans >= min_visits`
*before* calling `window_item(...)` to truncate to `max_visits` — the same
filter-then-truncate order every other adapter here uses
(`model/GELSTM/dataset.py:74-76, 187-191`) — so S0d already sees the same 248 CV / 64 test
subjects as the rest of Block A. No code changes are required for this block:
`max_visits` is already a forwarded hyperparam on the GELSTM and TFGN adapters
(`adapters/gelstm.py:69`, `adapters/tfgn.py:108`).

**Timing.** Registered now. The two GELSTM arms may dispatch any time after P1 completes
(they need the pooled GAAE checkpoint). The TFGN arm's `hyperparams` can only be written
after the ladder freezes, since they inherit the winning config — record that
finalisation as an addendum to `DOCS/temporal-first-ablation.md`, per its own rule.

**Reporting.** Two tables:
- Table A (matched short window, T∈[2,3]): BrainTokenGT (S0d) vs GELSTM-frozen/random
  (w3) vs TFGN-winner (w3) — the strict head-to-head.
- Table B (full trajectory, T≥2): the ladder as registered — quantifies what the
  recurrent models gain from visits 4–10.

Table B carries the contribution claim; Table A is the constrained-input head-to-head
required for a fair SOTA comparison and must not be read as the main result — a windowing
handicap that throws away the visits 4-10 information the LSTM architectures exist to
exploit cannot double as evidence for or against the thesis.

**Discipline.** Matched-window arms are evaluated on the CV pool (OOF) only and never
feed ladder keep/drop decisions. If a test-set number is wanted for the matched-window
winner, it joins the single frozen estimation pass (one read, same threshold
discipline) — never a separate peek.

**Sanity check.** Truncation must not change the subject pool: assert the w3 arms see the
same 248 CV / 64 test subjects as Block A (`min_visits=2` is unchanged, so no subject is
dropped — only visits are). Report the per-cohort OOF columns as usual: truncation
removes more follow-up from ADNI than from DELCODE, and the per-cohort rows are where
that asymmetry would surface.

---

## Phase 5 — Block B (written now, run only if Block A clears the rule)

Gate: **S1c–S3 must show a cumulative in-domain gain exceeding the SE of the seed-level
differences.** If the flip cannot clear the noise floor at 64k parameters, a 300k model
is not the answer and the honest thesis result is that signal quality and sample size,
not capacity, are the bottleneck — write that instead of scaling.

If the gate passes, `CLASSIFIER/experiments/temporal_first_scaled.yaml`, 6 arms × 4 seeds:
masked-FC modelling (15–30 % of node rows re-masked each epoch) added to P2; temporal
contrastive NT-Xent over augmented views; multi-step forecasting; 2-layer bidirectional
node-LSTM at d=64 with attention pooling over all T hidden states and a learned 16-d
region embedding; 3-layer GATv2 GVAE with 8 heads + residual + Jumping Knowledge +
DropEdge; long cosine schedule with EMA, manifold mixup on the fused `u_i`, label
smoothing. The supervised head stays small and the supervised phase stays short — the
capacity lives in the pretrainable half where long training is real learning.

---

## Verification

Run at each stage, not only at the end:

1. **Phase 1** — `python -m DATA.manifest.build_pooled_assets --cohort adni_delcode` and
   read its report: expect CV pool **248** (ADNI 153 + DELCODE 95), in-domain test **64**
   (ADNI 39 + DELCODE 25), external OASIS-3 **60**; with `--min-visits 3`, ~140 / ~37 / ~24.
   Then `pytest DATA/manifest/tests/ CLASSIFIER/tests/test_pooled_data.py -q`.
2. **Phase 1 leakage** — `python run_experiment.py --id sanity-split-hygiene` must still
   pass, and the notebook's `run_full_audit` + external-disjointness assertion must fire
   during the first pooled run.
3. **Phase 0.1 correctness fixes** — `pytest CLASSIFIER/tests/test_tfgn.py -q` covers all
   three: change-mask density in range, `mean(d̃) ≈ ρ`, finite non-negative centrality on
   signed input. These are the tests that would have caught the ΔA/BCE type error.
4. **Phase 3 smoke** — `python run_experiment.py --dry-run --id tfgn-s1-flip-pooled-seed42`
   then one foreground execution with `epochs: 2` to confirm the notebook completes end to
   end and writes `ext_oasis3_auc` and `cohort_probe_auc` into `run_summary.json`.
5. **Determinism** — run `tfgn-s1-flip-pooled-seed42` twice and diff the two
   `run_summary.json` metric blocks. They must be identical; if not, the stopping rule is
   measuring GPU noise and Phase 0.2 needs fixing before the ladder proceeds.
6. **Stage 0 sanity** — `tfgn-s0-gelstm-frozen-pooled` should land near the existing
   DELCODE-only GELSTM range (0.76–0.88 in-domain). A wildly different number means the
   pooled plumbing changed something it should not have.
7. **Before hand-off** — `python scripts/run_checks.py` from the repo root, once, after
   all of Phases 0–4 are implemented (`.claude/rules/ci.md`). Verify `CHECKS.json` against
   HEAD before trusting its "NEW" report (`[[feedback_checks_json_staleness]]`).
8. **Phase 4.5** — `pytest CLASSIFIER/tests/test_oof.py CLASSIFIER/tests/test_crossval.py
   CLASSIFIER/tests/test_frozen_read.py CLASSIFIER/tests/test_logreg_drift.py -q`. Then a
   foreground `epochs: 2` run of any ladder arm: confirm `oof_predictions.csv` exists,
   `run_summary["oof"]` is populated, and — with `defer_test_eval: true` — **no** `test_*`
   / `ext_*` keys are written. Re-run each of the 24 completed arms under the new
   contract before launching S1c; diff `tfgn-s1-flip-pooled-seed42`'s re-run `cv_results`
   against the archived one (must match — confirms the re-run is a pure artifact upgrade,
   per Phase 0.2's determinism check already having verified this once).
9. **Matched-window addendum** — `python run_experiment.py --dry-run --id
   tfgn-w3-gelstm-frozen-pooled-seed42` and the `-random-` sibling resolve with no config
   errors (no new code path: `max_visits` is already plumbed through `adapters/gelstm.py`
   and `model/GELSTM/dataset.py`). On the first real w3 run, confirm the notebook's
   `run_full_audit` reports the same 248 CV / 64 test subject counts as Block A — truncation
   must drop only visits, never subjects.

## Files

**New:** `DATA/manifest/build_pooled_assets.py` (+ test) · `CLASSIFIER/common/pooled_data.py` ·
`CLASSIFIER/model/TFGN/{__init__,dataset,layers,models,losses,train}.py` ·
`CLASSIFIER/configs/tfgn.py` + `configs/tfgn_pooled.json` ·
`CLASSIFIER/adapters/{tfgn,logreg_drift}.py` ·
`CLASSIFIER/experiments/temporal_first.yaml` ·
`CLASSIFIER/notebooks/LONGITUDINAL/LONGITUDINAL_TFGN_SSL_POOLED.ipynb` ·
`CLASSIFIER/notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb` ·
`CLASSIFIER/tests/{test_tfgn,test_pooled_data,test_logreg_drift}.py` ·
`DOCS/temporal-first-ablation.md` ·
`CLASSIFIER/common/{oof,frozen_read}.py` (+ `tests/{test_oof,test_frozen_read}.py`, Phase 4.5)

**Modified:** `SHARED/seeding.py` (opt-in strict determinism) ·
`CLASSIFIER/adapters/__init__.py` (two registry lines) ·
`CLASSIFIER/adapters/gelstm.py` (`prepare_data` cohort dispatch) ·
`CLASSIFIER/common/run_artifacts.py` (`record_external_metrics`, Phase 4.5's
`record_oof_artifacts`) ·
`CLASSIFIER/common/crossval.py` (Phase 4.5's `fold_probe` / `oof_folds` / `oof_extras`) ·
`CLASSIFIER/common/experiment_utils.py` (Phase 4.5's `oof.*` columns in `RESULTS.csv`) ·
`CLASSIFIER/adapters/logreg_drift.py` (Phase 4.5's `feature_set` knob + `load_state` fix) ·
`CLASSIFIER/notebooks/LONGITUDINAL/LONGITUDINAL_COMMON_DELCODE.ipynb` (pooled branch,
external-test cell, cohort probe, Phase 4.5's OOF-artifact cell + `defer_test_eval`
guards — all additive and guarded) ·
`CLASSIFIER/experiments/temporal_first.yaml` (matched-window addendum's 8
`tfgn-w3-gelstm-{frozen,random}-pooled-seed{42..45}` entries + a commented
`tfgn-w3-winner-pooled` placeholder pending the ladder freeze — no Python changes, the
`max_visits` knob is already wired end to end).

**Reused, not rewritten:** `LongitudinalSubjectDataset`, `common/crossval.run_kfold_cv`,
`common/thresholds.select_oof_threshold`, `common/comparison.{paired_delong_test,
paired_bootstrap_ci}`, `common/visits.visit_identity`, `common/early_detection`,
`common/run_artifacts.save_run`, `configs/encoder.py`'s four-arm enum,
`model/VGAE/losses.kl_divergence`, `model/classification/logreg_cv.train_logreg_cv`.
