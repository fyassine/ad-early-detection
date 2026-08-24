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
| **S1** | `tfgn-s1-flip-pooled` | `node_lstm_init: random`, gate off, `recon_target: none`, `fusion: z_only`, `readout: mean` | **Flip alone, no self-supervision anywhere → compare to S0c. Primary arm, carried forward (see "Ladder state and corrected order" below).** |
| **S1b** | `tfgn-s1b-ssl-pooled` | `node_lstm_init: pretrained_finetuned` (P2) | Does node-LSTM SSL forecasting help? **Dropped by Tier 2's one-SE tie-breaker — sensitivity arm, not primary.** |
| **S1c** | `tfgn-s1c-recon-pooled` (original, invalid) / `tfgn-s1c-recon-random-pooled` (re-run) | `recon_target: delta_a_topk`, `lambda_recon`, `beta_kl` + free bits + warmup; `node_lstm_init: random` in the re-run | **Headline arm: both encoders self-supervised → compare to S0b.** The original run inherited `pretrained_finetuned` from the since-reversed S1b fork and is recorded as undecidable; the re-run is protocol-valid. |
| **S2** | `tfgn-s2-gate-pooled` | `use_gate: true`, `lambda_sparse`, `lambda_drift = 0.1·λ_sparse`, `gate_rho: 0.15` | Does suppressing static regions help? Branches from **S1**. |
| **S3** | `tfgn-s3-fusion-pooled` | `fusion: concat_residual` | Does preserving unsmoothed H help? **VOID — the knob is inert under `recon_target: none`; the run reproduced S1 bit-for-bit. See "Ladder complete" below.** |
| **S4** | `tfgn-s4-attnpool-pooled` | `readout: attention` | Does attentive pooling beat mean pooling? |
| **S5** | `tfgn-s5-dualscore-pooled` | `dual_score: true`, `lambda_cent` | Interpretability at zero risk to the backbone. **Kept regardless of AUC by pre-registration — the stopping rule does not apply to it.** |
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
after the S1c re-run reports. **S1b was the fork**: per the correcting addendum in
`DOCS/temporal-first-ablation.md`, Tier 2's one-SE tie-breaker resolves it to
`node_lstm_init: random` (S1's own config), which is what S1c–S5 inherit — see "Ladder
state and corrected order (2026-08-24)" below for the full derivation.

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
# S0-S1b complete as of 2026-08-24 — see "Ladder state and corrected order" below
# for the verified results and the corrected continuation:
# tfgn-s1c-recon-random-pooled -> S2 -> S3 -> S4 -> S5 -> SENS
scripts/dispatch.sh --pkg CLASSIFIER --id tfgn-s1c-recon-random-pooled-seed{42..45}
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

**Outcome of the re-verified fork decision (2026-08-24).** Reading the re-run OOF
artifacts is what caught the winner's-curse selection this section's re-verification was
meant to prevent: the fold-matched Tier-2 statistic passes for S1b (ratio 6.46) while the
pooled per-seed statistic fails (ratio 0.19), and Tier 2's own one-SE tie-breaker
resolves the disagreement to **S1**, not S1b. See "Ladder state and corrected order
(2026-08-24)" below for the full numbers; `DOCS/temporal-first-ablation.md`'s "S1b fork
decision" correcting addendum is the pre-registration record.

---

## Ladder state and corrected order (2026-08-24)

Three protocol violations were caught in the scorecard read after S0a–S1c (32 runs)
completed, all traced back to `oof_predictions.csv` numbers, never the scorecard's own
summary. Full derivation and the corrected pre-registration text are in
`DOCS/temporal-first-ablation.md`'s "S1b fork decision" correcting addendum, "S1c
(2026-08-24 run) — recorded as undecidable", and "S1c re-run (random init)" sections —
this section carries the state forward into the execution plan.

**Verified ladder state** (pooled OOF AUC, mean ± SD over seeds 42–45, from
`run_summary.json["oof"]`):

| arm | OOF AUC | note |
|---|---|---|
| S0-demo floor | 0.5296 | Tier-1 demographics floor |
| S0c gelstm-random | 0.5625 | matched floor for S1 |
| S0d BrainTokenGT | 0.6207 | **below the linear ΔA baseline** — citable for the matched-window framing, BTGT non-reproducibility caveat attached |
| S0a logreg-drift | 0.7053 | linear floor |
| S0b gelstm-frozen | 0.7186 | spatial-first, pretrained |
| **S1 flip** | **0.7488 ± 0.0028** | **primary arm.** Tightest seed SD of any deep model in the table — cite in the stability section next to the strict-determinism work. Static N=1 row: **0.4919** (chance) vs full-trajectory 0.7488 — "all signal is longitudinal," worth a figure. |
| S1b ssl | 0.7502 ± 0.0125 | +0.0014 pooled (SE 0.0074, ratio 0.19) / +0.0120 fold-matched (SE 0.0019, ratio 6.46) vs S1 — the two Tier-2 statistics disagree; see the doc's Tier-2 clarification. Tier 2's one-SE tie-breaker selects S1 (simpler, within one SE either way). **Sensitivity arm, not primary.** |
| S1c recon (original run) | 0.5507 | built on `pretrained_finetuned` (inherited the now-reversed fork) — **protocol-invalid, recorded as undecidable, not a loss for the flip.** S0b↔S1c is unanswered pending the re-run below. |

**Three fixes baked into the plan:**

1. **S1, not S1b, is the primary arm.** S1b is retained only as a documented secondary
   sensitivity read at Tier 4.
2. **S0b↔S1c is undecidable, not a flip loss.** The original S1c run tested an
   unregistered configuration (SSL node-LSTM init on a since-dropped fork + reconstruction
   loss together). A protocol-valid re-run, `tfgn-s1c-recon-random-pooled-seed{42..45}`
   (`node_lstm_init: random`, everything else identical to the original S1c entries), is
   registered in `CLASSIFIER/experiments/temporal_first.yaml` and answers the contrast.
   The original run's artifacts are left untouched as the record of the invalid
   configuration (`.claude/rules/gpu-dispatch.md` — never repoint an existing id).
3. **The ladder is not finished.** S2 (gate) branches from **S1**, then S3, S4, S5, and
   SENS remain unrun. S2 and S5 are where the gate maps and the quadrant scatter live —
   the interpretability contribution and the main MICCAI differentiator. Frozen reads stay
   Tier 4, after SENS reports, per `DOCS/temporal-first-ablation.md`'s restated Tier-4
   gate.

**Corrected order, superseding Phase 4's original run-order list** (this block is itself
superseded by "Ladder complete — verified scorecard and corrected verdicts" below, now
that every rung has reported)**:**

```
tfgn-s1c-recon-random-pooled (S1c re-run, branches from S1)
  -> S2 (gate, branches from S1)
  -> S3 (fusion)
  -> S4 (attention pool)
  -> S5 (dual-score / interpretability)
  -> SENS (min_visits=3)
  -> comparison notebook (Tier 1-3 read between every block above)
  -> Tier-4 frozen reads: S1 lineage primary, S1b secondary if taken
  -> matched-window w3 arms
```

**Loss-component diagnostic** (supports, does not replace, the "recorded as
undecidable" framing): `model/TFGN/train.py::train_epoch` gains additive per-term loss
logging (bce/recon/kl/gate/drift/cent), wired through `adapters/tfgn.py`'s existing
`epoch_log_fn`. One short non-ladder run, `tfgn-s1c-diag-loss-components`, checks whether
the reconstruction/KL terms dominate `bce` from epoch 0 — an observation recorded in
`DOCS/temporal-first-ablation.md`, never a Tier-2 input and never grounds for a lambda
change without its own documented deviation (a λ sweep is explicitly out of scope as a
quiet re-run).

**All GPU runs in this section — the S1c re-run block and the loss-component diagnostic
— are held for explicit user approval before dispatch.** Everything else (docs, registry
entries, code instrumentation, tests, notebook cells) proceeds without waiting.

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

## Ladder complete — verified scorecard and corrected verdicts (2026-08-24, batch 5)

S2–S5, SENS and the three W3 matched-window arms have all reported. Every number below
was recomputed from `oof_predictions.csv` (per-subject, per-fold), not from any run's own
summary line. This section supersedes the "Corrected order" block above; the Tier-4
frozen read is now the only unfinished step, and four things must be fixed first.

### Verified scorecard (pooled CV OOF, mean ± SD over seeds 42–45)

| arm | N | pooled OOF AUC | ADNI | DELCODE | bal. acc | static N=1 |
|---|---|---|---|---|---|---|
| S0-demo (age+sex) | 248 | 0.5296 ± 0.0000 | 0.5162 | 0.5186 | 0.5139 | 0.5296 |
| S0c gelstm-random | 248 | 0.5625 ± 0.0292 | 0.5348 | 0.6067 | 0.5631 | 0.5140 |
| S0d BrainTokenGT | 248 | 0.6207 ± 0.0338 | 0.6194 | 0.6217 | 0.5889 | 0.5353 |
| S0a logreg-drift | 248 | 0.7053 ± 0.0000 | 0.5564 | 0.9129 | 0.6776 | 0.5653 |
| S0b gelstm-frozen | 248 | 0.7186 ± 0.0334 | 0.4971 | 0.9178 | 0.6505 | 0.4699 |
| **S1 flip (winner)** | 248 | **0.7488 ± 0.0033** | 0.6526 | 0.8741 | 0.7093 | 0.4919 |
| S1b ssl (sensitivity) | 248 | 0.7502 ± 0.0125 | 0.6624 | 0.8707 | 0.7037 | 0.5072 |
| S1c recon-random | 248 | 0.5433 ± 0.0311 | 0.5335 | 0.5589 | 0.5477 | 0.5213 |
| S2 gate | 248 | 0.7308 ± 0.0160 | 0.6266 | 0.8700 | 0.6811 | 0.4969 |
| S3 fusion | 248 | 0.7488 ± 0.0033 | 0.6526 | 0.8741 | 0.7093 | 0.4919 |
| S4 attn-pool | 248 | 0.6490 ± 0.0144 | 0.5415 | 0.8022 | 0.6111 | 0.5197 |
| S5 dual-score | 248 | 0.7331 ± 0.0173 | 0.6460 | 0.8530 | 0.6885 | 0.5010 |
| SENS (min_visits=3) | 140 | 0.7413 ± 0.0137 | 0.6328 | 0.8986 | 0.6907 | 0.5469 |
| W3 gelstm-random | 248 | 0.5885 ± 0.0304 | 0.5564 | 0.6385 | 0.5541 | 0.5190 |
| W3 gelstm-frozen | 248 | 0.7500 ± 0.0138 | 0.5848 | 0.9075 | 0.7000 | 0.5013 |
| W3 TFGN-winner | 248 | 0.7318 ± 0.0348 | 0.6584 | 0.8342 | 0.6756 | 0.4880 |

Report **SD** across seeds, as above. The batch-5 scorecard mixed SD and SE in the same
column (S1 as "± 0.0028" is neither) — one convention, stated in the caption.

### Tier-2 statistics, every rung against S1

| contrast | fold-matched Δ ± SE (ratio) | pooled Δ ± SE (ratio) | verdict |
|---|---|---|---|
| S1b vs S1 | +0.0120 ± 0.0019 (+6.46) | +0.0014 ± 0.0074 (+0.19) | disagree → one-SE tie-breaker → **S1** |
| S1c-random vs S1 | −0.1587 ± 0.0099 (−15.98) | −0.2055 ± 0.0150 (−13.72) | dropped |
| S2 gate vs S1 | −0.0064 ± 0.0075 (−0.85) | −0.0180 ± 0.0086 (−2.08) | dropped |
| S3 fusion vs S1 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | **VOID — not a result** |
| S4 attn-pool vs S1 | −0.0558 ± 0.0058 (−9.60) | −0.0999 ± 0.0083 (−11.96) | dropped |
| S5 dual-score vs S1 | −0.0046 ± 0.0067 (−0.68) | −0.0158 ± 0.0101 (−1.56) | **kept — see A below** |

### A. S5 is kept, not rejected

`DOCS/temporal-first-ablation.md:105` pre-registers S5 as *"Interpretability, zero risk
to the backbone (**kept regardless of AUC**)"*. The keep/drop stopping rule does not
apply to it, and applying it retroactively is exactly the post-hoc rule change Phase 0
exists to prevent. The correct record, to be written into both the doc and the
comparison notebook:

> S5 is **classification-neutral**: fold-matched Δ = −0.0046 ± 0.0067, i.e. |Δ| < SE —
> indistinguishable from zero, which is what "zero risk to the backbone" was
> pre-registered to mean. It is **kept as the interpretability layer**, and the §0.1d
> validation runs on its outputs regardless of the AUC delta.

**Frozen-arm decision, fixed here before `RUN_FROZEN_READ` is flipped:** the Tier-4
frozen arm is **S1** for the headline classification number, with **S5's artifacts
analysed on OOF only**. Rationale: S1 is the arm the ladder selected under the
pre-registered rule; S5's Δ is neutral but negative in both statistics, and freezing a
strictly-worse-in-expectation arm to get artifacts that are already available OOF buys
nothing. S5 is read at Tier 4 as a *secondary* arm in the same single pass (alongside
S1b), never substituted for the primary. Set `FROZEN_WINNER_ID =
'tfgn-s1-flip-pooled'`, `SECONDARY_SENSITIVITY_ID = 'tfgn-s1b-ssl-pooled'`.

### B. S3 is void, not rejected — the knob never executed

`tfgn-s3-fusion-pooled` sets `fusion: concat_residual` **with `recon_target: none`**.
Under `recon_target: none` the model builds no GVAE (`model/TFGN/models.py:95-105`),
`self.fusion_module = None`, and the forward pass takes `h_fused = h_T` regardless of
`self.fusion` (`models.py:175-183`). The knob is dead code on that branch. Verified, not
inferred: S3's `oof_predictions.csv` is **bit-identical to S1's** on all four seeds
(max |Δprob| = 0.0e+00), and every OOF metric matches to full precision.

So S3's row in the batch-5 scorecard is S1's number relabelled. Record it as
**"not testable at this rung — the fusion knob is inert without a latent `z`; requires a
`recon_target ≠ none` parent, which S1c-random ruled out"**, never as "ran and failed the
keep rule". The same fact is the cosmetic annotation the winner's config needs: the
winner's `fusion: z_only` is likewise a no-op under `recon_target: none` — annotate the
config string so no reader concludes the winning model uses a GVAE latent. It does not;
the winning TFGN is node-LSTM → mean-pool → linear head, with no graph propagation stage
at all. That is a substantive finding about the architecture and belongs in the results
text, not a footnote.

**Do not re-run S3 under a recon parent.** S1c-random (`recon_target: delta_a_topk`,
0.5433) collapses ~0.20 AUC below S1; any S3 branching from it inherits that collapse and
answers nothing. The honest record is that the fusion question is unanswerable within
this ladder, because its prerequisite rung failed.

### C. The quadrant map's temporal axis — decision

S2's rejection leaves the winner with no learned gate, so S5's `dual_scores.npy` supplies
`s_topo` but no `s_temp`. **Decision: use the model-free rank-sigmoid drift anchor `d̃`
as the temporal axis** — one learned axis (S5's `s_topo`), one measured axis (`d̃`).
Reasons it is the better of the two options: it is pre-registered already (§0.1c), it is
computable offline with **zero GPU cost** from
`model/TFGN/dataset.py::compute_drift_anchor` (a pure function of `X`, no checkpoint
needed), and it does not require importing a map from a rejected arm and then arguing the
rejection was "about AUC, not map validity". S2's `gate_scores.npy` is reported
**alongside** it as a supporting panel, with the rejection stated explicitly — not as the
primary axis.

Run the §0.1d validation (permutation null over 1 000 label permutations with the
DMN/hippocampal overlap percentile, cross-map Spearman stability, and both split per
cohort) on the (`s_topo`, `d̃`) pair.

**Artifact limitation, to be recorded as a documented deviation:** `adapters/tfgn.py`'s
`extra_artifacts` persists only the **best fold's** maps — `dual_scores.npy` and
`gate_scores.npy` are `(50, 200)`, one fold's validation subjects, per seed. The
pre-registered "cross-fold Spearman across 5 folds × 4 seeds" therefore cannot be
computed from existing artifacts. Two options, in preference order:

1. **Report cross-*seed* Spearman over the 4 best-fold maps** and record the reduction as
   a deviation in `DOCS/temporal-first-ablation.md`. Zero GPU cost. Recommended — the
   stability claim survives in weakened form and the deviation is stated.
2. Add a `fold_probe` that persists per-fold maps and re-run S5 (4 runs, ~2 h). Only
   worth it if a reviewer challenges option 1.

### D. Housekeeping — verified

- **S0b checkpoint provenance: confirmed correct.** All four S0b seeds (and all four W3
  gelstm-frozen seeds) record `gaae_run_name =
  dark-surf-2-gaae-pretrain-pooled-adni-delcode-2026-08-24_08-23-22_2026-08-24_10-43-31`
  — the pooled checkpoint, not `ethereal-planet-16`. The S0b↔S1c contrast rests on a
  valid pointer. No action.
- **P1's "failed" status is cosmetic** (full 500 epochs, val loss 0.018795, checkpoint
  saved; papermill died on a plotting cell). Confirmed by the provenance check above.

### E. SENS reads better than "direction-only" — but not in the direction claimed

SENS's 140 subjects are a strict subset of S1's 248. Restricting **S1's own OOF
predictions** to those same 140 subjects gives, per seed:

| seed | S1 restricted to the ≥3-visit subjects | SENS (trained on ≥3-visit only) |
|---|---|---|
| 42 | 0.7762 | 0.7385 |
| 43 | 0.7656 | 0.7499 |
| 44 | 0.7603 | 0.7230 |
| 45 | 0.7825 | 0.7535 |
| mean | **0.7712** | **0.7412** |

Two separable statements, and the batch-5 write-up conflated them:

1. **The ≥3-visit subgroup is easier** — S1 scores 0.7712 on them vs 0.7488 on the full
   pool. This *is* the pre-registered "does the advantage grow with sequence length"
   signal, and it is positive.
2. **Training only on ≥3-visit subjects is worse** — 0.7412 vs 0.7712 on identical
   subjects. Shrinking the pool 248 → 140 costs more than the longer sequences gain.

Report both, with (1) as the sequence-length evidence and (2) as a sample-size result.
Keep the pre-registered "too small to decide on its own" language on both. Do **not**
report SENS's 0.7413 next to S1's 0.7488 as a like-for-like row — different N, different
subjects, not fold-matched.

### F. The matched-window head-to-head — TFGN loses to spatial-first

This is the batch-5 result most in need of stating plainly, and the write-up omitted it:

| arm (T ∈ [2,3]) | pooled OOF AUC |
|---|---|
| BrainTokenGT (S0d) | 0.6207 ± 0.0338 |
| W3 GELSTM-random | 0.5885 ± 0.0304 |
| **W3 GELSTM-frozen** | **0.7500 ± 0.0138** |
| W3 TFGN-winner | 0.7318 ± 0.0348 |

Fold-matched: **W3-TFGN vs W3-GELSTM-frozen = −0.0244 ± 0.0027 (ratio −8.94)** — a
consistent loss across all four seeds, well outside noise. W3-TFGN vs BrainTokenGT =
+0.0999 ± 0.0162 (+6.17) — a clean win over the SOTA competitor.

The honest framing, which Table A must carry: **under the competitor's short-window
constraint the temporal-first flip loses its advantage over spatial-first.** TFGN's win
in Table B (0.7488 vs 0.7186, full trajectory) comes from the visits the window throws
away — which is precisely the claim Table B was built to make, and this is the
confirmation of it, not a contradiction. State it as such: the flip's gain is a
*long-sequence* gain. Never present Table A's TFGN row without its GELSTM-frozen
neighbour.

### G. Tier-4 frozen reads — state, and the blocker that must be fixed first

**Nothing has been read.** Verified across all 76 ladder runs: every `run_summary.json`
carries `defer_test_eval: true` and **zero** `test_*` and **zero** `ext_*` keys. The
in-domain test set (n=64) and OASIS-3 (n=60) are both completely unspent. The deferral
discipline held.

**Both target splits are present and fully resolvable:**

- In-domain test: `DATA/POOLED_ADNI_DELCODE/SPLITS/downstream/test.csv` — **64** subjects
  (ADNI 39 + DELCODE 25), 24 converters / 40 non-converters. Matches the pre-registered count.
- External OASIS-3: `DATA/OASIS3/__metadata__/SPLITS/downstream/{train,val,test}.csv`
  concatenated — **60** subjects (35+12+13), 31 converters / 29 non-converters. All 60
  resolve ≥2 FC files under `min_visits=2` (T=2: 36, T=3: 13, T=4: 9, T=6: 2), so none is
  dropped. `COHORT_ROOTS['oasis3']` is wired in `common/pooled_data.py`, and the notebook
  tags the frame `cohort='oasis3'`, so the multi-cohort dispatch handles it.

`pytest CLASSIFIER/tests/test_frozen_read.py -q` — 6 passed.

**STATUS 2026-08-24: FIXED AND VALIDATED.** All 12 target checkpoints (S1, S1b, S5 ×
4 seeds) now carry the statistics, every one validated against its own recorded
predictions. Details under "Resolution" at the end of this section. The original
diagnosis is kept below because it is the record of what went wrong and why.

**BLOCKER — the frozen read will crash on the first seed as things stand.**
`TFGNAdapter.load_state` (`adapters/tfgn.py:558-565`) rebuilds the eval state by reading
`log_dt_scaler_mean`, `log_dt_scaler_scale`, `cent_mean`, `cent_std` from the checkpoint.
`model_state_for_save` (`adapters/tfgn.py:537-538`) returns **only** `state["model_state"]`,
so those four keys are dropped at checkpoint-write time and are absent from every saved
TFGN checkpoint — confirmed by inspecting S1 and S5 seed-42 checkpoints (top-level keys
are `model_config`, `training_config`, `optimizer_state_dict`, `scheduler_state_dict`,
`rng_state`, `torch_rng_state`, `env`, `git`, `val_auc`, `best_threshold`,
`threshold_method`, `best_fold`, `gaae_checkpoint`, `run_name` — none of the four).
`_apply_state_normalization` (`adapters/tfgn.py:351-354`) then does a bare
`state["log_dt_scaler_mean"]` lookup.

This fails **loudly** with `KeyError`, not silently with unnormalised features — the
`.claude/rules/errors.md` discipline is what makes this a two-hour fix instead of a
silently wrong headline number. It was never exercised because no TFGN run has ever been
reloaded: `defer_test_eval: true` meant nothing ever called `load_state`. Same class of
latent bug as the `LogRegDriftAdapter.load_state` unwrapping fix in §4.5.4.

**Fix — CPU-only, no retraining, no GPU.** The winning fold's statistics are exactly
recoverable, because the CV split is deterministic and recorded:

- `StratifiedGroupKFold` in `common/crossval.run_kfold_cv` takes no seed and no shuffle,
  so fold *i* is the same subject group for every seed and every arm. Verified: the
  `subject_id → fold` map in `oof_predictions.csv` is byte-identical across
  `tfgn-s1-flip-pooled-seed{42,45}`, `tfgn-s1b-ssl-pooled-seed44` and
  `tfgn-s5-dualscore-pooled-seed43`. Folds are {1:50, 2:50, 3:50, 4:49, 5:49}.
- `best_fold` is in every checkpoint (= 1 for all four S1 seeds — consistent with the
  `(50, 200)` artifact shape in §C).

So the winning fold's train set is exactly the `oof_predictions.csv` rows with
`fold != best_fold`, and `log_dt_scaler` / `cent_mean` / `cent_std` can be recomputed from
those subjects by the same code that produced them (`adapters/tfgn.py:225-234, 321-324`).

Steps:

1. Extend `TFGNAdapter.model_state_for_save` to persist the four keys alongside
   `model_state` (matching what `load_state` already expects), so future runs never hit
   this. Add a test that round-trips `model_state_for_save` → `load_state` and asserts all
   four survive — the test that would have caught this.
2. Write a one-off CPU backfill that, for each of the 12 arms to be frozen-read
   (S1, S1b, S5 × 4 seeds), rebuilds the `best_fold` train items and patches the four
   statistics into the existing checkpoint. **Never repoint or re-run the id**
   (`.claude/rules/gpu-dispatch.md`) — this patches the artifact in place, additively.
3. **Validate the backfill before spending the read.** With the recomputed statistics,
   re-score each run's own held-out `best_fold` and confirm it reproduces the checkpoint's
   stored `val_auc` (S1: 0.8125 / 0.8125 / 0.8090 / 0.7917 for seeds 42–45) and the
   matching `prob` column of `oof_predictions.csv`. If it reproduces, the statistics are
   provably the originals and the frozen read is sound. If it does not, stop — do not
   flip `RUN_FROZEN_READ`, and fall back to re-running the 12 arms.

**Second defect, lower priority — the notebook's `adapter_key` map is incomplete.**
`run_frozen_reads` maps `model_type` → adapter key with
`{'tfgnclassifier': 'tfgn', 'logregdriftadapter': 'logregdrift'}`. GELSTM arms report
`model_type = 'GELSTMClassifier'` → `'gelstmclassifier'`, which is not a registered
adapter key. Harmless for the planned pass (S1/S1b/S5 are all TFGN), but §F notes the
matched-window winner is **W3-GELSTM-frozen** — if a test number is ever wanted for it,
this bites. Add `'gelstmclassifier': 'gelstm'` (and `'braintokengt*'`) while fixing the
above.

**Third defect — the one-shot read has no idempotency guard.**
`score_frozen_split` → `record_test_metrics` (`common/run_artifacts.py:116-141`) patches
`run_summary.json` unconditionally. Re-executing the notebook with `RUN_FROZEN_READ=True`
would silently read the test set a second time and overwrite the first result — the exact
failure the whole Tier-4 protocol exists to prevent, guarded today only by a hand-set
boolean. Add a guard to `score_frozen_split`: if the target `test_*` / `ext_*` keys already
exist in `run_summary.json`, raise unless an explicit `allow_overwrite=True` is passed.
This is the cheapest possible insurance on the single most expensive-to-lose asset in the
project.

**Resolution (2026-08-24) — implemented, validated, no GPU used.**

*Code (all additive; single-cohort and non-TFGN behaviour unchanged):*

- `adapters/__init__.py` — new base hook `LongitudinalAdapter.checkpoint_extras(state)
  -> {}`. Declares non-weight state that must ride inside the full-state checkpoint,
  which is exactly where `load_state` reads it back from.
- `adapters/tfgn.py` — `STATE_NORMALIZATION_KEYS` is now the single source of truth for
  the four statistics; `checkpoint_extras` returns them (raising if `train_fold` omitted
  one), and `load_state` **raises `KeyError` naming the backfill script** instead of
  silently skipping a missing key and failing later inside
  `_apply_state_normalization` with an opaque message.
- `common/run_artifacts.py::save_run` — new `checkpoint_extras=` parameter merged into
  `save_full_checkpoint`'s top level.
- `LONGITUDINAL_COMMON_DELCODE.ipynb` cell 23 — passes
  `checkpoint_extras=adapter.checkpoint_extras(BEST_MODEL_STATE)`, so every future run
  is correct by construction.
- `common/frozen_read.py::score_frozen_split` — new `allow_overwrite=False`: refuses to
  score a split whose `test_*` / `ext_*` keys already exist. The one-shot read can no
  longer be spent twice by re-executing a cell.
- `COMPARISON_TEMPORAL_FIRST_LADDER.ipynb` — `adapter_key` map extended with
  `gelstmclassifier` and `braintokengtclassifier` (both occurrences).
- `tests/test_adapter_checkpoint_roundtrip.py` (new, 5 tests) — asserts
  `checkpoint_extras`'s key set equals what `load_state` demands (the assertion that
  would have caught this), round-trips through a real checkpoint file, and pins both
  loud-failure paths. `pytest tests/test_adapter_checkpoint_roundtrip.py
  tests/test_frozen_read.py -q` → **11 passed**.

*Backfill (`CLASSIFIER/scripts/backfill_tfgn_norm_stats.py`, new):* recovers the winning
fold's training subjects from `best_fold` + `oof_predictions.csv`, rebuilds the adapter
through `frozen_read.build_adapter_from_run` (the same path the frozen read itself uses,
so the refit cannot diverge from its consumer), refits the scaler and centrality
statistics with the identical code `train_fold` uses, and patches them into the
checkpoint. Each original checkpoint is copied to `*.pth.pre-backfill` first; the script
refuses to touch a checkpoint that already carries the keys.

*Validation — the criterion, and why it is the right one.* `--validate` re-scores each
run's own held-out winning fold with the recomputed statistics and compares against the
predictions that run originally wrote. Two conditions, both required:

1. `|ΔAUC| ≤ 1e-9` against the recorded fold AUC. This is the decision-relevant test —
   every downstream number derives from it, and mis-scaled features cannot leave it
   invariant.
2. `max|Δprob| ≤ 1e-5` per subject, confirming the agreement is pointwise rather than a
   coincidence of tied ranks. Exact equality is unavailable: the model emits float32 and
   `oof_predictions.csv` round-trips through text. Observed residuals are 4.8e-07 to
   2.2e-06 — a wrong scaler moves probabilities by ~1e-1, five orders of magnitude above
   this floor.

**Result: 12/12 PASS, every one at `|ΔAUC| = 0.00e+00` exactly.** Each run's re-scored
fold AUC reproduces both its `oof_predictions.csv` rows and the `val_auc` stored in its
checkpoint (S1: 0.812500 / 0.812500 / 0.809028 / 0.791667; S1b: 0.802083 / 0.777778 /
0.792115 / 0.782986; S5: 0.812500 / 0.810764 / 0.824653 / 0.795139). `best_fold` is 1 for
all four S1 and S5 seeds; S1b seeds 44 and 45 won on folds 5 and 2, and the backfill
handles each correctly. The statistics are provably the originals, not an approximation.
`adapter.load_state()` on a backfilled run now returns all five keys. **Tier 4 is
unblocked and no GPU time was needed.**

### H. The "no graph stage" corollary, and its capacity defense

State the winner exactly as it is: **node-shared LSTM → mean-pool → linear head.** With
`recon_target: none` no GVAE is constructed (`model/TFGN/models.py:95-105`), so the
winning TFGN contains no graph propagation stage at all. This is the honest result and
the more interesting one — but it invites the obvious reviewer question, *"is the flip's
win just capacity or bandwidth?"* Close it with the parameter counts, measured from the
runs' own configs via `_build_model()`:

| arm | total params | trainable | OOF AUC |
|---|---|---|---|
| **S1 flip (TFGN, winner)** | **68,417** | **68,417** | **0.7488** |
| S5 dual-score | 68,482 | 68,482 | 0.7331 |
| S2 gate | 72,642 | 72,642 | 0.7308 |
| S0b gelstm-frozen | 965,897 | 520,905 | 0.7186 |
| S0c gelstm-random | 965,897 | 965,897 | 0.5625 |

**Correction to the ~30k / ~240k figures:** the true counts are 68,417 and 965,897. The
ratio is therefore **14.1×** against S0c (not ~8×), and **7.6×** against S0b's trainable
parameters. Use the measured numbers — the argument is stronger than the estimate, and a
parameter count is trivially checkable by a reviewer.

The one-sentence mechanism statement this licenses:

> The winning model is 14× smaller than the spatial-first baseline it beats by 0.186 AUC
> and 7.6× smaller (trainable) than the pretrained one it beats by 0.030, so the gain
> cannot be capacity — it comes from deferring pooling until after node-level temporal
> encoding, which is precisely what the flip hypothesis claimed.

That closes the capacity critique without another run. Note the same table also kills a
"the graph stage was never given a chance" objection from the other direction: S2 and S5
*add* parameters to S1 and both score lower.

### I. The matched-window result is two-sided — state both directions

§F's single-sided "TFGN loses to GELSTM-frozen under the short window" is half the
finding. Truncation moves the two architectures in **opposite** directions:

| arm | full trajectory (T≥2) | matched window (T∈[2,3]) | Δ from truncation |
|---|---|---|---|
| GELSTM-frozen (spatial-first) | 0.7186 | 0.7500 | **+0.0314** |
| TFGN (temporal-first) | 0.7488 | 0.7318 | **−0.0170** |
| GELSTM-random | 0.5625 | 0.5885 | +0.0260 |

Spatial-first **gains** from truncation; temporal-first **pays** for it. That two-sided
statement is far stronger evidence for "the flip's value lives in visits 4–10" than the
loss alone, and it makes Table B load-bearing by construction rather than by assertion:
the only regime where temporal-first wins is the one with long sequences, and the only
regime where spatial-first wins is the one where the trajectory has been cut to a
difference. Write it as a crossover, not a defeat.

**Reconciling +0.0999 (fold-matched) with +0.1111 (raw means) — the reviewer's guessed
cause is not the actual one.** It is *not* single-class fold exclusion: checked
explicitly, all 5 folds carry both classes in both arms across all 4 seeds (fold sizes
50/50/50/49/49), and no fold is dropped from either statistic. The real cause:

- **Fold-matched Δ** averages per-fold AUCs: 0.7586 (W3-TFGN) − 0.6587 (BTGT) = +0.0999.
- **Pooled Δ** ranks all 248 subjects together: 0.7318 − 0.6207 = +0.1111.

Pooled AUC is *lower* than the mean per-fold AUC for both arms, because pooling
additionally penalises cross-fold score incomparability — each fold is a separately
trained model with its own probability scale. That penalty is **−0.0268 for W3-TFGN and
−0.0379 for BrainTokenGT**, and the 0.0112 difference between the two penalties is
exactly the gap between the two statistics.

So the reconciling line in the notebook should say: both statistics use all 5 folds × 4
seeds with no exclusions; they differ because pooled AUC also charges for cross-fold
calibration drift, and charges BrainTokenGT more. That is itself a reportable result —
TFGN's per-fold outputs are more mutually comparable, consistent with its being the
tightest-seed-SD deep model in the table (§"Verified scorecard").

### J. Sequence-length evidence — the SENS decomposition is the section, not a footnote

Promote §E from a caveat to the thesis's sequence-length result, stated as the
decomposition rather than as SENS's headline number. SENS's 140 subjects are a strict
subset of S1's 248, which is what makes the two effects separable — SENS alone cannot
separate them:

- **Subgroup difficulty (+0.022):** S1, trained on the full pool, scores **0.7712** on
  the ≥3-visit subgroup vs **0.7488** overall. Longer trajectories are more predictable.
  This is the sequence-length signal, and it is positive.
- **Training-pool cost (−0.030):** SENS, trained only on those subjects, scores
  **0.7412** on the same 140 — worse than S1 scores on them. Shrinking the pool
  248 → 140 costs more than the subgroup's easiness returns.

Net: restricting to ≥3 visits is a losing trade at this sample size, *and* longer
sequences carry more signal. Both are true and they are not in tension. Keep the
pre-registered "too small to decide on its own" language on both halves, and never
report SENS's 0.7413 beside S1's 0.7488 as a like-for-like row — different N, different
subjects, not fold-matched.

### K. Remaining bookkeeping — confirmed

- **Cross-seed-only Spearman.** Best-fold maps are `(50, 200)`, so 4 maps not 20. Log as
  a documented deviation in `DOCS/temporal-first-ablation.md`; zero cost, no re-run.
- **`SECONDARY_SENSITIVITY_ID` takes a single id.** The frozen pass needs S5 *and* S1b as
  secondaries, so either call `run_frozen_reads` twice or make the parameter a list and
  loop. Prefer the list — one pass, one guard, no chance of a half-run. Record S5's
  secondary read as **the interpretability layer's number, not a competing endpoint**;
  the label string already printed on every line (`'SECONDARY (sensitivity arm, not
  primary)'`) should be specialised for S5 to say so.
- **Block B closed.** Correct read of Phase 5's own wording. Paired with §H's parameter
  counts it is a clean conclusion: signal quality and sample size, not capacity, are the
  bottleneck — and the winner being 14× smaller than the baseline it beats is the
  evidence for that sentence, not merely consistent with it.

### L. Tier 4 executed (2026-08-24) — results, and a missed pre-registered escalation

**RUN_FROZEN_READ flipped and executed.** One papermill pass, `FROZEN_WINNER_ID =
'tfgn-s1-flip-pooled'` (PRIMARY), `SECONDARY_SENSITIVITY_ID =
['tfgn-s1b-ssl-pooled', 'tfgn-s5-dualscore-pooled']` (both SECONDARY, S5 labelled as the
interpretability layer's number per §A). Confirmed unspent immediately beforehand — every
one of the 12 target checkpoints still carried zero `test_*`/`ext_*` keys. Executed
notebook archived at
`CLASSIFIER/notebooks/COMPARISON/_results/temporal_first_ladder_tier4_frozen_read_2026-08-24.ipynb`.

**Pre-flight fix, caught by dry-running Tier 1–3 first with `RUN_FROZEN_READ=False`
before spending anything.** The papermill parameter cell (cell 1) had no `tags:
["parameters"]` metadata. Without it papermill *prepends* an `injected-parameters` cell
instead of replacing the original — so the notebook's own untagged cell then re-executes
its hardcoded defaults immediately afterward and **silently stomps every override back**,
including `RUN_FROZEN_READ` itself. Uncaught, this would not have spent the test read
(the flag would have stayed `False`) but would have made "flip the flag and run" a silent
no-op, indistinguishable from success in the executed-notebook output. Fixed by tagging
the cell; re-verified with a second dry run showing no injection warnings and the correct
override taking effect. Also extended `RUNG_PREFIXES` with `S2_gate`, `S3_fusion`,
`S4_attnpool`, `S5_dualscore`, `SENS` (Tier 1–3 only, needed for the transport-check
diagnostic on the S5 secondary; deliberately left out of `RUNG_CHAIN`, per the comment
now in the cell, since none branch from `S1c_recon_random`).

**Results — in-domain test (n=64) and OASIS-3 (n=60), mean ± SD over 4 seeds:**

| arm | role | in-domain test AUC | OASIS-3 AUC | OOF (for reference) |
|---|---|---|---|---|
| **S1 flip** | **PRIMARY** | **0.7909 ± 0.0162** | **0.4892 ± 0.0224** | 0.7488 ± 0.0033 |
| S1b ssl | secondary (sensitivity) | 0.7760 ± 0.0568 | 0.4602 ± 0.0274 | 0.7502 ± 0.0125 |
| S5 dual-score | secondary (interpretability layer) | 0.7870 ± 0.0139 | 0.5070 ± 0.0109 | 0.7331 ± 0.0173 |

**Transport checks** (95% prediction interval from OOF SE ⊕ test SE, per `run_frozen_reads`):

- S1: OOF 0.7488, PI [0.7326, 0.7650], test 0.7909 → **inconsistent** — test lands *above*
  the interval, not below. In-domain test AUC is higher than the pooled OOF estimate.
- S1b: OOF 0.7502, PI [0.6932, 0.8072], test 0.7760 → consistent (wide PI — n=4 test-seed
  SD of 0.0568 dominates).
- S5: OOF 0.7331, PI [0.7113, 0.7548], test 0.7870 → inconsistent, same direction as S1.

**Read this soberly, not as a second win.** The in-domain test set is n=64 — the plan's
own "Honest expectation" (top of this document) already states differences below ~0.08
are not resolvable at this size, and a positive transport-check surprise at this n is not
grounds to prefer the test number over OOF as more reliable; if anything it argues the
reverse, since OOF pools 248 subjects across 5 folds and the point estimate is the
better-powered one. Report both numbers, flag the direction as "did not degrade
in-domain, contrary to the winner's-curse prior" — a fact, not evidence of anything
beyond what n=64 can support — and do not update the headline claim on it.

**The load-bearing result is OASIS-3, and it is at chance.** All three arms land within
noise of AUC=0.5: under the null the per-seed SE at n≈60 (31/29 split) is ≈0.075, so even
a single seed's 0.47–0.52 range is unremarkable, and the 4-seed-mean SE (≈0.011–0.037) puts
every arm's mean within ~1 SE of 0.5. This is **not** "below-chance failure" — it is
**no signal transferred to a cohort never seen in training or pretraining**, tightly and
consistently across all three arms and all four seeds (S1: 0.4705–0.5217; S1b:
0.4416–0.5006; S5: 0.4972–0.5217). Contrast with in-domain test (0.77–0.79 for all three):
the winning model works well within the ADNI+DELCODE distribution it was pooled from, and
carries essentially zero information to a genuinely external cohort.

**Why — a pre-registered mechanism, and the escalation trigger it should have fired
was missed until this read.** `DOCS/flipped/PLAN.md`'s own "Cohort-shift control" section
(Phase 3) pre-registers exactly this risk and a mandatory probe for it: *"if
`cohort_probe_auc > 0.75` on the winning arm, run `cohort_conditioning: 'adversarial'`
... as an additional arm and report both."* Every completed TFGN arm's `run_summary.json`
already carries this probe, and it has been **above threshold since the first ladder
runs, unnoticed through every prior section of this document**:

| arm | `cohort_probe_auc` | escalation (>0.75) |
|---|---|---|
| S1_flip | 0.860 ± 0.010 | **True** |
| S1b_ssl | 0.890 | **True** |
| S1c_recon_random | 0.860 | **True** |
| S2_gate | 0.869 | **True** |
| S3_fusion | 0.860 (identical to S1, §B) | **True** |
| S4_attnpool | 0.847 | **True** |
| S5_dualscore | 0.863 | **True** |
| SENS (min_visits=3) | 0.742 | False (just under, smaller pool) |

A logistic probe decodes ADNI vs. DELCODE from the pooled patient latent at **~0.86 AUC**
despite `cohort_conditioning: 'none'` — the model is encoding cohort identity strongly as
a side effect of learning to classify, with no explicit signal telling it to. That is
exactly the shortcut the "no unseen one-hot slot" design in Phase 3 was meant to prevent
downstream harm from, and OASIS-3 is precisely the unseen-category case the pre-
registration flagged as ill-defined under it. The near-chance OASIS-3 result is the
predicted consequence of an un-escalated cohort shortcut, not an unrelated failure.

**This is a genuine process gap, not a data problem.** The probe was computed and
persisted correctly at every step; nothing was hidden. It simply was never read against
its own pre-registered threshold until this Tier-4 pass pulled every arm's numbers
together. Every §A–§K analysis above is unaffected — the stopping-rule verdicts, the
parameter-count argument, the matched-window crossover, and the SENS decomposition are
all in-domain (OOF/CV) results and do not depend on cross-cohort transfer. Only the
external-generalization claim is affected, and Table B / the thesis's OASIS-3 line must
now read **"no evidence of transfer to an unseen cohort, consistent with an un-escalated
cohort-identity shortcut (cohort_probe_auc≈0.86 ≫ 0.75)"** rather than as a second
performance number alongside in-domain test.

**Escalation status: run, and reported as attempted-and-failed (§M).** The pre-registered
escalation arm (`cohort_conditioning: 'adversarial'`, gradient-reversal cohort head) was
implemented and run on all 4 seeds. It did not recover external transfer — see §M for the
full result (OOF AUC dropped 0.7488 → 0.7066, and `cohort_probe_auc` *rose* 0.86 → 0.94,
the wrong direction). Decision taken 2026-08-24: report the OASIS-3 gap as-is, with both
the missed threshold and the failed mitigation attempt stated explicitly — not as an
unexplored follow-up, but as a limitation that was pursued and did not resolve.

The OASIS-3 line in Table B / the thesis write-up must therefore read: **"no evidence of
transfer to an unseen cohort (AUC 0.4892 ± 0.0224, indistinguishable from chance),
consistent with an un-escalated cohort-identity shortcut (`cohort_probe_auc`≈0.86 ≫ 0.75
threshold); an adversarial gradient-reversal mitigation was attempted and did not recover
transfer (§M) — cohort-invariant representation learning under this pooling protocol is an
open problem, not a solved one."** Do not report OASIS-3 as a clean external-validation
number, and do not omit that a fix was tried and failed — either omission misrepresents
what is known.

### M. Adversarial-conditioning escalation — run, and it failed on both axes

The escalation arm from §L (`tfgn-s1-advcohort-pooled-seed{42..45}`,
`cohort_conditioning: "adversarial"`, gradient-reversal cohort head, everything else
identical to S1) completed all four seeds. It does not recover external transfer, and it
should not be pursued further at this lambda without a documented reason to expect a
different outcome.

**Implementation, for the record.** `model/TFGN/layers.py` gained a standard DANN
gradient-reversal function (`grad_reverse` — identity forward, negated-and-scaled
gradient backward) and `CohortAdversaryHead` (binary ADNI-vs-DELCODE MLP), attached to
`h_pooled` — the exact representation `patient_embeddings` feeds to the cohort probe, so
the escalation targets precisely the quantity the probe measures. `cohort_conditioning:
"film"` now raises instead of silently building a `"none"` model (it was documented in
Phase 3 but never implemented; the same gap the adversarial arm was closing is now closed
for both cases). 9 new tests cover the reversal's gradient sign/scale, head-construction
gating, the loss-component contract, and a fail-loud path if an unmapped cohort (e.g.
OASIS-3) ever reached the adversarial loss. A 2-epoch foreground smoke test (before
committing GPU time to the real block) showed `cohort_probe_auc` dropping from S1's ~0.86
to 0.74 — encouraging at the time, and, in hindsight, a trap: see below.

**Results (pooled OOF, mean ± SD over 4 seeds):**

| arm | OOF AUC | `cohort_probe_auc` |
|---|---|---|
| S1 (baseline) | 0.7488 ± 0.0033 | 0.8600 ± 0.0074 |
| **S1 + adversarial** | **0.7066 ± 0.0075** | **0.9411 ± 0.0131** |

Tier-2 stopping-rule statistic (adversarial vs. S1): fold-matched **−0.0193 ± 0.0022
(ratio −8.68)**, pooled **−0.0422 ± 0.0051 (ratio −8.31)** — both far past the |ratio|>1
threshold, in the loss direction, consistently across all four seeds. This is not noise.

**Both intended effects failed, and the escalation made the diagnostic worse, not
better.** Classification AUC dropped ~0.042 — larger than any single rung's loss in the
entire Block A ladder. The cohort probe, the quantity the escalation exists to suppress,
*increased* from 0.86 to 0.94: the model became **more** cohort-decodable, not less. The
2-epoch smoke number (0.74) was not a preview of the trained result — it was an artifact
of an undertrained network where *no* structure, cohort-relevant or otherwise, was
strongly encoded yet. Read that as a standing caution for this codebase's other early-stop
smoke checks: a probe or diagnostic computed at 2 epochs measures "how much has this
network learned to encode anything," not the phenomenon under test, whenever the network
starts near-random. It happened to move in the reassuring direction here by chance.

**Likely mechanism, stated as a hypothesis, not fact.** `cohort_adv_lambda: 1.0` scales
only the reversed gradient at the GRL, with no separate weight on the cohort loss term
itself (`losses.py::cohort_adversarial_bce` is unweighted BCE) — the textbook-minimal DANN
form. If the primary BCE gradient dominates the shared encoder's gradient at this scale,
a `lambda=1.0` reversal may be too weak to counteract cohort-correlated features the
network finds useful for classification as a side effect of fitting harder — and a weak,
losing adversarial game can *increase* observed cohort separability relative to no
adversarial term at all, if the classifier head only partially learns to ignore what
little invariance pressure it receives while the encoder keeps specializing. This is a
plausible, common DANN failure mode, not a verified diagnosis — no gradient-magnitude
comparison was run to confirm it.

**What was NOT done, deliberately.** No second Tier-4 frozen read was spent on this arm.
Reading OASIS-3 for a model that already lost on OOF classification *and* moved the
targeted diagnostic in the wrong direction would spend a one-shot resource on an arm the
ladder's own stopping rule already rejects — the same discipline that has governed every
rung since S2. No lambda sweep was run either: quietly retrying with a different
`cohort_adv_lambda` after seeing this result would be an undocumented re-run of the exact
kind Phase 0's pre-registration exists to prevent, even in service of a plausible fix.

**Decision (2026-08-24): option 1 — report the escalation as attempted and failed.** No
`lambda_cohort_adv` sweep will be run and the adversarial path is not being pursued
further under this plan. The pre-registration asked for an attempt at
threshold-crossing, not a guaranteed fix; a documented negative result closes the loop
honestly. The write-up combines this with §L's finding: OASIS-3 transfer is at chance, a
cohort-identity shortcut is the leading hypothesis, and the one attempted mitigation
(gradient-reversal, `lambda=1.0`) did not work — and, on the diagnostic it targeted,
moved in the wrong direction. The under-powered-reversal hypothesis in the section above
stays a hypothesis: it is not being tested by a sweep, so it must not be stated as a
settled explanation, only as the leading candidate for *why* this specific attempt
failed. The two options not taken (a `lambda_cohort_adv` sweep as a new pre-registered
arm; abandoning the adversarial framing entirely) remain available as future work if a
reviewer or later phase of this project wants to revisit cohort-invariant training, but
neither is in scope for this plan going forward.

### Next steps, in order

1. **Docs first, no GPU.** Write A–F into `DOCS/temporal-first-ablation.md` as a
   "Batch 5 verdicts (2026-08-24)" addendum: S5 kept (classification-neutral); S3 void
   with the `models.py:175-183` inertness citation and the bit-identity evidence; the
   quadrant temporal-axis decision (`d̃`); the cross-fold → cross-seed Spearman
   deviation; the frozen-arm decision (S1 primary, S5 + S1b secondary).
2. ~~**Comparison notebook.**~~ **DONE 2026-08-25** — `RUNG_PREFIXES` carries `S2_gate`,
   `S3_fusion`, `S4_attnpool`, `S5_dualscore`, `SENS`, and the three `W3_*` arms; S3's row
   is marked VOID in `RUNG_SUMMARY_TABLE` itself (a `status` column, not only prose); the
   SENS-restricted-comparison cell (§E) and Table A/Table B (§F) are wired. Executed
   end-to-end via papermill with `RUN_FROZEN_READ=False`: every number reproduces the
   verified scorecard above exactly (e.g. S1 pooled OOF AUC 0.748816, S3 bit-identical to
   S1, SENS-restricted S1 read 0.7711 vs the doc's 0.7712).
3. ~~**§0.1d interpretability validation**~~ **DONE 2026-08-25** on (`s_topo` from S5, `d̃`
   computed offline) — permutation null (as a DMN network-label spin test, 1000
   permutations), cross-seed Spearman, per-cohort split, and the quadrant scatter, with
   S2's gate map as a supporting panel. Two documented deviations recorded in
   `DOCS/temporal-first-ablation.md`'s "Gate-map validation" section: the atlas TFGN
   consumes has no hippocampal ROI (DMN-only overlap), and "1000 label permutations" is
   implemented as a DMN network-label spin test (subject-label permutation does not apply
   to an anatomical overlap statistic). **Result: neither `s_topo` (percentile 77.9,
   p=0.351) nor `d̃` (percentile 41.6, p=0.739) clears the DMN enrichment test** — the
   pre-registered "gate targets DMN/hippocampal regions" claim is not supported. `s_topo`
   is nonetheless cross-seed stable (mean r=0.928) and correlates with the independent
   `d̃` axis (r=0.456, p=1.2e-11) — reproducible, but not preferentially DMN. Reported
   plainly, not spun positive; this is now the *entire* interpretability contribution,
   since every performance rung above S1 was dropped.
4. ~~**Fix the Tier-4 blocker (§G).**~~ **DONE 2026-08-24** — `checkpoint_extras` hook
   wired end to end, 12/12 checkpoints backfilled and validated at `|ΔAUC| = 0`,
   overwrite guard and adapter-key map fixed, 5 new round-trip tests passing. Tier 4 is
   unblocked. Remaining sub-item: make `SECONDARY_SENSITIVITY_ID` accept a list (§K) so
   S5 and S1b are read in the same single pass.
5. ~~**Tier-4 frozen read.**~~ **DONE 2026-08-24** — S1 primary, S1b + S5 secondaries,
   one pass. In-domain test 0.7909 ± 0.0162 (n=64); OASIS-3 0.4892 ± 0.0224 (n=60,
   statistically indistinguishable from chance). See §L for the full results and a
   pre-registered escalation trigger (`cohort_probe_auc≈0.86 > 0.75`) that fired on every
   TFGN arm and was never actioned.
5b. ~~**Adversarial-conditioning escalation.**~~ **DONE 2026-08-24, FAILED** — implemented
   and run on all 4 seeds (§M). OOF AUC 0.7488 → 0.7066 (Tier-2 ratio −8.68, a clear
   loss); `cohort_probe_auc` rose 0.86 → 0.94, the wrong direction. **Decision: report as
   attempted-and-failed (§L/§M), no further lambda sweep, adversarial path closed for
   this plan.** The OASIS-3 gap is written up as an open limitation with a tried-and-failed
   mitigation attached, not as an unexplored trigger.
6. **Block B gate: closed.** Per Phase 5's own wording, the gate is a cumulative gain
   from S1c-random through S5 exceeding the SE of the seed-level differences. The chain
   delivered −0.1587 (S1c-random), −0.0064 (S2), void (S3), −0.0558 (S4), −0.0046 (S5).
   No rung above S1 was kept. **Block B does not run.** Write Phase 5's own stated
   conclusion — signal quality and sample size, not capacity, are the bottleneck — as a
   thesis result rather than scaling to 300k parameters.
7. ~~**`CHECKS.json` is 7 weeks stale — regenerate it on a clean tree.**~~ **DONE
   2026-08-25.** Verified 2026-08-24: the baseline was written **2026-07-04**, before any
   TFGN file existed, so `run_checks.py` reported every TFGN-era finding as "NEW" no
   matter who wrote it (`[[feedback_checks_json_staleness]]`). Confirmed against HEAD in a
   scratch worktree rather than assumed: bandit and ruff-format deltas at that point were
   fully explained by staleness, not by this work. Regenerated the honest way — deleted
   the gitignored `CHECKS.json` and reran `scripts/run_checks.py` on the tree as it stood
   after items 1-3 above (only `.ipynb`/`.md` changes since the last verification, no `.py`
   touched, so the backlog count is unaffected by this session's edits) rather than
   hand-editing it (`.claude/rules/ci.md`). Fresh baseline: ruff-format 41, McCabe C90 52,
   mypy 0, bandit 173, pip-audit 111 files/findings, all "baseline established", blocking
   gates (`ruff check`, `pytest`) PASS. `CHECKS.json` is gitignored, so there is nothing to
   commit for this step — it is a local ratchet cache, not tracked.
8. ~~`python scripts/run_checks.py` once before hand-off.~~ **DONE 2026-08-25** — see item 7;
   `RESULT: PASS — no new issues introduced.` No dirty `CLASSIFIER/outputs/` artifacts
   remained to commit (the SENS/W3 run artifacts were already committed in prior batch-5
   commits); only this session's doc and notebook edits are pending commit.

**No further GPU runs are required to finish the ladder.** The §G backfill validated
12/12 at `|ΔAUC| = 0`, so the re-run fallback is closed out. The only remaining compute
is the single Tier-4 frozen-read pass — CPU-side scoring of saved checkpoints.

---

## Phase 5 — Block B (written now, run only if Block A clears the rule)

> **Gate evaluated 2026-08-24: CLOSED — Block B does not run.** No rung above S1 was
> kept; see "Ladder complete" §Next steps item 5 above for the derivation.

Gate: **S1c-random–S5 must show a cumulative gain (OOF, Tier 2 fold-matched statistic)
exceeding the SE of the seed-level differences, read only after SENS reports** — per
"Ladder state and corrected order (2026-08-24)" above, S2–S5 branch from **S1**, not
S1b, and the gate is evaluated once the whole chain (S1c-random through SENS) has run,
not at S1c-random↔S3 in isolation. If the flip cannot clear the noise floor at 64k
parameters, a 300k model is not the answer and the honest thesis result is that signal
quality and sample size, not capacity, are the bottleneck — write that instead of
scaling.

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
