# Temporal-first vs spatial-first: does flipping the pipeline order help?

**Status:** pre-registered 2026-08-23, before any TFGN run exists. Nothing in this
document may be changed after seeing a result — if a knob turns out to need a different
value, that goes in a follow-up doc as a deviation, not a silent edit here.

## The question

Every longitudinal classifier in this repo (GELSTM, GEGRU, GEC, GEP) is
**spatial-then-temporal**: each visit's whole-brain FC graph is pooled to one vector by a
graph encoder *before* any temporal model sees it (`GELSTMClassifier.encode_visit`,
`CLASSIFIER/model/GELSTM/models.py:191`). Region identity is destroyed by mean-pooling
before the trajectory is modelled at all.

TFGN (`CLASSIFIER/model/TFGN/`) inverts the order — **temporal-then-spatial**: a
node-shared LSTM encodes each of the 200 regions' own FC-row trajectory first, a learned
gate suppresses regions that never change, and only then does a graph encoder (a
variational GAT, "GVAE") propagate the surviving per-node dynamics across the baseline
connectome. This document fixes, in advance, every knob and every statistical decision
the ladder in `DOCS/flipped/PLAN.md` Phase 4 depends on.

## Protocol (fixed, not a knob)

- **Training pool:** ADNI + DELCODE `downstream` train+val, `min_visits=2` → 248 subjects.
- **In-domain test:** ADNI + DELCODE `downstream` test, `min_visits=2` → 64 subjects.
  Drives every ladder decision below.
- **External test:** all 60 OASIS-3 subjects, scored once per arm at the in-domain
  OOF-derived threshold. Never used to select an arm, never re-thresholded.
- **Seeds:** 42, 43, 44, 45 for every arm.

## The stopping rule

For ladder step *k* vs *k−1*, the independent unit is the **seed** (n=4), not the fold —
folds within a seed share the same model class and are not independent draws. For each
seed, compute the mean paired per-fold ΔAUC (step *k* minus step *k−1*, matched fold for
fold); then report the mean and the standard error of those 4 seed-level means.

**A step is kept only if `mean(Δ) > SE(Δ)` on the in-domain test AUC.**

At n=4 no p-value is claimed, and the SE itself is high-variance — this is a heuristic
screen, not a significance test. Its two possible outcomes are asymmetric and must be
described that way in every write-up:

- **Fails (`mean(Δ) ≤ SE(Δ)`): "undetectable at this sample size."** Never "harmful,"
  never "the mechanism doesn't work" — a real effect below ~0.08 AUC is invisible here
  regardless of sign.
- **Passes (`mean(Δ) > SE(Δ)`): "worth carrying forward."** Never "significant," never
  "proven" — it is the bar for keeping a knob in the chain, not a publishable result on
  its own.

External OASIS-3 AUC is reported for every kept arm with a bootstrap CI
(`CLASSIFIER/common/comparison.py::paired_bootstrap_ci`) and is descriptive only — it
never decides which arm survives.

## The arms

One config object, `TFGNTrainConfig` (`CLASSIFIER/configs/tfgn.py`), with every ladder
rung a knob change from the previous rung — the same pattern as `encoder_init` in
`CLASSIFIER/configs/encoder.py` for the reconstruction-value ablation.

| Rung | knob change vs previous rung | Question |
|---|---|---|
| S0a `logreg-drift` | linear model on `ΔA` PCA features | Floor: is there signal in raw change at all? |
| S0b `gelstm-frozen` | today's spatial-first model, pretrained frozen GAAE | Spatial-first **with** a self-supervised encoder |
| S0c `gelstm-random` | spatial-first, `encoder_init: random` | Spatial-first **without** one — matched floor for S1 |
| S0d `braintokengt` | competitor baseline | External reference (see caveat below) |
| S1 `flip` | `node_lstm_init: random`, gate off, `recon_target: none`, `fusion: z_only`, `readout: mean` | Flip alone, no self-supervision anywhere → compare to **S0c** |
| S1b `ssl` | `node_lstm_init: pretrained_finetuned` (self-supervised node-LSTM, see Phase 2 of the plan) | Does node-level forecasting pretraining help? |
| S1c `recon` | `recon_target: delta_a_topk`, GVAE reconstruction losses on | Both encoders self-supervised → compare to **S0b**. **Headline arm.** |
| S2 `gate` | `use_gate: true` + sparsity/drift anchors | Does suppressing static regions help? |
| S3 `fusion` | `fusion: concat_residual` | Does preserving unsmoothed per-node H help? |
| S4 `attnpool` | `readout: attention` | Does attentive pooling beat mean pooling? |
| S5 `dualscore` | `dual_score: true` | Interpretability, zero risk to the backbone (kept regardless of AUC) |
| SENS `minvisits3` | winning config, `min_visits: 3` | Does the flip's advantage grow with longer sequences? |

**BrainTokenGT caveat.** `[[project_stability_audit_btgt_gelstm]]` established that
BrainTokenGT's "stabilized" arm is not run-to-run reproducible on GPU (same-seed test AUC
spanning 0.357–0.708) because `torch.use_deterministic_algorithms` is never engaged for
its scatter/gather ops. S0d is reported as a reference point with that caveat attached,
not treated as a clean baseline the way S0b/S0c are.

**The two clean contrasts.** Without S0c and S1c the flip is handicapped: S0b's GAAE has
3 700 unlabelled graphs of reconstruction pretraining behind it, while a naive S1 GVAE
learns only from the classification gradient on 248 labelled subjects — a loss there would
be impossible to attribute to "temporal-first is worse" versus "this particular encoder
never got pretrained." The two contrasts that isolate the flip itself:

- **S0c ↔ S1** — neither encoder pretrained.
- **S0b ↔ S1c** — both encoders self-supervised. **This is the number reported as the
  thesis headline**, not S0b↔S1.

## Reconstruction target (Phase 2/S1c)

`σ(ZZᵀ) ∈ (0,1)` (the standard inner-product decoder) cannot be trained with BCE against
`ΔA = A^{(T)} − A^{(1)} ∈ [−2, 2]` — that target is signed and unbounded relative to a
BCE target's `[0,1]` range. Three well-typed alternatives, one pre-registered default:

| `recon_target` | decoder | target | loss |
|---|---|---|---|
| **`delta_a_topk`** (ladder default) | `σ(ZZᵀ)` | binary change-mask `M_ij = 1[\|ΔA_ij\| ≥ q_{1−κ}]`, `κ=0.10`, quantile computed **per subject** | `BCEWithLogitsLoss(pos_weight=(1−κ)/κ)` |
| `delta_a_mse` (fallback) | `tanh(ZZᵀ/√d)` — no sigmoid | `ΔA/2 ∈ [−1,1]` | MSE |
| `a_last` (Block-B contrast only) | `σ(ZZᵀ)` | `(A^{(T)}+1)/2 ∈ [0,1]` | BCE |

`delta_a_topk` is the default because it literally encodes "which edges changed" — the
premise the whole architecture is built on — keeps the sigmoid decoder mathematically
valid, and is the same quantity Block B's edge-change-ranking objective reuses. Guard,
enforced in `test_tfgn.py`: if the per-subject positive-edge fraction falls outside
`[0.01, 0.5]` for any subject, construction raises rather than silently training on a
degenerate mask — that is the trigger to fall back to `delta_a_mse`, not a runtime warning.

**Addendum (fix A0.4, 2026-08-24).** `delta_a_mse`'s `ΔA/2 ∈ [−1,1]` target and `a_last`'s
`(A^{(T)}+1)/2 ∈ [0,1]` target both assume `A` is raw Pearson `r ∈ [-1,1]`. The pooled
pipeline's default `file_variant` is `z_transformed` (Fisher-z, unbounded), under which
both targets fall outside their loss's valid range. `TFGNAdapter.__init__`
(`CLASSIFIER/adapters/tfgn.py`) now raises `ValueError` if either target is selected
against `file_variant="z_transformed"`. `delta_a_topk` (the ladder default) is scale-free
— its change-mask is a per-subject quantile threshold, invariant to the FC transform — so
this guard never fires on the pre-registered ladder path; it exists for the Block-B `a_last`
contrast arm and the `delta_a_mse` fallback, both of which must pass `file_variant="raw"`
explicitly if used.

## Anchoring quantities (Phase 3, S2/S5)

**Topological anchor** (`s_i^{topo}` target). Eigenvector centrality is ill-defined on a
signed FC matrix — negative edge weights break the Perron–Frobenius guarantee the power
iteration relies on. The pre-registered anchor is **strength centrality on `|A_0|`**: row
sums of the absolute, kNN-sparsified baseline adjacency, z-scored with train-fold
statistics. (Eigenvector centrality on `|A_0|`, the all-positive version, is a valid
alternative and may be reported as a secondary column, but strength is what the ladder
scores against.)

**Drift anchor** (`s_i` target for `lambda_drift`), reconciled with the sparsity prior. A
raw drift target `‖x_i^{(T)} − x_i^{(1)}‖₂` is dense — every node has some nonzero drift —
so anchoring `s_i` to it directly fights `KL(s̄ ‖ ρ=0.15)`, which is pushing most gates
toward zero. The anchor is instead the **within-subject drift rank pushed through a sharp
sigmoid centred at the `(1−ρ)` quantile**:

```
q_i  = rank(d_i) / (N − 1)                         # drift quantile, per subject, in [0,1]
d̃_i = σ( (q_i − (1 − ρ)) / τ_d ),   τ_d = 0.05      # ≈1 for the top-ρ nodes, ≈0 elsewhere
```

By construction `mean(d̃) ≈ ρ`, so the two regularisers now agree instead of pulling the
gate in opposite directions; using ranks instead of raw drift also makes the anchor
scale-free across cohorts whose FC has different amplitude. Additionally,
`λ_drift = 0.1 · λ_sparse`, so if the two ever do disagree at the margin the sparsity
prior — which is directly tied to the model's own gate statistics rather than an external
target — dominates. Test: `mean(d̃)` within tolerance of `ρ` on synthetic drift vectors.

## Cohort-shift control

ADNI and DELCODE differ in scanner, protocol, and follow-up rhythm — DELCODE is ~90 %
regular 12-month visits, ADNI's median inter-visit gap is 371 days with IQR 207–419 (CV
0.65). A pooled model can therefore learn cohort identity as a shortcut rather than
disease signal. Feeding cohort as a FiLM covariate was considered and rejected: it makes
external transfer to OASIS-3 ill-defined, since OASIS-3 would be an untrained one-hot
category at test time and the model's behaviour there is not specified by training.

**Default: `cohort_conditioning: "none"`.** Every pooled run instead reports a mandatory
**cohort probe** — a logistic regression decoding `cohort ∈ {adni, delcode}` from the OOF
patient latents — as `cohort_probe_auc` in `run_summary.json`.

**Pre-registered escalation.** If `cohort_probe_auc > 0.75` on the arm that would
otherwise be reported as the winner, run that arm again with
`cohort_conditioning: "adversarial"` (a gradient-reversal cohort-decoding head trained
adversarially against the shared encoder) and report both numbers side by side. This
threshold and this specific remedy are fixed now so a high probe score cannot be quietly
reinterpreted after the fact.

The aggregation notebook additionally reports the gate map and the S5 quadrant scatter
**split by cohort** regardless of the probe's outcome.

## Gate-map validation (S2, S5) — pre-registered so it survives an underperforming rung

Fixed now, independent of whether S2/S5 clear the stopping rule on AUC, because a gate
that fails to improve classification can still be a valid — or invalid — interpretability
artifact, and that question must not be quietly dropped if the AUC story is negative:

- **Permutation null.** 1000 label permutations; report the percentile of the observed
  DMN/hippocampal overlap in `s` against the null distribution.
- **Cross-fold stability.** Spearman correlation of the gate vector `s` across the 5 folds
  × 4 seeds (20 gate vectors, pairwise or against the fold-42 reference — report both).
- **Per-cohort split.** Both statistics above computed separately for ADNI-only and
  DELCODE-only subjects, to catch a gate that is really tracking cohort rather than
  disease.

This mirrors the repeatability caveat already flagged for saliency-style outputs in
medical imaging in the original architecture proposal — it is mandatory, not optional,
precisely because saliency maps routinely fail this check silently.

## Determinism

TFGN uses GATv2's scatter-based attention aggregation and sparsemax, both of which have
nondeterministic backward kernels on GPU by default. Without forcing determinism, the
stopping rule's seed-level SE would be measuring GPU scheduling noise, not model variance.
Every TFGN run sets `strict_determinism: true`, which calls
`SHARED.seeding.set_seed(seed, strict=True)` — `torch.use_deterministic_algorithms(True)`
plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Verified once before the ladder starts: run
`tfgn-s1-flip-pooled-seed42` twice and diff `run_summary.json`'s metric block; they must
be byte-identical.

## Running it

From `CLASSIFIER/`, with the project-root `.venv` active — see `DOCS/flipped/PLAN.md`
Phase 4 for the full dispatch sequence and the registry
(`CLASSIFIER/experiments/temporal_first.yaml`). Each rung's `hyperparams` are only
finalised once the previous rung has reported against the stopping rule above; that
finalisation is recorded as an addendum to this document, never a silent edit to the
tables above.
