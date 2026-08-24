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
  Held out for a single final read on the frozen winner(s) — **not** used to drive
  ladder decisions; see the 2026-08-24 addendum to "The stopping rule" below.
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

**Addendum (2026-08-24, before any ladder decision was read — S1/S1b were in flight,
none had reported).** The stopping rule now reads **CV out-of-fold (OOF) AUC**, not
in-domain test AUC. This strengthens rather than weakens the pre-registration and is
recorded here per its own deviation rule ("nothing in this document may be changed
after seeing a result... a follow-up doc as a deviation, not a silent edit"):

- **More statistical power.** OOF is n=248 vs test n=64 — the noise floor scales
  ~1/√n, so the "~0.08 AUC unresolvable" caveat above shrinks to **~0.04**. The ladder
  can resolve smaller effects than the original text implies.
- **No selection-on-test.** The 64-subject in-domain test set is now read exactly
  once, on the frozen final winner(s) — never during the ladder.
- **Mechanically unchanged.** `mean(Δ) > SE(Δ)` over 4 seed-level means, matched fold
  for fold, is computed on fold-validation AUCs instead of test AUCs — same formula,
  same `common/comparison.py` machinery (`paired_delong_test`, `paired_bootstrap_ci`
  on paired OOF predictions), just a different column.
- **The S0c↔S1 and S0b↔S1c headline contrasts move to OOF too** — same paired-seed
  logic, read at the same point in the chain.
- **Final read, once the ladder is frozen:** one in-domain test read (n=64) + one
  external OASIS-3 read (n=60, unchanged — descriptive, never selects), both at the
  OOF-derived threshold, both with bootstrap CIs.
- **Winner's-curse caveat, keep in the write-up.** OOF-based selection reuses the
  same 248 subjects across every ladder decision, so the winning arm's own OOF AUC is
  mildly optimistic — it is not an unbiased performance estimate. The single final
  test read is what supplies that; report it, not the OOF number, as the headline
  in-domain figure. Order of claims: **select on OOF → report on test → generalize on
  OASIS-3.**

Every `run_summary.json`'s `test_*` / `ext_oasis3_*` fields are computed and recorded
as before — this addendum changes what the aggregation notebook *reads* for the
stopping-rule decision, not what any notebook computes or writes.

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

**S1b fork decision (2026-08-24).** S1 and S1b both completed all 4 seeds (pooled
protocol, `POOLED_WHOLE_BRAIN`); the stopping rule (OOF AUC, per the addendum above) was
read on the S1b-vs-S1 pair — the comparison that sets `node_lstm_init` for S1c–S5, per
"Each rung inherits the kept knobs of rungs before it... S1b is the only fork" — from
each run's `cv_results.val_auc` (5 folds):

| seed | per-fold ΔAUC (S1b − S1) | seed mean |
|---|---|---|
| 42 | −0.0104, −0.0139, +0.0087, +0.0533, +0.0556 | +0.01865 |
| 43 | −0.0347, +0.0694, +0.0278, −0.0496, +0.0323 | +0.00903 |
| 44 | −0.0174, +0.0347, +0.0382, +0.0312, +0.0054 | +0.01844 |
| 45 | −0.0313, +0.0243, +0.0451, +0.0055, −0.0609 | −0.00344 |

`mean(Δ) = 0.01067`, `SE(Δ) = 0.00521`, ratio `2.05` → **passes** (`mean(Δ) > SE(Δ)`,
3 of 4 seeds positive). Per the rule's own required language: **"worth carrying
forward,"** not "significant," not "proven." Corroborating, not decisive on its own: a
persistence-baseline check on P2's val split (`x(t+1)=x(t)`, MSE 0.0630) shows the
untrained node-LSTM started *worse* than persistence (0.0658 at epoch 0) and training
pulled it 27.2% below (0.0459 at epoch 83) — the SSL checkpoint learned something real,
consistent with S1b's win here being a genuine effect rather than noise.

**Decision: `node_lstm_init: pretrained_finetuned` carries forward into S1c–S5.** This
is not a silent edit — S1's own config (`node_lstm_init: random`) is untouched in the
registry; only S1c onward inherit the winning value, per the pre-registered chain rule.

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

## Evaluation & Comparison Protocol (addendum, 2026-08-24)

Pre-registered addendum, written before any ladder decision was read from
in-domain test (S0a-S0d/S1/S1b were complete or in flight at the time; none
had been *selected on* — the OOF-only stopping rule above already governed the
S1b fork decision). This section is the operational contract the comparison
notebook (`notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb`)
implements; "The stopping rule" above states *why* CV cannot predict the test
AUC (winner's curse + sampling noise) — this section is the *how*.

### Tier 1 — floor gates (run once, before any arm comparison)

An arm that does not beat every applicable floor on OOF AUC is stopped, not
compared:

| Floor | Arm / source | Question |
|---|---|---|
| Demographics | `tfgn-s0-demo-pooled` — `logregdrift` adapter, `feature_set: "demo"` (features = `[age, sex]` only, no PCA/ΔA — `adapters/logreg_drift.py`) | Is there signal beyond age/sex at all? |
| Static baseline | `oof_static_n1_auc` in every arm's `run_summary.json["oof"]` (no extra runs) | Does longitudinal information beat baseline-scan-only? |
| Persistence (SSL only) | `tfgn-nodelstm-ssl-pooled`'s own `run_summary.json["persistence_baseline"]` (already computed by the SSL notebook itself — `val_loss_mse=0.0630` vs the trained LSTM's `0.0459`, a 27.2% improvement) | Did the node-LSTM learn dynamics or just copying? |

**Deviation from the original checklist, recorded here per this document's own
rule.** The static baseline is the OOF-side N=1 row, not
`early_detection_table`'s in-domain-test N=1 row — the latter is a frozen
read (Tier 4) and reading it during floor-gating would violate "read exactly
once, after the ladder is frozen." Mechanically identical (same
`truncate_to_n_visits`/`eval_split` hooks at N=1), just scored on the CV pool
via a per-fold probe (`common.crossval.run_kfold_cv`'s `fold_probe=`,
`common/oof.py::build_oof_frame`) instead of the test bundle.

### Tier 2 — selection (the stopping rule, mechanically unchanged)

Unchanged from "The stopping rule" above: per-seed mean paired fold-matched
ΔAUC, `mean(Δ) > SE(Δ)` across the 4 seed means, on OOF AUC. **New tie-breaker**:
among kept arms, prefer the simplest configuration within one SE of the best
(modified one-standard-error rule) — the defense against winner's-curse bias
accumulating across ten rungs. External OASIS-3 AUC is recorded per arm and
never used for selection.

### Tier 3 — robustness vetoes (thresholds fixed here, before any result)

A kept rung must additionally clear every veto below. Changing a threshold
after seeing a result is a documented deviation, never a silent edit.

| Veto | Metric | Threshold | Code path |
|---|---|---|---|
| Per-cohort collapse | OOF AUC on ADNI-only / DELCODE-only subjects | veto if either < the demographics floor's per-cohort OOF AUC | `common/oof.py::oof_metrics` → `oof_auc_<cohort>` |
| Threshold instability | SD of `cv_results["best_threshold"]` pooled across the 4 seed runs' `run_summary.json` (5 folds × 4 seeds = 20 values) | veto if SD > 0.15 | comparison notebook, Tier-3 section (no new artifact — `cv_results.best_threshold` was already saved per run) |
| Calibration | ECE after temperature scaling fit on OOF | veto if ECE > 0.10 | `calibration.json`'s `ece_oof_cal` (already written per run by `common/calibration.py` via the notebook's calibration cell) |
| Scan-count shortcut | OOF Spearman(predicted risk, n_scans), within-stable | veto if r > 0.3 | `common/oof.py::oof_metrics` → `oof_prob_nscans_spearman_non_converter`; mechanism corroborated on a kept arm's saved checkpoint via `common.visit_confound.within_subject_prob_slopes` |
| Cohort shortcut | `cohort_probe_auc` (already mandatory — "Cohort-shift control" above) | escalation at > 0.75 → adversarial arm (unchanged) | notebook cohort-probe cell |

**Mandatory reporting, no veto:** PR-AUC (`oof_pr_auc`) and balanced accuracy
(`oof_balanced_accuracy`) alongside AUC in every rung table row.

### Tier 4 — estimation (the only place test sets appear)

After the ladder is frozen, every ladder arm having run with
`defer_test_eval: true` (no `test_*`/`ext_*` keys written during the ladder —
see "Determinism"/`LONGITUDINAL_COMMON_DELCODE.ipynb`'s Configuration cell):

1. **In-domain test (64):** one read of the winning arm, via
   `common/frozen_read.py::score_frozen_split(..., record_as="test")` — reloads
   the saved checkpoint (`adapter.load_state`), scores at the OOF-derived
   threshold (`adapters.read_run_threshold`), records through the same
   `record_test_metrics` every non-deferred run already uses. Bootstrap CI via
   `common/comparison.py::paired_bootstrap_ci`.
2. **External OASIS-3 (60):** one read, same threshold,
   `score_frozen_split(..., record_as="external", cohort="oasis3")`, bootstrap CI.
3. **Transport criterion (pre-registered):** CV→test transport is *consistent*
   if the test AUC lies inside `mean_OOF ± 1.96·√(SE_OOF² + SE_test²)` — never
   "equal to CV."
4. **Winner's-curse statement (mandatory in the write-up):** the winning arm's
   OOF AUC is expected to be optimistic; the frozen test read is the unbiased
   estimate; report both side by side.

### What the thesis claims (three separated claims)

1. **Mechanism (from CV):** which components matter — flip, node-LSTM SSL,
   GVAE reconstruction, gate, fusion, readout — each with its paired ΔAUC ± SE.
2. **Performance (from the frozen reads):** final AUC + CI, in-domain and
   external. Framed as estimation, never as "predicted by CV."
3. **Trust (from the vetoes + gate-map validation):** the model is not reading
   demographics, cohort identity, or scan count; the gate map is stable across
   folds and seeds (per "Gate-map validation" above).

### Comparison-notebook wiring checklist

`notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb` implements this
protocol when its sections:

- [ ] read the stopping rule from `oof_predictions.csv` / `run_summary["oof"]`, never `test_*`
- [ ] carry per-cohort OOF AUC columns (ADNI-only, DELCODE-only) next to the pooled number
- [ ] carry the threshold-SD (across seeds) and ECE (OOF-fit) columns
- [ ] carry PR-AUC and balanced accuracy columns
- [ ] report the OOF N=1 row as each arm's static baseline, and the SSL persistence-baseline cell
- [ ] include the demographics-floor arm (`tfgn-s0-demo-pooled`)
- [ ] run the scan-count-shortcut veto + mechanism check on every kept arm
- [ ] end with exactly one frozen in-domain test read and one frozen OASIS-3 read
      (`common.frozen_read.score_frozen_split`), each with a bootstrap CI and the
      §Tier-4.3 transport statement
- [ ] carry a Table-A section (matched window, T∈[2,3]) alongside the Table-B ladder table
- [ ] assert the w3 arms' `oof_predictions.csv` subject sets equal Block A's (248 CV / 64
      test) before reporting Table A

## Matched-window SOTA comparison (addendum, 2026-08-24)

Pre-registered before any `tfgn-w3-*` run, per this document's own "never a silent edit"
rule. Motivation and arm table are in `DOCS/flipped/PLAN.md`'s "Matched-window SOTA
comparison" addendum — this section fixes the parts that must not move once results
exist.

**Arms and exact knob deltas.** `tfgn-w3-gelstm-frozen-pooled` and
`tfgn-w3-gelstm-random-pooled` are byte-identical to S0b / S0c respectively
(`adapter: gelstm`, same `config_path`, same pooled-GAAE `checkpoint_path`) with exactly
one added key: `max_visits: 3`. `tfgn-w3-winner-pooled` is `adapter: tfgn` with the
winning ladder config plus `max_visits: 3`; its `hyperparams` are written only once S1c–SENS
have reported, as a further addendum to this section, never guessed ahead of the ladder.

**Not a ladder rung.** The w3 block is evaluated on OOF only and never enters Tier 2
selection — it answers "how do we compare under the competitor's input constraint", not
"which knob should the ladder keep". None of Tier 1–3 gate it against the ladder's own
floors; it is reported alongside, not folded into, the rung table.

**S0d reuse — the guard, not just the assumption.** BrainTokenGT is not re-run for this
block: `BRAINTOKENGT/adapter.py:184-186` applies `n_scans >= min_visits` before
`window_item(..., max_visits=...)` truncates, the same filter-then-truncate order every
adapter in this ladder uses (`model/GELSTM/dataset.py:74-76, 187-191`) — so S0d's existing
248 CV / 64 test subjects are already the matched-window pool, not a superset later cut
down. S0d's numbers are reused verbatim in Table A with the existing BrainTokenGT
determinism caveat (same-seed test AUC spanning 0.357–0.708, "The arms" above) attached
unchanged.

**Reporting contract.** Table A (matched window, BrainTokenGT/S0d vs GELSTM-frozen/random
w3 vs TFGN-winner w3) is the strict head-to-head; Table B (the ladder as registered, full
trajectory) carries the thesis contribution claim. A windowing cap to T∈[2,3] discards
exactly the visits-4-10 information the LSTM/GVAE architectures exist to exploit, so
Table A must never be presented as evidence for or against the flip itself — only as "how
we compare to the SOTA competitor under its own input constraint."

**Test-set discipline.** If a matched-window test number is wanted, it is read exactly
once, as part of the same single Tier-4 frozen-read pass used for the ladder winner
(`common/frozen_read.py::score_frozen_split`) — never a second, separate peek at test.

## Running it

From `CLASSIFIER/`, with the project-root `.venv` active — see `DOCS/flipped/PLAN.md`
Phase 4 for the full dispatch sequence and the registry
(`CLASSIFIER/experiments/temporal_first.yaml`). Each rung's `hyperparams` are only
finalised once the previous rung has reported against the stopping rule above; that
finalisation is recorded as an addendum to this document, never a silent edit to the
tables above.
