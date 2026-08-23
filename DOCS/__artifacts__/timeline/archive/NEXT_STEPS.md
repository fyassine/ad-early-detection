# What to do after the two defect fixes — 22 Aug 2026, evening

Companion to `SUBMISSION_RUNWAY.md` (dates and cut order are authoritative there),
`STABILITY_AUDIT.md` (the BrainTokenGT/GELSTM variance thread) and `comparison-plan-v2.md`
(phase definitions). This doc covers one question only: **given that external validation
came back at chance on both cohorts, what is the next thing to run?**

**Deadline context:** evidence freezes **26 Aug**, draft to supervisor **28 Aug**. Everything
below is scoped to fit inside that, and Phase 3 is the part that must happen even if
Phases 1–2 are abandoned.

---

## Correction to the triage order written earlier today

`SUBMISSION_RUNWAY.md` finding 3 and `comparison-plan-v2.md` §5 list three hypotheses for
the at-chance external result. **Two of them are now excluded by evidence collected since,
and the ordering they imply is wrong.** Do not work that list top-down.

| Hypothesis (as written) | Status now | Evidence |
|---|---|---|
| Label harmonization fault | ❌ **excluded** | `converter_status` and `label` agree perfectly in both cohorts (ADNI 65/127, OASIS-3 31/29, zero off-diagonal). The dataset takes `converter_status` first, and it is `int64` 0/1. |
| Δt normalisation (`MAX_INTERVAL_MONTHS = 108`) | ❌ **excluded as a within-cohort cause** | The normaliser divides *inter-visit* intervals, and ADNI's longest is 2031 d ≈ 66.8 months — inside 108, so nothing clips. Scale differs from DELCODE, but training and testing happen on the same scale within a cohort, so it cannot explain at-chance CV. |
| Visit-count / length shortcut (added as a candidate) | ❌ **excluded, in all three cohorts** | AUC of `n_scans` alone: DELCODE **0.457**, ADNI **0.477**, OASIS-3 **0.523**. Nobody has a length shortcut, so its absence in ADNI cannot explain the gap either. |
| Data loading dropped or mangled subjects | ❌ **excluded** | The executed notebook logs `CV pool {stable: 101, converter: 52}` and `Test {stable: 26, converter: 13}` — exactly the split CSVs. `min=2 max=10 mean=3.1` scans/subject matches the manifest. The `allowed_days` filter is not eating visits. |
| **Feature quality — the ADNI/OASIS-3 FC matrices themselves** | ❌ **excluded, 23 Aug** | Phase 1 label-free positive control (`DATA/manifest/probe_sex_decoding.py`), sex decoded from baseline FC, 5-fold CV logistic regression: DELCODE 0.6191 ± 0.0691 (n=167), **ADNI 0.7131 ± 0.0883 (n=192)**, OASIS-3 0.5368 ± 0.1129 (n=60). Zero subjects missing FC in any cohort. ADNI decodes sex *better* than DELCODE — the A.3 extraction is not broken for ADNI. OASIS-3 is inconclusive (CI spans chance to ~0.65 at n=60), consistent with it being the underpowered secondary cohort, not evidence of a fault. |
| **Genuine domain difficulty** | ✅ **this is now the finding** | Row above excluded → Branch A fires (below). The at-chance ADNI conversion result stands as a reportable finding, not a bug. |

**Note on BrainTokenGT's within-ADNI CV of 0.705.** Read it as evidence of overfitting, not
of feature quality: the same runs score **0.427** on held-out test. A model with CV 0.705 and
test 0.427 is not telling you the features are informative.

---

## Phase 0 — the two defects — **done, 22 Aug**

**Defect 1 — `training_config.seed` was `42` in every run on disk. Fixed.**

`CLASSIFIER/common/experiment_utils.py::build_config` layered dataclass defaults < JSON
config < `hyperparams` < `eval_config` and never injected `exp["seed"]`, so the dataclass
default survived into `RESOLVED_CONFIG` → `TRAIN_CONFIG` → `run_summary.json`. Fixed by
adding `config["seed"] = exp["seed"]` as a final layer after all others, so no JSON config or
hyperparams block can shadow it. Checked the sibling runners: `ABI/common/experiment_utils.py`
already did this correctly (`config["RANDOM_STATE"] = exp["seed"]`), `PROGNOSER` passes
`random_state=exp["seed"]` directly in `build_experiment`, and BrainTokenGT imports
`build_config` from `CLASSIFIER.common.experiment_utils`, so it inherits the fix — no
gap found in any of the three sibling runners.

- Confirmed **no training code reads `TRAIN_CONFIG["seed"]`/`RESOLVED_CONFIG["seed"]`** — grepped
  every adapter and `model/*/train.py`, zero hits. Real seeding runs entirely off the
  papermill `SEED` parameter (`build_parameter_dict` already injected `exp["seed"]`
  correctly) → `set_seed(SEED)`. **This was a provenance-only defect; it invalidates no
  result**, confirmed again by the byte-equality re-gate below.
- **Did not rewrite the ~60 existing `run_summary.json` files** with the wrong field —
  back-editing them would destroy the provenance they exist to provide. Note the affected
  date range (pre-22 Aug) in the thesis Appendix instead.
- Added a regression test (`test_build_config_registry_seed_shadows_config_and_hyperparams`)
  proving a JSON config or hyperparams block claiming a different seed cannot win.

**Defect 2 — silent fallbacks in `CLASSIFIER/model/GELSTM/dataset.py`. Fixed.**

The `cohorts_csv` load wrapped everything in `except Exception: pass` — and the loaded
DataFrame was never read after that block regardless, so it was dead code doing nothing.
Removed rather than instrumented.

The second, worse one: the label-resolution chain (`converter_status` → `label` →
`diagnosis`) ended in a bare `else: label = 0` — a subject whose label column was missing or
misspelled was silently labelled non-converter. Now raises `ValueError` instead. Added a
regression test (`test_missing_label_columns_raises_rather_than_defaulting`).

**Verification, run twice:** `recon-ablation-gelstm-none` re-run at HEAD after both defects
were fixed, then again after the Defect 1 fix landed separately — `run_summary.json`
bit-identical to the 19 Aug baseline both times (test AUC 0.7607142857142857, all 5 fold
AUCs, threshold unchanged). Both fixes and the seed-provenance fix are confirmed true no-ops
on DELCODE model output; only the recorded metadata changed. Full test suite (556 passed, 2
skipped) and `ruff check` clean on every touched file.

Committed as `b7e2560` (D1 wiring + both defect fixes) with a follow-up commit for the
seed-provenance fix in `experiment_utils.py`.

---

## Phase 1 — one diagnostic, before any other external work

**Run a label-free positive control on the FC matrices themselves.**

Predict **sex** (and secondarily age) from each cohort's baseline FC, with the same
subject-level splits, using a plain classifier — logistic regression on the vectorised upper
triangle is enough; this is a data check, not a modelling exercise. Sex is strongly and
reliably decodable from functional connectivity, so it is a floor test that depends on
**neither the conversion label nor any domain-shift argument**.

Why this and not the A.5 zero-shot arm first: zero-shot confounds feature quality with domain
shift, so a null result there tells you nothing you can act on. This separates them.

Run it on **all three cohorts** — DELCODE is the reference line, and without it the ADNI
number is uninterpretable.

**Pre-register the decision rule before looking at the output:**

| DELCODE sex-AUC | ADNI / OASIS-3 sex-AUC | Reading | Go to |
|---|---|---|---|
| high | comparably high | Features carry real signal. The at-chance conversion result is **a finding**. | Branch A |
| high | at chance | **The A.3 extraction is broken for these cohorts.** Every external number is void. | Branch B |
| at chance | at chance | The probe or its harness is wrong — fix the probe, not the pipeline. | re-run probe |

Cost: well under a day, no GPU. **Do not write a single sentence about the external result
until this returns.**

**Result, 23 Aug — Branch A fires.** `DATA/manifest/probe_sex_decoding.py`, 5-fold CV
logistic regression on the vectorised upper triangle of baseline FC:

| cohort | n | sex AUC (5-fold CV) |
|---|---|---|
| DELCODE | 167 | 0.6191 ± 0.0691 |
| ADNI | 192 | **0.7131 ± 0.0883** |
| OASIS-3 | 60 | 0.5368 ± 0.1129 |

ADNI decodes sex *better* than DELCODE (the reference line) — the A.3 extraction is not
broken for ADNI, so the at-chance conversion result is a genuine finding, not a bug. OASIS-3
sits between chance and DELCODE with a wide CI at n=60; read as underpowered, not as a
second data point against ADNI's result. DELCODE's own 0.619 is lower than sex-decoding
AUCs typically reported in the literature (usually 0.8+), so the absolute floor this probe
sets is soft — the reading that carries weight is the **relative** one the decision rule was
built on (no ADNI-specific fault), not an absolute signal-quality claim.

---

## Phase 2 — branch on the answer

### Branch A — features are fine, the result is real

The at-chance external performance becomes a reportable finding, and it is a good one: it
belongs to the same argument as the variance decomposition. Then, in value order:

1. **Metadata floor on ADNI and OASIS-3.** Without it there is no reference line saying 0.50
   is bad rather than typical for these cohorts. Cheap, CPU-only, and it is the same probe
   harness Phase 1 just built.
2. **Zero-shot DELCODE→ADNI** with the frozen DELCODE threshold — the missing half of the
   A.5 dual gate. Now interpretable, because Phase 1 excluded the feature explanation.
3. **Write the per-cohort tables with degeneracy counts.** Half the GELSTM external runs
   predict every subject positive; a mean that hides that is the failure this thesis audits
   in other people's work.
4. **Do not** run the Δt ablation on ADNI. Ablating a Δt input inside a model that is at
   chance measures nothing. The Δt *contribution* stays a DELCODE-only claim; the Δt
   *interval-distribution* table (D5) stands on its own and ships in Ch. 3 regardless.

### Branch B — features are broken

1. **Cut the external block from the thesis.** Ch. 6 reports DELCODE only; Ch. 9 records the
   wiring as complete, the extraction as suspect, and names the probe that found it.
2. **Do not attempt to fix A.3 before 26 Aug.** It is an extraction-pipeline bug of unbounded
   scope, four days from the freeze, and the thesis does not depend on it.
3. The D5 interval table **still ships** — it comes from the manifests and `allowed_days`,
   not from the FC matrices, so a broken extraction does not touch it.

Either branch keeps the Ch. 3 Δt argument and the Ch. 5 baseline audit whole. That is the
point of running the probe rather than guessing.

---

## Phase 3 — DELCODE-side work, independent of both branches

**This is the part that is actually on the 28 Aug critical path.**

1. ~~D2 byte-equality re-gate.~~ **Done, 22 Aug 22:46 — passes.** Re-ran
   `recon-ablation-gelstm-none` at HEAD; `run_summary.json` is bit-identical to the 19 Aug
   baseline (test AUC 0.7607142857142857, all 5 fold AUCs, threshold). Re-run again after the
   two Phase 0 defect fixes landed (commit `b7e2560`) — still bit-identical. **Every DELCODE
   number in Ch. 6 is gate-cleared.**
2. ~~`recon-ablation-gelstm-pretrained-finetuned-seed45` at HEAD.~~ **Done, 22 Aug 22:58.**
   Test AUC 0.8536. The single-commit-state table for the fine-tuned arm is complete
   (`STABILITY_AUDIT.md` Finding 2): mean 0.6924 ± 0.2000 (ddof=1) across seeds 42–45, range
   [0.4250, 0.8536] — between-seed variance dominates, mirroring BrainTokenGT's profile in
   the opposite direction.
3. **Phase B1/B2/B3 notebook corrections.** Still open, still never cut — the invalid
   *p*-values must not reach the supervisor. B1 must additionally **filter globbed runs by
   `git.short_commit`**, or the glob-everything fix reintroduces the bug in a new form. This
   is now the only item left in Phase 3.

Phase 0's two defects were also fixed and committed the same session (`b7e2560`): the dead
`cohorts_csv` load block in `GELSTM/dataset.py` was removed (it read a CSV and never used the
result), and the label-resolution chain now raises on an unmatched subject instead of
silently defaulting to non-converter. Both re-verified as no-ops on DELCODE via item 1 above.

---

## Explicitly not now

- **Fixing the A.3 extraction**, if Branch B fires. Unbounded, four days out.
- **The LR-scaled fine-tuning arm.** Still unimplemented (`adapters/gelstm.py:274` is one
  Adam group). It cannot change a claim before hand-off; narrow the claim in words instead.
- **Δt ablation on the external cohorts.** See Branch A item 4.
- **Rewriting the ~60 existing `run_summary.json` files** to correct the seed field.
- **Any new registry entries.** The registry has 24 external experiments already; nothing
  below the freeze needs a 25th.

---

## Supervisor cross-cohort request — logged 23 Aug

Five items came in as a cross-cohort training/testing strategy plus a fine-tuning-instability
explanation. Four of the five were already carried somewhere in this directory; **one —
supervised multi-cohort pooled training — was not, and is specified below.** Coverage audit
first, so nothing gets written twice:

| Item | Already in `DOCS/timeline/`? | Where |
|---|---|---|
| **A.1 Zero-shot external generalization** (train DELCODE → test ADNI & OASIS-3, frozen DELCODE threshold) | ✅ yes, as a planned action | Phase 2 Branch A item 2 above; `SUBMISSION_RUNWAY.md` §"Two things the gate needs"; `comparison-plan-v2.md` §3 A.5 (the unrun half of the dual gate) |
| **A.2 Multi-cohort pooled training** (train DELCODE + ADNI → test OASIS-3) | ❌ **no** | Only *unsupervised* pooling existed (`comparison-plan-v2.md` Phase E: pooled GAAE pretraining pool). Phase C names "pooled leave-one-cohort-out" as a third claim in one clause and never defines or registers it. **Spec added below.** |
| **B.1 Naive fine-tuning at a shared LR** (encoder + head both at 1e-3) | ✅ yes, as the governing caveat | `STABILITY_AUDIT.md` Finding 2 point 4 (`adapters/gelstm.py:274`, one Adam group; contrast with BrainTokenGT's `give_lr_scale: 0.1`); Phase 3 item 3 / `SUBMISSION_RUNWAY.md` Phase B2 carry the prose fix |
| **B.2 Capacity vs sample size** (≈966 k weights, N ≈ 76 per fold) | ✅ yes | `SOTA_POSITIONING.md` §1–§2 (small-N framing, "the graph encoder does not earn its place"); `comparison-plan-v2.md` §2 |
| **B.3 Pretext–downstream objective mismatch** (reconstruction vs binary conversion) | ✅ yes | `SOTA_POSITIONING.md` §2.3 (the Brain-JEPA row is exactly this argument); thesis `08_discussion.tex` |

Two writing-lane items in the request are already tracked and must **not** be opened as new
tasks: the fine-tuning reframing is `SUBMISSION_RUNWAY.md` **Phase B2** (still open), and the
19-run variance decomposition beside GELSTM's byte-reproducibility is **Phase B1** (done,
23 Aug — the notebook reports within-seed SD 0.090 vs between-seed 0.068 over the 18 runs it
keeps after the commit filter; `STABILITY_AUDIT.md` Finding 1's 0.1011 / 0.0706 is the
unfiltered 19-run figure. Both are correct; quote the filtered pair in the notebook and the
19-run pair in the audit, and say which population each is over). The LR-scaled fine-tuning
arm stays where `SUBMISSION_RUNWAY.md` put it — out of scope before hand-off, a stated
limitation instead.

### A.2 — multi-cohort pooled training (train DELCODE + ADNI → test OASIS-3)

**Status: not started, not registered, and explicitly below the 26 Aug freeze.** It is a
post-hand-off / paper-runway item (Ch. 9 + paper), recorded here so it is a decision rather
than an omission. Nothing about it can change a number in the handed-off draft.

What already exists:

- **Time unification is done.** `CLASSIFIER/common/visits.py::visit_identity` converts each
  cohort's native time to cumulative months — DELCODE protocol months pass through unchanged
  (the A.2 byte-equality gate), ADNI/OASIS-3 elapsed days become `days / DAYS_PER_MONTH` —
  and `GELSTM/dataset.py:199` divides inter-visit deltas by `MAX_INTERVAL_MONTHS = 108`. That
  wiring landed in `b7e2560` (D1). A pooled loader inherits a consistent month scale for free;
  no new normaliser is needed.
- Per-cohort manifests and splits (`DATA/manifest/build_*_manifest.py`,
  `build_cohort_splits.py`), with zero subject overlap across cohorts.

What is missing, in order:

1. **A pooled split.** `GELSTMDataset` takes a single `cohort` string and one split CSV.
   Pooling needs either a `cohorts: [delcode, adni]` list or a pre-built pooled split CSV
   carrying a `cohort` column per subject. Build it from the *downstream train/val* subjects
   of both cohorts only — DELCODE's held-out test subjects must not leak into a pooled
   training pool that later gets compared against DELCODE-only numbers.
2. **A site/cohort covariate decision, made explicitly and recorded.** Pooling two cohorts
   with different scanners, intervals (DELCODE 90.0% at 12 mo vs ADNI 21.9%) and converter
   base rates (DELCODE 34.0% vs ADNI 33.9%) invites the model to learn cohort identity. Either
   pass cohort as an input feature or hold it out — do not leave it implicit — and report a
   cohort-decoding probe on the pooled representation the way Phase 1's sex probe was run.
3. **Registry entries** in `CLASSIFIER/experiments/external_validation.yaml`, arms `none` and
   `pretrained_frozen`, seeds 42–45, evaluating on the **held-out OASIS-3 test split**.
4. **Reporting rule, inherited unchanged:** OASIS-3's test split is n=13 (7 converters), so
   AUC moves in steps of 0.024. Pooled training does not enlarge the evaluation set — read the
   seed distribution, never a single point estimate, and keep the degeneracy count
   (all-positive predictions) beside every mean, per Branch A item 3.

**Precondition — do not run this before the zero-shot arm (A.1).** Zero-shot DELCODE→ADNI is
the interpretable baseline that pooled training has to beat; without it, a pooled result has
nothing to be compared against and repeats the confound Phase 1 was built to remove. Phase 1
has already excluded the feature-quality explanation, so the ordering is A.1 → metadata floor
→ A.2.

**Note the reporting rule this does not violate.** `SUBMISSION_RUNWAY.md` and
`SOTA_POSITIONING.md` §4.7 say OASIS-3 is "never pooled with ADNI" — that is a rule about
pooling *metrics* across cohorts into one mean that hides cohort variance. Pooling *training
data* and testing on a single named held-out cohort is a different design and is compatible
with it. Do not read one as licence for the other.
