[← §2 — Status dashboard](SECTION_02_STATUS_DASHBOARD.md) | [Master Plan](MASTER_PLAN.md) | [§4 — Diagnostic track: Phases 0–3, and the branch logic →](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md)

---

# §3 — Execution track: Phases A–F

Each phase below: what it was *for*, why it belongs at that point in the sequence, what it
produced, and current state.

## Phase A — ADNI/OASIS-3 to FC matrices, manifest-first

**Purpose:** get from raw scans to usable feature matrices without repeating the two
silent-count bugs (below) that plagued the first attempt. Manifest-first means every
downstream step consumes one asserted CSV instead of re-globbing the filesystem.

### A.0 — Build the manifest before touching any model code — done, 21 Aug

`DATA/manifest/` (`build_adni_manifest.py`, `build_oasis3_manifest.py`,
`build_delcode_manifest.py`, `build_cohort_manifest.py`, `schema.py`, tests), wired into
ratcheted CI. One `cohort_manifest.csv` per dataset with columns `cohort, subject_id,
session_id, days_from_baseline, visit_index, protocol_month, delta_t_months, bold_path,
fc_path, label, scanner_vendor, scanner_model, site`, generated once and asserted:

- every listed path exists and is non-empty (kills the empty-dir bug class below)
- every subject dir contributes ≥1 session
- per-cohort totals match expected counts
- `visit_index` contiguous from 0 per subject; `delta_t_months` strictly increasing
- no subject in both converter and stable CSVs

**Why this had to come first:** two separate "count the directory, not the content" bugs had
already cost days. `monitor_flatten_progress.sh` counted directory existence and reported
"128/142 settled" while 46 of those 128 OASIS-3 directories were empty shells (fMRIPrep
output existed, postprocessed BOLD did not). A second, unrelated bug in the same family: a
reorientation-affine error (`final_reorient.py`'s `affine[:, 0] *= -1` negated only the
direction-cosine column and never recomputed the translation) shifted every ADNI/OASIS-3
volume ~192mm out of atlas alignment, silently. The manifest's build-time assertions are the
structural fix — every count is now defined as *files matching the expected content
pattern*, not directory existence.

### A.1 — Cohort-aware visit parsing — helpers done 21 Aug, wired 22–23 Aug

**The design problem this solves.** DELCODE's `_M12_` is a nominal protocol month — a small
discrete scheduled-visit label, and the basis of `allowed_months` leakage filtering. ADNI/
OASIS-3's `ses-d0381` is continuous elapsed days from baseline. Collapsing elapsed days onto
the nearest protocol month throws away the irregularity the Δt-conditioning mechanism exists
to exploit; keeping it continuous breaks the allow-list filter that prevents leakage. This is
exactly the failure mode Chantal (supervisor) named as her fairness criterion 2: training on
uniform indices and testing on irregular ones without noticing.

**Resolution — split the conflated concept into three fields, one code path for all cohorts:**

| Field | Type | Used for | DELCODE | ADNI / OASIS-3 |
|---|---|---|---|---|
| `visit_index` | int, 0-based | ordering, window selection | rank of `M<n>` | rank of `d<n>` |
| `protocol_month` | int \| None | `allowed_months` leakage filter | `M<n>` | nearest scheduled visit |
| `delta_t_months` | **float** | model input (`use_time_delta`) | `M<n>` diff | `days_diff / 30.44` |

Implemented in `CLASSIFIER/common/visits.py` (`parse_day`, `parse_adni_protocol_month`,
`visit_identity()`, commit `87cf03c`). Wired into `GELSTM/dataset.py` and threaded through
`adapters/{gelstm,gec,gep}.py` + `BRAINTOKENGT/adapter.py` as a typed `cohort` field
(default `"delcode"`, never inferred silently) — commit landed 22 Aug, two follow-up fixes
committed 23 Aug as `b7e2560`:

1. A dead `cohorts_csv` load wrapped in `except Exception: pass` — the loaded DataFrame was
   never read afterward regardless, so it did nothing. Removed.
2. A worse one found alongside it: the label-resolution chain
   (`converter_status → label → diagnosis`) ended in a bare `else: label = 0` — any subject
   with a missing/misspelled label column was silently marked non-converter. Now raises
   `ValueError`, per `.claude/rules/errors.md`. Both regression-tested.
3. A third, separate defect found in the same pass: `experiment_utils.py::build_config`
   never injected `exp["seed"]`, so every `run_summary.json`'s recorded
   `training_config.seed` was the dataclass default (42) regardless of the real seed used.
   Confirmed provenance-only — no training code reads that field, real seeding runs off the
   papermill `SEED` parameter. Fixed by layering the registry seed last in `build_config` so
   nothing can shadow it (see §1's standing rule on trusting this field).

### A.2 — DELCODE reproduction gate *(hard gate)* — passed, 21 Aug

Re-ran a saved GEGRU DELCODE evaluation at the same seed/checkpoint/hyperparameters. Result:
**AUC 0.7929**, not the previously-cited 0.8321.

**This is not a regression from A.0/A.1.** Commit `87cf03c` (the visit-identity helpers) is a
pure addition — `GELSTM/dataset.py` didn't import from it yet at the time this gate ran, so
the new code was provably dead on the DELCODE path. Root cause traced instead to commit
`0823ca6` (4 Jul, "month-based visit filtering to prevent temporal leakage"), which postdates
every run that produced 0.8321. That commit made `GELSTM/dataset.py` drop any visit outside
`allowed_months` — closing a **label leak**: converter patients' post-conversion (already
demented) scans had previously been included in the eval trajectory. **0.8321 was leaky;
0.7929 is the corrected number after that leak closed.**

**Decision:** retire 0.8321, adopt 0.7929 as the DELCODE baseline, treat A.2 as passed. This
is a downward correction from closing a bug, not a live regression — does not block A.3/A.4.
Flag to Chantal since 0.8321 is cited elsewhere (older cross-dataset-drift notebooks).

Re-verified a further two times after A.1's wiring landed and after the two Phase-0-adjacent
defect fixes: `run_summary.json` **byte-identical** to the 19 Aug baseline all three times
(test AUC 0.7607142857142857, all 5 fold AUCs, threshold unchanged). Every DELCODE number in
Ch. 6 is gate-cleared.

### A.3 — Schaefer-200 extraction → FC, both cohorts — done, 22 Aug

**674 FC matrices / 268 subjects (ADNI)**, **234 / 128 (OASIS-3, post-dedup)** — `fc_path`
populated for every manifest row, `--require-fc` clean on both.

Extraction was blocked until the reorientation-affine bug (A.0's second bug class) was found
and fixed: compensating the translation term (`affine[:3,3] += x_dir*(nx-1)`) and threading
`--overwrite` through the reorient call so already-flattened corrupted files got redone.
ADNI's straggler gap (31 subjects/108 sessions denoised but stuck unflattened, same bug
family as the OASIS-3 empty-dir case) closed the same evening, landing on 268/272 eligible
ADNI subjects (4 excluded by motion QC).

### A.4 — Splits, both cohorts — real run done 22 Aug

`DATA/manifest/build_cohort_splits.py` reimplements DELCODE's stratified 60/20/20 protocol
locally (keeping the DELCODE-side module frozen for A.2's gate), using cohort-native columns
(`subject_id`/`allowed_days`, not `Pseudonym`/`allowed_months`).

| Cohort | train | val | test | total | converters |
|---|---|---|---|---|---|
| **ADNI** | 115 (76/39) | 38 (25/13) | 39 (26/13) | **192** | **65 (33.9%)** |
| **OASIS-3** | 35 (17/18) | 12 (6/6) | 13 (6/7) | **60** | **31 (51.7%)** |

**⚠️ ADNI 192/65 is authoritative.** An earlier dry run against a stale 237-subject ADNI
manifest produced 162/51 — that figure appears in some archived-doc sections and is
superseded everywhere by the 192/65 figure above (the corrected 268-subject manifest yielded
30 more eligible subjects). OASIS-3's 60/31 is unchanged and confirmed. Zero split overlap in
both cohorts.

Interval structure — this is deliverable **D5**, and the entire payoff of the A.1
protocol-month-vs-elapsed-time design decision:

| Cohort | scans/subj | median interval | IQR | CV | modal month bucket |
|---|---|---|---|---|---|
| DELCODE | — | 12 mo | — | — | **90.0%** at 12 mo |
| ADNI | 3.11 (2–10) | 371 d | 207–419 d | **0.647** | 21.9% at 12 mo |
| OASIS-3 | 2.65 (2–6) | 1012 d | 708–1342 d | **0.574** | 8.1% at 36 mo |

This table stands **independently of whether any classifier works** — it's a property of the
cohorts, not the models. Ships in Ch. 3 regardless of how the diagnostic track resolves.

### A.5 — Transfer gate, with a positive control — half-run, superseded by Track 2

Original design: run both a within-ADNI 5-fold CV and a zero-shot DELCODE→ADNI arm, and read
them jointly (a genuinely at-chance result under both readings distinguishes "real domain
shift" from "pipeline bug," where a naive single-arm zero-shot reading cannot). The
within-ADNI control ran (0.54–0.58, at chance for GELSTM) but the zero-shot arm was never
registered — and BrainTokenGT scoring 0.705 on the identical splits meant A.5's own table had
no row to classify the result into. **See §0 and §4 — Track 2 was opened to resolve this
instead, and its conclusion supersedes A.5's inference.**

## Phase B — DELCODE fairness work (C1/C2/C3), matched-cohort comparison

**Purpose:** make the GELSTM-vs-BrainTokenGT comparison defensible under Chantal's three
named fairness criteria, independent of and in parallel with Phase A:

- **C1 — same dataset & splits.** Every arm consumes the identical split CSVs by path, hashed
  into the run summary.
- **C2 — consistent processing within a model.** The shared visit-window helper (A.1) is what
  structurally prevents "uniform in train, irregular in test."
- **C3 — decent optimization, documented.** An LR collapse screen (fixed grid, reject
  divergent/flat runs, select on validation only, same early-stopping rule for every arm,
  publish train/val curves). Not tuning — a floor. "No need for extensive hyperparameter
  tuning" is explicit supervisor permission to stop there.

**B.1 — GELSTM vs BrainTokenGT matched-cohort head-to-head.** This is C3 applied to the
model-comparison claim specifically, not just a single arm's collapse screen.

- **B.1.0 — gate: is BrainTokenGT's stabilization reliable?** Two mechanisms, often
  conflated: the literal cross-visit edges (`time_alignment()`, deterministic, already
  correct) vs GIVE/GRCU (EvolveGCN-H — a GRU whose hidden state *is* the GCN's weight
  matrix). As released, `train_give=False` means this module never trains (frozen at random
  init); enabling it caused `TopK.forward` to hit non-finite scores. A stabilization config
  (separate optimizer group, `give_weight_decay=0.001`, `give_lr_scale=0.1`) fixed the crash.
  **Answered by the stability audit — see §5, Finding 1: it stops crashing, but is not
  reproducible at fixed seed.**
- **B.1.1–B.1.3** — closing the cohort-window gap (add `min_visits` floor to GELSTM's dataset
  to mirror BrainTokenGT's fixed `min_visits=2, max_visits=3`, verified by diffing subject-ID
  lists between pipelines rather than assuming), confirming folds are paired (not just
  same-seed), registering matched-cohort experiments at seeds 42–45 for both models.
- **B.1.4–B.1.5** — run both, compare with a paired test if B.1.2 confirms paired folds,
  write up in a dedicated comparison doc with the fairness checklist stated explicitly.
  **Done, 23 Aug — see §5 and §8 (Phase B1).**

**Context for why this matters beyond "who wins":** the pooled 4-seed reconstruction-value
ablation already shows GELSTM's own encoder isn't earning its place on DELCODE (`none` mean
test AUC 0.831 vs `pretrained_frozen` 0.781, `pretrained_finetuned` collapsing at seed 43).
That's an in-family finding, not evidence about BrainTokenGT — B.1 is the only controlled way
to test whether the same pattern holds across models, since BrainTokenGT never had a
reconstruction objective to ablate.

## Phase C — ADNI head-to-head

**Status: run, 22 Aug.** 12 runs (3 arms × 4 seeds), all at chance on held-out test — see §6.
At n=192 the *cohort* is ~5× DELCODE's, but the **held-out test split is only n=39** (13
converters), so the evaluation set is barely larger than DELCODE's test split and CIs are
correspondingly wide — report per-cohort, not pooled.

## Phase D — OASIS-3 probe, demoted by design

**Status: run, 22 Aug, alongside ADNI (not after, as originally planned).** 12 runs. Test
split is **n=13 (7 converters)** — 42 label pairs, so AUC moves in steps of 0.024. A single
OASIS-3 test AUC carries almost no information; only the seed distribution is readable, and
barely. Two honest uses:

1. **Unlabeled pretraining data** — all 128 subjects × 239 sessions feed GAAE pretraining
   regardless of conversion labels.
2. **Secondary robustness probe** — single-vendor, three scanner generations, report with
   CIs, at 31 converters this is closer to a real secondary arm than a purely qualitative one
   — but still explicitly underpowered relative to the ADNI decision. **Never pooled with
   ADNI into one mean** — that would hide cohort-level variance, which is exactly the failure
   this thesis audits in other people's work.

## Phase E — pretraining-scale ablation

Unchanged in intent, depends on nothing above resolving. The honest unlabeled pool is
DELCODE + ADNI(237) + OASIS-3(128). `none ≈ pretrained_frozen` is now a **live finding**, not
a hypothesis (§6) — cite regardless of how Track 2 resolves.

## Phase F — write-up

Per-cohort tables (never pooled means hiding cohort variance), CIs everywhere, degeneracy
flags, the C1/C2/C3 fairness checklist, label-harmonization sensitivity analysis, honest
BrainTokenGT convergence reporting with curves. Feeds Ch. 6 directly.

---

[← §2 — Status dashboard](SECTION_02_STATUS_DASHBOARD.md) | [Master Plan](MASTER_PLAN.md) | [§4 — Diagnostic track: Phases 0–3, and the branch logic →](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md)
