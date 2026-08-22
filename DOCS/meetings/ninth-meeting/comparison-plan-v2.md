# Model-comparison plan v2 — external validation

**Supersedes** the previous cross-dataset plan. Two inputs forced the revision:
Chantal's definition of a fair comparison (Slack, Aug 20) and a recount of what
is actually on disk after the `flatten_fmriprep.sh` glob fix.

Verified against disk on **2026-08-20 ~21:15**, re-verified **~21:30** — counts
unchanged at that point. **Re-verified again 2026-08-21 ~10:45** — counts
*have* moved since (§1.3a). All counts below are reproducible from the commands
in §7.

> **Which product the counts refer to.** Two flat products exist per cohort and
> they are *not* interchangeable. FC extraction consumes the **postprocessed**
> one (`__fmri_wholebrain_sch200_flat__`). The job feeding
> `__fmriprep_wholebrain_flat__`, one step upstream, is still landing new ADNI
> subjects; the OASIS-3 postprocessed product moved too, but because a bug got
> fixed, not because new data arrived — see §1.3a.
>
> ```
> ADNI    __fmriprep_wholebrain_flat__     dirs=268 empty= 0 sessions=675
> ADNI    __fmri_wholebrain_sch200_flat__  dirs=237 empty= 0 sessions=567
> OASIS3  __fmriprep_wholebrain_flat__     dirs=128 empty= 0 sessions=239
> OASIS3  __fmri_wholebrain_sch200_flat__  dirs=128 empty= 0 sessions=239
> ```
>
> Consequence for ADNI: the postprocessed count (237) hasn't moved, but the
> fmriprep-flat count grew from 250→268 since 2026-08-20, so the pool of
> subjects waiting on a postprocessing pass before FC extraction can see them
> is now **31, not 13** (§3, A.3) — the `monitor_flatten_progress.sh` WARN of
> "35 no dir" for the postprocessed-flat job is consistent with this: those
> subjects have fmriprep output but the postprocessing job that would produce
> their flat product has already exited. The CORE→Fritz pull for
> `ADNI/postprocessed` is still "in progress" per
> `monitor_flatten_progress.sh --once` (2026-08-21 10:43), so more of that gap
> may close once that pull finishes and a postprocessing pass is re-run — don't
> assume it closes on its own, budget the pass explicitly.
>
> Consequence for OASIS-3: **the empty-dir bug is fixed** — see §1.3a. The
> remaining 14 "no dir" subjects (matching `monitor_flatten_progress.sh`'s WARN
> for both OASIS-3 products) are a distinct, still-open gap: fully missing
> subjects, not empty shells.

---

## 1. What changed since v1

### 1.1 Chantal's fairness bar is narrower than v1 assumed — that's good news

> - test on the same dataset & splits
> - training, val, test processing should be the same for one model (so don't
>   train with uniform indices and test with irregular ones e.g.)
> - all models should be decently optimized (don't continue training if the model
>   is clearly overfitting, don't use a lr where training collapses or does not
>   learn at all etc. — but no need to do super extensive hyperparameter tuning,
>   that's just not realistic)

v1 treated "fair comparison" as an open-ended research problem and treated
*getting BrainTokenGT to converge* as the risk you control least. Chantal's
criterion 3 dissolves that. You are not obliged to make the baseline win, or to
make it optimal. You are obliged to show it was **not sabotaged**: no collapsed
LR, no training run past obvious overfitting. That is a bounded, checkable,
documentable obligation — see the adequacy protocol in §4.

Her criterion 2 is not generic advice. "Don't train with uniform indices and test
with irregular ones" is *precisely* the visit-window issue v1 identified, which
means the shared visit-window helper is now supervisor-mandated rather than
self-imposed polish. Promote it.

Her criterion 1 has a consequence v1 glossed over: **if the headline moves to
ADNI, every arm in the comparison must be run on ADNI.** A cross-dataset story
does not license comparing our model on ADNI against BrainTokenGT on DELCODE.

### 1.2 The dataset numbers are materially worse than v1 claimed — mostly OASIS-3

*Updated 2026-08-21 — the OASIS-3 row moved; see §1.3a.*

| | v1 claimed | actual (2026-08-20) | **actual (2026-08-21)** | delta vs. v1 |
|---|---|---|---|---|
| ADNI, ≥2 sessions, in cohort CSVs | 176 (55 conv / 121 stable) | 162 (51 conv / 111 stable) | **162 (51 conv / 111 stable)** | −14, unchanged since 08-20 |
| OASIS-3, ≥2 sessions, in cohort CSVs | 72 (33 conv / 39 stable) | 40 (19 conv / 21 stable) | **60 (31 conv / 29 stable)** | **−12 (−17%), up from −32** |
| Combined external, ≥2 sessions | ~248 | 202 | **222** | −26, up from −46 |

ADNI is close enough that the v1 argument survives, and is stable — the
postprocessed-flat product for ADNI hasn't moved since 08-20 (§1 callout). The
OASIS-3 shortfall is smaller than it looked on 08-20: fixing the empty-dir bug
(§1.3a) recovered 20 more ≥2-session subjects (12 converters), bringing it from
"cannot carry a validation claim" territory toward "same order of magnitude as
ADNI's shortfall, still smaller in absolute n." §5's Phase D framing is revised
accordingly — read it with the caveat that 14 OASIS-3 subjects with no
directory at all are still unaccounted for and could move this again.

### 1.3 Root cause of the OASIS-3 shortfall (as of 2026-08-20): another count-the-directory bug

Same family as the `.html` glob you just fixed, different step:

```
OASIS3 fmriprep-flat : 128 subject dirs,  0 empty, 239 sessions
OASIS3 postproc-flat : 128 subject dirs, 46 empty, 152 sessions
```

46 subjects have fMRIPrep output but **zero** postprocessed BOLD — and the
directory was created anyway. `monitor_flatten_progress` counts directory
existence, so it reports `128/142 (90%) settled` while 46 of those 128 are empty
shells. The real usable OASIS-3 denominator was **82 subjects, not 128**.

**Generalized lesson, now applied twice:** progress counters that glob paths
count artifacts of the filesystem, not data. Every count in this plan is
defined as *files matching the expected content pattern*, and Phase A.0 makes
that structural.

### 1.3a Update, 2026-08-21 — the 46-subject triage is done, and it fully recovered

The triage §5 flagged as "the highest-value-per-hour item in the whole plan"
has been run:

```
OASIS3 fmriprep-flat : 128 subject dirs,  0 empty, 239 sessions
OASIS3 postproc-flat : 128 subject dirs,  0 empty, 239 sessions   <- was 46 empty, 152 sessions
```

The postprocessed product now matches the fMRIPrep product **exactly** — same
dir count, same session count, zero empty shells. All 46 subjects were
recoverable, not a genuine exclusion; this drove the §1.2 OASIS-3 update (40 →
60 subjects with ≥2 sessions, 19 → 31 converters).

**Resolved, 2026-08-21 (same day):** the 14 no-dir subjects
(`monitor_flatten_progress.sh --once`: both OASIS-3 rows reported "128/142
(90%) remaining 14 (14 no dir, 0 empty dir)") are a different failure mode
from the one just fixed — these subjects never landed on disk in the first
place, not an empty-shell artifact — but they're not a pending-transfer gap
either. Diffing `derivatives/fmriprep` against `__fmriprep_wholebrain_flat__`
(both OASIS-3 products give the identical 14-subject set, confirming this is
one root cause, not two) and reading `flatten_fmriprep_20260820_205831.log`
for each of the 14 gives a definitive, per-subject cause:

- **12 — legitimate motion-QC exclusion.** Every session for the subject had
  mean FD > 0.5mm, so nothing passed QC to flatten:
  `OAS30095, OAS30218, OAS30224, OAS30428, OAS30675, OAS30686, OAS30717,
  OAS30733, OAS30862, OAS30961, OAS30978, OAS31089`. Same threshold applied to
  every other subject in the cohort — not a bug, not recoverable, correctly
  excluded.
- **2 — fMRIPrep never emitted confounds, so QC couldn't even run:**
  `OAS30797`, `OAS31416` (`ERROR: no *_desc-confounds_timeseries.tsv under
  .../func/ — fMRIPrep sometimes fails to emit confounds; cannot QC this
  subject`). These subjects do have a source directory with `func/` output,
  just no confounds TSV — this reads as a genuine upstream fMRIPrep gap
  rather than a CORE→Fritz transfer artifact (the pull for OASIS3/fmriprep
  reported complete). Not yet spot-checked against what CORE actually
  produced; low priority given it's 2 subjects out of 128, but flag it if an
  fMRIPrep re-run pass ever happens for other reasons.

**Consequence: none for the denominator.** All 14 were already excluded from
the 128 landed-subject count reported throughout this doc (§1.2, §7) — this
was never a pending recovery like the 46 empty-shells; it just confirms 128
is the correct, settled OASIS-3 denominator with a documented reason for
every excluded subject, not a placeholder that could still move.

### 1.4 The session-naming bug is confirmed, not predicted — and still live, 2026-08-21

v1 flagged this as a risk. It is real and I can point at the lines. The
cohort-aware helper functions that fix it now exist (§3, A.1) but are not yet
wired into the model dataset classes below — see A.1's update. Don't assume
this is closed.

`CLASSIFIER/common/visits.py:23` — `_MONTH_RE = re.compile(r"_(M\d+)_")`

| cohort | filename | `parse_month` |
|---|---|---|
| DELCODE | `sub-011d501d1_ses-01_M0_task-rest_...` | `0` ✓ |
| ADNI | `sub-ADNI002S2043_ses-d0000_task-rest_...` | **`None`** |
| OASIS-3 | `sub-OAS30057_ses-d0075_task-rest_...` | **`None`** |

`CLASSIFIER/model/GELSTM/dataset.py:175-177` then does `if month is None:
continue` — so **every ADNI and OASIS-3 scan is silently discarded**, the subject
lands with zero visits, and the cohort vanishes without an error.
`visits.py:84` (`month_allowed`) fails the same way. Run ADNI through today's
loader and you get a clean, wrong, empty result.

---

## 2. The design decision v1 hand-waved: protocol month vs. elapsed time

v1 said ADNI/OASIS-3 "need a per-cohort visit-time parser." That understates it,
because the two cohorts encode a *different quantity*:

- **DELCODE** `_M12_` is a **nominal protocol month** — the scheduled visit
  label. It is a small discrete set (0/12/24/36/48/60) and it is what
  `allowed_months` set-membership filtering is built on.
- **ADNI / OASIS-3** `ses-d0381` is **actual elapsed days from baseline**. It is
  continuous and irregular. `round(381/30.44) = 13`, which is not in any
  DELCODE-shaped allow-list, so naive conversion re-creates the silent drop.

Collapsing elapsed days onto the nearest protocol month throws away the
irregularity that GELSTM's `use_time_delta` exists to exploit. Keeping it
continuous breaks the allow-list filter. **This is exactly Chantal's criterion 2
failure mode** — it is how you end up training on uniform indices and testing on
irregular ones without noticing.

**Resolution — split the two concepts, which are currently conflated:**

| concept | type | used for | DELCODE | ADNI / OASIS-3 |
|---|---|---|---|---|
| `visit_index` | int, 0-based | ordering, window selection | rank of `M<n>` | rank of `d<n>` |
| `protocol_month` | int \| None | `allowed_months` label-leakage filter | `M<n>` | nearest scheduled visit, or the cohort CSV's own visit code |
| `delta_t_months` | **float** | model input (`use_time_delta`) | `M<n>` diff | `days_diff / 30.44` |

`delta_t` becomes a float everywhere, including DELCODE, so **one code path
serves all three cohorts** — the single most important thing for satisfying
criterion 2. DELCODE's values happen to be integer-valued; nothing else changes
for it, and the existing DELCODE numbers must reproduce exactly (§3, gate A.2).

Anchoring data exists: `ADNI/__metadata__/adni_visit_baselines.csv` (3685 rows,
`subject_id,baseline_date,baseline_source`); OASIS-3's cohort CSVs already carry
`date_diff_days` and `d####` session codes directly.

---

## 3. Phase A — ADNI to FC matrices, manifest-first

### A.0 Build the manifest before touching any model code — **done, 2026-08-21**

`DATA/manifest/` now exists (`build_adni_manifest.py`, `build_oasis3_manifest.py`,
`build_delcode_manifest.py`, `build_cohort_manifest.py`, `schema.py`, plus
`tests/`), wired into the ratcheted CI checks
(`chore(ci): extend blocking/ratcheted checks to DATA/manifest`). Both external
`cohort_manifest.csv` files exist and their subject/label counts match §1.2's
2026-08-21 numbers exactly (ADNI: 237 subjects / 162 with ≥2 sessions, 111
stable + 51 converter; OASIS-3: 128 subjects / 62 with ≥2 sessions, 29 stable +
31 converter — manifest row counts are marginally lower than the raw
`.nii.gz`-glob counts in §7 for OASIS-3, 234 vs 239, presumably from a QC/dedup
filter — worth a note in the write-up rather than an inconsistency to chase).
The direct answer to two silent-count bugs in two weeks. One
`cohort_manifest.csv` per dataset, generated once, asserted, and consumed by
every downstream step so nothing re-globs:

```
cohort, subject_id, session_id, days_from_baseline, visit_index,
protocol_month, delta_t_months, bold_path, fc_path, label,
scanner_vendor, scanner_model, site
```

Build-time assertions that **fail loudly** rather than dropping rows:
- every listed path exists and is non-empty (kills the empty-dir class)
- every subject dir contributes ≥1 session (kills the `.html` class)
- per-cohort subject and session totals match §7's expected counts
- `visit_index` is contiguous from 0 per subject
- `delta_t_months` is strictly increasing per subject
- no subject appears in both converter and stable CSVs

Cost: an afternoon. It is cheap insurance and it is also the provenance table
the thesis needs anyway.

### A.1 Cohort-aware visit parsing — **helpers done, 2026-08-21; not wired in yet**

The §2 three-field split (`visit_index` / `protocol_month` / `delta_t_months`)
is implemented in `CLASSIFIER/common/visits.py` (`feat(classifier): add
ADNI/OASIS-3 cohort-aware visit identity`, commit `87cf03c`): `parse_day`,
`parse_adni_protocol_month`, and `visit_identity()`, with regression tests in
`CLASSIFIER/tests/test_visits.py`.

**Not done yet:** the consuming dataset classes still import the old,
cohort-blind functions — `CLASSIFIER/model/GELSTM/dataset.py:41` imports
`parse_allowed_months, parse_month` (not `visit_identity`), and
`CLASSIFIER/model/GEC/dataset.py:9` imports `allowed_months_map,
month_allowed` (same vintage). So §1.4's bug — every ADNI/OASIS-3 scan
silently discarded because `parse_month` returns `None` for `ses-d####`
filenames — **still reproduces if you run ADNI through the model dataset
classes today.** The new helpers exist and are unit-tested in isolation but
haven't been wired into `GELSTM/dataset.py` / `GEC/dataset.py` yet. Treat that
wiring as the remainder of A.1, not yet complete, before A.3/A.4 depend on it.

### A.2 DELCODE reproduction gate *(hard gate)* — **run 2026-08-21; 0.8321 target retired, not a regression**

Re-ran the saved GEGRU DELCODE evaluation
(`divine-ocean-4-74659c77b-2026-08-21_11-13-01`, commit `74659c77b`, seed 42,
same GAAE checkpoint `ethereal-planet-16_2026-06-10...` — 76 params loaded,
identical hyperparameters in `resolved_config.json` except for
`encoder_init`/`encoder_grad`, which are no-op fields added for the
reconstruction-value ablation). Result: **AUC 0.7929, not 0.8321.**

This is **not** a bug introduced by A.0/A.1. Both are provably innocent:
commit `87cf03c` (ADNI/OASIS-3 visit identity) is a pure-addition diff — it
only adds `parse_day`/`parse_adni_protocol_month`/`visit_identity` to
`visits.py` and does not touch `GELSTM/dataset.py` or `GEC/dataset.py`, which
still import the old `parse_month`/`allowed_months_map` functions unchanged
(A.1's own "not wired in yet" note above already says as much). The new
cohort-aware code is dead code on the DELCODE path and cannot have caused
this drift.

**Root cause: commit `0823ca6`, 2026-07-04, "feat: month-based visit
filtering to prevent temporal leakage."** It postdates every run that
reproduced 0.8321 (`magic-stream-1`/`upbeat-water-2`/`lucky-harbor-3`, all
2026-06-20–21) and is live in the pipeline now: the consumed split CSVs
(`DATA/DELCODE/__metadata__/SPLITS/downstream/{train,val,test}.csv`) carry an
`allowed_months` column, and `GELSTM/dataset.py:112-123` reads it and drops
any visit file outside that list. Per the commit's own stated purpose,
converter patients' post-conversion (already-demented) scans — previously
included — are now excluded from the trajectory. **The 0.8321 baseline was
measured with label leakage** (post-conversion scans in the eval trajectory);
**0.7929 is the corrected number after that leak was closed on 2026-07-04.**
No re-baseline run was taken between the July 4 fix and this one, so the old
target was stale, not a live gate.

**Determinism check (same-seed re-run):** in progress, 2026-08-21 — see
run status before treating 0.7929 itself as the adopted target end-to-end.

**Decision:** retire 0.8321 as the DELCODE reproduction target. Adopt 0.7929
(pending the determinism re-runs above) as the new baseline and treat A.2 as
**passed** — this is a downward correction from closing a leakage bug, not a
regression from the visit refactor, so it does not block A.3/A.4. Flag this
correction to Chantal, since 0.8321 may already be cited elsewhere (e.g.
`gegru-cross-dataset-drift-validation.md` §1, the `SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT`
and `COMPARISON_GEGRU_CROSS_DATASET` notebooks, and
`sanity-gegru-synthetic-scanner-drift`/`comparison-gegru-cross-dataset`
outputs, all of which anchor on 0.8321 and will need re-anchoring on 0.7929
once the determinism check confirms it).

### A.3 Schaefer-200 extraction → FC for ADNI **and OASIS-3**

*Done, 2026-08-22.* **674 FC matrices / 268 subjects (ADNI)**, **234 / 128
(OASIS-3, post-MB4-dedup)** — `fc_path` populated for every manifest row in
both cohorts, `--require-fc` passes clean on both.

Changes made (2026-08-21 wiring): `process_using_schaeffer_atlas.py` gained a
`--manifest` flag (mutually exclusive with `--fmri-root`) reading exact
`bold_path` rows via a new `DATA/manifest/load.py`, instead of re-globbing;
`build_adni_manifest.py` / `build_oasis3_manifest.py` gained `--fc-root` and
now populate `fc_path` (was hardcoded `None`); `schema.py` gained
`assert_fc_paths_present`, wired behind `--require-fc` on all three builder
CLIs. `build_delcode_manifest.py` and the split it feeds A.2 are untouched.

**Reorientation-affine bug (same family as §1.3's), found and fixed
2026-08-21:** every on-disk ADNI/OASIS-3 `_bold_reoriented.nii.gz` file had a
corrupted affine — translation `x=-96.5` where the correct
MNI152NLin2009cAsym-2mm origin is `+95.5` (DELCODE's flat product had the
correct value). Root cause: `DATA/PREPROCESSING/src/fritz/final_reorient.py`'s
`affine[:, 0] *= -1` negated only the direction-cosine column and never
recomputed the translation, shifting the whole field of view ~192mm outside
where the atlas expects the brain. Fixed by compensating the translation term
(`affine[:3,3] += x_dir*(nx-1)`); `flatten_fmriprep.sh` /
`postprocess_local.sh` threaded `--overwrite` through to the reorient call so
already-flattened (corrupted) files got redone.

**ADNI's straggler gap, closed the same evening:** the 2026-08-21
`postprocess_local.sh --dataset both --flatten-only --overwrite` run both
applied the affine fix and caught up the 31 subjects / 108 sessions that were
denoised but stuck unflattened (fmriprep-flat 268 vs postprocessed-flat 237,
same bug family as §1.3's OASIS-3 empty-dir gap) — landing on 268/272 eligible
ADNI subjects (4 excluded by motion QC), zero duplicate-same-day sessions.
`build_adni_manifest.py`'s `EXPECTED_SUBJECTS`/`EXPECTED_SESSIONS` were stale
at 237/567 (built from a manifest snapshot taken *before* that evening's run
completed) and needed bumping to 268/674 — see §7's 2026-08-22 re-run. Real
extraction was launched against the stale 237/567 manifest first, missing the
31 recovered subjects; re-running against the corrected 268/674 manifest
picked up exactly the missing 107 sessions (`skipped=567, processed=107,
failed=0`) with no wasted recomputation, since the extractor skips any BOLD
file whose `.npz` pair already exists.

### A.4 Splits — **ADNI and OASIS-3**

*Updated 2026-08-21 — generalized from ADNI-only; code done and validated,
real output pending A.3.* New `DATA/manifest/build_cohort_splits.py`
(+ `DATA/manifest/demographics.py` for the `sex`/`age` GELSTM needs, sourced
from ADNI's `__artifacts__/All_Subjects_PTDEMOG_*.csv` and OASIS-3's
`OASIS3_demographics.csv`) reimplements DELCODE's stratified 60/20/20
protocol (`create_downstream_data_splits.py::_stratified_split`, same seed
policy) locally rather than importing it, so that module stays frozen for the
A.2 gate. Cohort-native columns throughout (`subject_id`/`allowed_days`, not
`Pseudonym`/`allowed_months`) — aliasing them wasn't enough to make the CSVs
drop-in anyway, since GELSTM also filters `diagnosis ∈ {mci, converter}` and
parses month allow-lists; that consumer-side generalization is A.1-wiring's
remaining scope, not A.4's.

A dry run against the real manifests (bypassing the `fc_path` eligibility
gate, back when A.3 hadn't produced any yet) reproduced §1.2's counts exactly:
**ADNI 162 (111 stable / 51 converter)**, **OASIS-3 60 (29 / 31)**, zero
split overlap. `fc_path` is now fully populated for both cohorts (A.3, done
2026-08-22) — those counts were against the stale 237-subject ADNI manifest,
though, so the real run — `python -m DATA.manifest.build_cohort_splits
--cohort {adni,oasis3}` — still needs to happen against the corrected 268/674
ADNI manifest and hasn't been run yet.

### A.5 Transfer gate — with a positive control *(revised)*

v1's gate: "DELCODE-trained GELSTM scores above chance zero-shot on ADNI; if at
chance you have a bug, not a finding." **That inference is wrong**, and acting on
it would burn days debugging a pipeline that works. At-chance zero-shot has two
very different causes:

| within-ADNI 5-fold | zero-shot DELCODE→ADNI | conclusion |
|---|---|---|
| above chance | above chance | everything works — proceed to Phase C |
| **above chance** | **at chance** | **real domain shift — a finding, not a bug.** Proceed; report it. |
| at chance | at chance | pipeline or label-harmonization bug — stop and debug |

So run **both**, and read them jointly. The within-ADNI positive control is one
cheap 5-fold run of a single arm and it is what makes the gate diagnostic
instead of ambiguous. Pre-register the pass threshold before looking.

---

## 4. Phase B — DELCODE fairness work, re-scoped to Chantal's three criteria

Runs concurrently with Phase A; independent of it. Restructured so each item
maps to a criterion she named, which is also the structure of the thesis's
fairness checklist.

**C1 — same dataset & splits.** All arms consume the identical split CSVs by
file path, asserted at run start (hash the split file into the run summary). No
arm regenerates its own splits.

**C2 — consistent processing within a model.** The shared visit-window helper
(§2). One code path builds train, val and test windows; the window spec is
recorded per run. This is where "uniform in train, irregular at test" gets
structurally prevented rather than checked by eye.

**C3 — decent optimization, documented.** The **optimization-adequacy protocol**,
applied identically to every arm including BrainTokenGT:

1. LR sweep over a fixed small grid (e.g. `{1e-2, 3e-3, 1e-3, 3e-4, 1e-4}`),
   same grid for all arms. Not tuning — a collapse screen.
2. Reject any LR where train loss diverges or is flat over the first N epochs.
3. Select on **validation** metric, never test.
4. Early stopping already configured (`early_stopping_patience: 15`,
   `epochs: 50` in `gelstm_delcode_whole_brain.json`) — apply the same rule to
   all arms.
5. Save and publish the train/val curves for every arm.

Item 5 is what converts "we tried to be fair" into evidence. If BrainTokenGT
still underperforms after this, the curves *are* the defence — and Chantal's
"no need for super extensive hyperparameter tuning" is explicit permission to
stop there. Report the protocol; do not chase the baseline further.

Remaining v1 items unchanged: determinism check on BrainTokenGT, all four
encoder arms, Tier-C baselines.

### B.1 — GELSTM vs BrainTokenGT matched-cohort head-to-head (detail on the
"matched-cohort (2–3 visit) head-to-head not yet run" item in §8)

This is C3 applied specifically to the model-comparison claim, not just to a
single arm's LR/collapse screen. Right now none of "fair" holds: different
visit windows, and BrainTokenGT's temporal module either contributes nothing
(frozen) or hasn't been shown to converge reliably (unfrozen). Verified
against the live code on 2026-08-21 — file/line refs below are current, not
from the meeting transcript that first proposed this plan.

**B.1.0 — Gate: is BrainTokenGT's GRCU stabilization actually reliable?**
(do this first — it decides whether B.1.1–B.1.5 are worth running)

Two mechanisms get conflated and shouldn't be:

- The literal cross-visit edges — `time_alignment()` /
  `DHT()` in `BRAINTOKENGT/model/transformer.py:81-134`. Connects node *i* at
  visit *t* to node *i* at visit *t+1*, fixed weight, deterministic. Already
  correct, not the unstable part, nothing to fix here.
- GIVE/GRCU (EvolveGCN-H, `BRAINTOKENGT/model/grcu.py`) — a GRU whose hidden
  state *is* the GCN's weight matrix, evolved per visit. As released,
  `train_give=False` reproduces an upstream bug where the parameters never
  register, so this module is frozen at random init and contributes nothing
  learned (`model/transformer.py:49-52`). Turning it on (`train_give=True`)
  previously caused `TopK.forward` to hit non-finite scores
  (`model/grcu.py:132-147`).

A fix already exists in `configs/braintokengt_repaired_delcode_fix_stabilized.json`:
a separate optimizer param group for the GIVE/GRCU params —
`give_weight_decay: 0.001` (regularizes the recurrent weight matrix),
`give_lr_scale: 0.1` (slows its evolution rate). One run completed cleanly
under it, but the config's own comment attributes the prior divergence to
"GPU floating-point-order sensitivity, not data-dependent" — i.e. plausibly
nondeterministic — so one clean run isn't enough to trust.

Action: re-run `braintokengt-delcode-whole-brain-repaired-fix-stabilized` 2–3
more times at seed 42.
- Converges reliably every time → proceed to B.1.1.
- Diverges intermittently → that instability is itself a legitimate finding
  to report ("BrainTokenGT's temporal module does not converge reliably on
  this cohort size, even after gradient-scale/weight-decay stabilization"),
  and B.1.1–B.1.5 becomes "report that" instead of a clean head-to-head.

**B.1.1 — Close the cohort-window gap (GELSTM side, code change)**

`BRAINTOKENGT/configs/braintokengt_repaired_delcode.json` already sets
`"min_visits": 2, "max_visits": 3`. `CLASSIFIER/model/GELSTM/dataset.py` has
`max_visits`/`require_full_window` (lines 5-6, 65-72, 88-89, 98-99, 128-132,
160-163) but **no `min_visits` floor** — confirmed still true, 2026-08-21.

1. Add `min_visits: int | None = None` to the relevant GELSTM config
   dataclass (per `configs.md` — typed field with a default, not a loose
   kwarg).
2. Mirror BrainTokenGT's exact filter semantics in `GELSTM/dataset.py` (same
   `n_scans >= min_visits` keep-rule, same truncation order as
   `window_item` in `BRAINTOKENGT/model/sequences.py:92`) — don't just add a
   floor with different tie-breaking.
3. Wire it through `CLASSIFIER/adapters/__init__.py`.
4. Verification, not assumption: dump the resulting subject ID list from
   both pipelines (GELSTM with `min_visits=2,max_visits=3` vs BrainTokenGT's
   existing windowed set) and diff them. If they don't match exactly, the
   "fair" claim is dead on arrival — fix the mismatch before running
   anything.

**B.1.2 — Confirm folds are actually paired, not just same-seed**

Both already read the same split CSVs
(`DATA/DELCODE/__metadata__/SPLITS/downstream/{train,val,test}.csv` —
confirmed present) via the same seeding helpers (C1's "hash the split file
into the run summary" requirement covers this). Still confirm per-fold
subject membership is identical for a given seed between a GELSTM run and a
BrainTokenGT run, not just trust "same seed → same folds." If they match,
this is a paired comparison (much stronger at this sample size); if not,
it's unpaired.

**B.1.3 — Register matched-cohort experiments**

- `CLASSIFIER/experiments/` (confirmed registry files:
  `ablation_seeds.yaml`, `ablation.yaml`, `comparison.yaml`,
  `data_journey.yaml`, `explain.yaml`, `longitudinal.yaml`, `sanity.yaml`,
  `static.yaml`): new entries
  `recon-ablation-gelstm-pretrained-frozen-2to3v-seed{42,43,44,45}` with
  `min_visits=2, max_visits=3`, same headline hyperparams — no new tuning,
  matching BrainTokenGT's fixed-config state (parity with C3: leave both
  fixed, or give both the same search budget).
- `BRAINTOKENGT/experiments/` (confirmed registry files:
  `ablation_seeds.yaml`, `ablation.yaml`, `longitudinal.yaml`):
  seed-suffixed entries for `braintokengt-delcode-whole-brain-repaired-fix-stabilized`
  at 43/44/45, same `ablation_seeds.yaml` pattern already used for GELSTM —
  mechanically new YAML entries, no code needed.

**B.1.4 — Run**

```bash
# BrainTokenGT reliability check + seeds (from BRAINTOKENGT/)
python run_experiment.py --id braintokengt-delcode-whole-brain-repaired-fix-stabilized --dry-run
python run_experiment.py --id braintokengt-delcode-whole-brain-repaired-fix-stabilized   # B.1.0 repeats

# once B.1.0-B.1.3 done, background both sets
python run_experiment.py --background   # BrainTokenGT seeds
python run_experiment.py --status
python run_experiment.py --collect

# from CLASSIFIER/
python run_experiment.py --background   # GELSTM 2-3v seeds
python run_experiment.py --collect
```

**B.1.5 — Compare and write up**

- Pull `test_auc`/`test_f1`/CV numbers from both outputs' `RESULTS.csv`, 4
  seeds each.
- If B.1.2 confirmed paired folds: paired test (Wilcoxon signed-rank or
  paired t-test across the 4 seeds × 5 folds) rather than just comparing
  pooled means.
- New doc, `DOCS/gelstm-vs-braintokengt-comparison.md`: state the fairness
  checklist explicitly (matched cohort ✓, matched seeds ✓, matched folds
  ✓/✗, no tuning either side ✓, BrainTokenGT convergence reliability —
  report B.1.0's finding honestly either way).

Context for why this comparison matters beyond "who wins": the pooled
4-seed reconstruction-value ablation (`DOCS/reconstruction-value-ablation.md`)
already shows GELSTM's own encoder isn't earning its place on DELCODE
(`none` mean test AUC 0.831 vs `pretrained_frozen` 0.781, `pretrained_finetuned`
degrading further and collapsing at seed 43, AUC 0.421). That's an
in-family finding, not evidence about BrainTokenGT — B.1 is the only
controlled way to test the trajectory-vs-reconstruction-fidelity hypothesis
this raises, since BrainTokenGT never had a reconstruction objective to
ablate and its own temporal module status is the open question B.1.0 gates.

---

## 5. Phase C/D — external validation, re-scoped to the real numbers

**Phase C — ADNI (n=162; 51 converters, 31.5%).** This is where the head-to-head
is decided. Full CV, seeds 42–45, all arms + BrainTokenGT + Tier-C baselines,
per criterion 1. Three claims in increasing strength: zero-shot transfer with
the frozen DELCODE threshold; within-ADNI CV; pooled leave-one-cohort-out.

At n=162 vs the 34-subject DELCODE test split, this is roughly 5× the
evaluation set — the reason the pivot is worth it. Note the CIs will still be
wide; report them.

**Phase D — OASIS-3, demoted, revised 2026-08-21.** The 46-subject triage
predicted here has run and fully recovered (§1.3a): OASIS-3 is now **60
subjects / 31 converters** with ≥2 sessions, not 40/19. That's a meaningfully
better power position than the demotion below assumed, though still smaller in
absolute n than ADNI (162/51) — this section's framing softens accordingly, but
"third validation cohort on equal footing with ADNI" is still not warranted
until the remaining 14 no-directory subjects (§1.3a) are resolved one way or
the other. Two honest uses, unchanged in kind, updated in scale:

1. **Unlabeled pretraining data** — all 128 subjects × 239 sessions feed the
   GAAE regardless of conversion labels (was 82 × 152). This is still OASIS-3's
   primary value, now larger.
2. **Qualitative-to-moderate robustness probe** — single-vendor, three scanner
   generations; report with CIs. At 31 converters this is closer to a genuine
   secondary validation arm than a purely qualitative probe, but still state
   explicitly that it's underpowered relative to the ADNI head-to-head decision
   in Phase C.

Both uses now have a concrete path to executable, not just counted: A.3/A.4
(§3) wire OASIS-3 through the same manifest → FC → splits pipeline as ADNI, on
equal footing — currently blocked by the reorientation-affine bug §3's A.3
describes, not by anything cohort-specific to OASIS-3.

**Remaining open item:** the 14 subjects with no directory at all (§1.3a) — a
distinct gap from the one just triaged. Check whether that data is still
in-flight from CORE or a genuine processing failure before finalizing the
OASIS-3 denominator for write-up.

**Phase E — pretraining ablation at scale.** Unchanged in intent. The honest
unlabeled pool is DELCODE + ADNI(237) + OASIS-3(128) as of 2026-08-21 — updated
from the 82 recorded on 08-20 now that the OASIS-3 triage recovered the full
128 fmriprep-flat subjects, still short of the ~380 v1 assumed. Still a large
multiple of DELCODE alone, so the test of whether `none ≈ frozen` is a property
of the method or of data volume remains valid. (`none ≈ frozen` is itself now a
live finding, not a hypothesis — see the pooled 4-seed reconstruction-value
ablation in `DOCS/reconstruction-value-ablation.md`: `none` mean test AUC 0.831
vs. `pretrained_frozen` 0.781, CV AUC 0.891 vs. 0.921 — well inside fold-to-fold
spread. Worth citing in Phase F's write-up regardless of how Phase C resolves.)

**Phase F — write-up.** Per-cohort tables (never a pooled mean hiding cohort
variance), CIs everywhere, degeneracy flags, the §4 fairness checklist mapped to
Chantal's three criteria, label-harmonization sensitivity analysis, honest
reporting of BrainTokenGT's convergence behaviour with curves.

---

## 6. Recommended headline claim

v1 asked whether the paper claims *"our architecture beats BrainTokenGT"* or
*"our approach generalizes across cohorts"*, and recommended the second on the
strength of three cohorts. **With the corrected numbers, "three cohorts" is not
honestly available** — one of the three has 19 converters.

Recommended framing:

> Two-cohort validation (DELCODE + ADNI), with the model comparison decided on
> ADNI (n=162) under a documented fairness protocol, and OASIS-3 reported as a
> secondary robustness probe with explicit power caveats.

This is weaker than v1's pitch and stronger than anything DELCODE-only. It also
does not depend on winning the head-to-head: under Chantal's criteria, a
documented fair comparison that our model *loses* is still a publishable,
defensible result — and the optimization-adequacy curves are what let you say so.

---

## 7. Reproducing the counts

```bash
cd /mnt/e/fyassine/ad-early-detection/DATA

# sessions per subject, both cohorts (content-based, not dir-based)
python3 -c "
import os,glob,collections
for coh in ['ADNI','OASIS3']:
    root=f'{coh}/__fmri_wholebrain_sch200_flat__/fmri'
    cnt={s:len(glob.glob(os.path.join(root,s,'*.nii.gz')))
         for s in os.listdir(root) if os.path.isdir(os.path.join(root,s))}
    print(coh,'dirs',len(cnt),'>=2 sessions',sum(v>=2 for v in cnt.values()),
          'sessions',sum(cnt.values()),
          dict(sorted(collections.Counter(cnt.values()).items())))
"
```

Result on 2026-08-20:

```
ADNI   dirs 237  >=2 sessions 162  sessions 567  {1:75, 2:71, 3:39, 4:34, 5:14, 6:2, 7:1, 8:1}
OASIS3 dirs 128  >=2 sessions  42  sessions 152  {0:46, 1:40, 2:26, 3:8, 4:5, 5:2, 6:1}
```

The `0:46` entry in the OASIS-3 histogram is §1.3's bug.

**Re-run 2026-08-21** (post-triage, §1.3a):

```
ADNI   dirs 237  >=2 sessions 162  sessions 567  {1:75, 2:71, 3:39, 4:34, 5:14, 6:2, 7:1, 8:1}
OASIS3 dirs 128  >=2 sessions  62  sessions 239  {1:66, 2:35, 3:13, 4:9, 5:2, 6:3}
```

ADNI is byte-for-byte unchanged from 08-20. OASIS-3's `0:46` bucket is gone —
those 46 subjects redistributed into the 1–6 buckets, which is exactly what
"the empty dirs got filled with real data" predicts, not new subjects
appearing.

Cohort intersection (ADNI IDs map `002_S_0729` → `ADNI002S0729`), via
`DATA/manifest/cohort_manifest.csv` (§3, A.0):

08-20:

```
ADNI   converters  cohort=456   flat>=1= 62  flat>=2= 51
ADNI   stable      cohort=1285  flat>=1=175  flat>=2=111
OASIS3 converters  cohort= 73   flat>=1= 41  flat>=2= 19
OASIS3 stable      cohort= 86   flat>=1= 39  flat>=2= 21
```

**08-21:**

```
ADNI   converters  cohort=456   flat>=1= 62  flat>=2= 51   (unchanged)
ADNI   stable      cohort=1285  flat>=1=175  flat>=2=111   (unchanged)
OASIS3 converters  cohort= 73   flat>=1= 64  flat>=2= 31   (was 41 / 19)
OASIS3 stable      cohort= 86   flat>=1= 62  flat>=2= 29   (was 39 / 21)
```

ADNI's 62 + 175 = 237 exactly accounts for every flat subject — the ADNI flat
product *is* the MCI cohort, no strays. Same holds for OASIS-3 post-triage:
64 + 62 = 126, close to the 128 flat dirs (2 subjects present on disk but not
matched into either cohort CSV — worth a spot-check, not urgent).

**Stability check, superseded.** The 21:30 (08-20) re-run returned identical
numbers to 21:15, and the doc originally read that as "these counts are
stable, will only move when the triage or a postprocessing pass is run." That
prediction held — but the triage *was* run since, and the counts *did* move
(§1.3a). ADNI's postprocessed-flat product genuinely is still static
(`process: NOT RUNNING`, settled short at 237/272 per
`monitor_flatten_progress.sh --once`, 2026-08-21 10:43); OASIS-3's moved
because someone re-ran its flatten/postprocessing job, not because the
CORE→Fritz pull delivered new raw data. Re-run the block above after any
further triage or postprocessing pass, and update §1.2 — this doc's own
generalized lesson (§1.3) applies to trusting *this document's* stability
claims too, not just the raw directory counts.

To compare both products directly:

```bash
python3 -c "
import os,glob
for coh in ['ADNI','OASIS3']:
    for prod in ['__fmriprep_wholebrain_flat__','__fmri_wholebrain_sch200_flat__']:
        root=f'{coh}/{prod}/fmri'
        ds=[d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
        empty=[d for d in ds if not glob.glob(os.path.join(root,d,'*.nii.gz'))]
        n=sum(len(glob.glob(os.path.join(root,d,'*.nii.gz'))) for d in ds)
        print(f'{coh:7} {prod:32} dirs={len(ds):4} empty={len(empty):3} sessions={n}')
"
```

Result, 2026-08-21 (superseding §1's earlier callout table, reproduced here for
traceability):

```
ADNI    __fmriprep_wholebrain_flat__     dirs= 268 empty=  0 sessions=675
ADNI    __fmri_wholebrain_sch200_flat__  dirs= 237 empty=  0 sessions=567
OASIS3  __fmriprep_wholebrain_flat__     dirs= 128 empty=  0 sessions=239
OASIS3  __fmri_wholebrain_sch200_flat__  dirs= 128 empty=  0 sessions=239
```

**Re-run 2026-08-22, ADNI only** (OASIS-3 unchanged from 08-21 — no
postprocessing pass touched it in between): the 08-21 evening
`postprocess_local.sh --flatten-only --overwrite` run (applying the
reorientation-affine fix, §3 A.3) landed *after* the snapshot above was taken.
ADNI's `__fmri_wholebrain_sch200_flat__` is no longer stuck at 237 — it now
matches `__fmriprep_wholebrain_flat__`'s subject count exactly (session count
differs by 1, 674 vs 675, plausibly one file that didn't pass QC/day-parsing;
not investigated further, immaterial at this margin):

```
ADNI    __fmri_wholebrain_sch200_flat__  dirs= 268 empty=  0 sessions=674
```

`build_adni_manifest.py`'s `EXPECTED_SUBJECTS`/`EXPECTED_SESSIONS` (used by
`assert_counts_match`) were updated from 237/567 to 268/674 to match. A.3's
Schaefer-200 extraction has since been run to completion against the
corrected manifest for both cohorts — see §3, A.3.

---

## 8. Sequencing

| | track | gate | status, 2026-08-21 |
|---|---|---|---|
| now | A.0 manifest + A.1 visit parsing | — | A.0 **done**; A.1 **helpers done, wiring not done** — see §3 |
| now, parallel | B (C1/C2/C3 fairness work) | — | in progress — BrainTokenGT fidelity/seed ablations landed (`3aa424d`); matched-cohort (2–3 visit) head-to-head not yet run |
| now, parallel | OASIS-3 46-subject triage (§5) | cheap, high value | **done** — full recovery, 82→128 usable subjects (§1.3a); 14 no-dir subjects remain a separate open item |
| then | A.2 DELCODE reproduction | ~~AUC 0.8321 exactly~~ → **0.7929, target retired** | **passed** — 0.8321 was leaky (pre-`0823ca6` leak fix); determinism re-run in progress (§3, A.2) |
| then | A.3 ADNI **+ OASIS-3** FC extraction | manifest assertions pass | **done, 2026-08-22** — both cohorts, `fc_path` 100% populated (ADNI 268/674, OASIS-3 128/234); see §3 A.3 |
| then | A.4 ADNI **+ OASIS-3** splits | fc_path eligibility gate | code done; real run not yet executed against the corrected 268/674 ADNI manifest — see §3 A.4 |
| decision | A.5 dual gate (zero-shot × within-ADNI) | see §3 table | not started |
| then | C — ADNI head-to-head | — | not started |
| then | D/E/F | — | not started |

Phase B is a few days and makes the DELCODE results defensible regardless of
what Phase A finds. Phase A is the higher-risk, higher-payoff track. **A.2 has
now run and passed** — the 0.8321→0.7929 drift traces to the `0823ca6`
label-leakage fix (2026-07-04), not to the A.0/A.1 visit refactor, so it is a
correction to adopt, not a regression to chase. Pending the same-seed
determinism re-runs, **A.3/A.4's code is done for both ADNI and OASIS-3** —
manifests carry `fc_path`, `DATA/manifest/demographics.py` +
`build_cohort_splits.py` exist and are tested, a dry run reproduces §1.2's
counts exactly — but real extraction is blocked on the reorientation-affine
bug (§3, A.3), not on A.2. Wiring A.1's cohort-aware helpers into
`GELSTM/dataset.py`/`GEC/dataset.py` (still open, see §3 A.1) remains a
separate, parallel item before those phases touch ADNI/OASIS-3 through the new
visit code.
