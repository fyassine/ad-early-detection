# Model-comparison plan v2 — external validation + Chantal's fairness bar

**Supersedes** the previous cross-dataset plan. Two inputs forced the revision:
Chantal's definition of a fair comparison (Slack, Aug 20) and a recount of what
is actually on disk after the `flatten_fmriprep.sh` glob fix.

Verified against disk on **2026-08-20 ~21:15**, re-verified **~21:30** — counts
unchanged. All counts below are reproducible from the commands in §7.

> **Which product the counts refer to.** Two flat products exist per cohort and
> they are *not* interchangeable. FC extraction consumes the **postprocessed**
> one (`__fmri_wholebrain_sch200_flat__`), which is static — its job has exited.
> The job still running at the time of writing feeds
> `__fmriprep_wholebrain_flat__`, one step upstream:
>
> ```
> ADNI    __fmriprep_wholebrain_flat__     dirs=250 empty= 0 sessions=620
> ADNI    __fmri_wholebrain_sch200_flat__  dirs=237 empty= 0 sessions=567
> OASIS3  __fmriprep_wholebrain_flat__     dirs=128 empty= 0 sessions=239
> OASIS3  __fmri_wholebrain_sch200_flat__  dirs=128 empty=46 sessions=152
> ```
>
> Consequence: the ~35 late-arriving ADNI subjects do **not** become usable when
> the running job finishes. They land in fmriprep-flat and then need a
> *postprocessing* pass before FC extraction can see them. Budget that step
> explicitly (§3, A.3) rather than assuming the numbers grow on their own.

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

| | v1 claimed | **actual (verified)** | delta |
|---|---|---|---|
| ADNI, ≥2 sessions, in cohort CSVs | 176 (55 conv / 121 stable) | **162 (51 conv / 111 stable)** | −14 |
| OASIS-3, ≥2 sessions, in cohort CSVs | 72 (33 conv / 39 stable) | **40 (19 conv / 21 stable)** | **−32 (−44%)** |
| Combined external, ≥2 sessions | ~248 | **202** | −46 |

ADNI is close enough that the v1 argument survives. OASIS-3 is not — it lost
nearly half, and 19 converters cannot carry a validation claim.

### 1.3 Root cause of the OASIS-3 shortfall: another count-the-directory bug

Same family as the `.html` glob you just fixed, different step:

```
OASIS3 fmriprep-flat : 128 subject dirs,  0 empty, 239 sessions
OASIS3 postproc-flat : 128 subject dirs, 46 empty, 152 sessions
```

46 subjects have fMRIPrep output but **zero** postprocessed BOLD — and the
directory was created anyway. `monitor_flatten_progress` counts directory
existence, so it reports `128/142 (90%) settled` while 46 of those 128 are empty
shells. The real usable OASIS-3 denominator is **82 subjects, not 128**.

This also means the "14 genuine exclusions" figure understates the loss by 46.
The 87 missing sessions are worth one triage pass before OASIS-3 is written off
(§5, Phase D) — they exist on the fMRIPrep side, so this may be recoverable
rather than a genuine exclusion.

**Generalized lesson, now applied twice:** progress counters that glob paths
count artifacts of the filesystem, not data. Every count in this plan is
defined as *files matching the expected content pattern*, and Phase A.0 makes
that structural.

### 1.4 The session-naming bug is confirmed, not predicted

v1 flagged this as a risk. It is real and I can point at the lines.

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

### A.0 Build the manifest before touching any model code *(new in v2)*

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

### A.1 Cohort-aware visit parsing

Implement the §2 three-field split in `CLASSIFIER/common/visits.py`, dispatching
on cohort. Add a **regression test asserting non-zero visits per cohort** — the
specific failure mode that would otherwise pass silently.

### A.2 DELCODE reproduction gate *(hard gate)*

After the refactor, re-run the saved GEGRU DELCODE evaluation. It must reproduce
**AUC 0.8321** exactly (per `gegru-cross-dataset-drift-validation.md` §1). Any
drift means the visit refactor changed DELCODE behaviour — fix before
proceeding. This is the same reload-consistency check that already caught the
`adjacency_k=8` vs `16` bug, reused.

### A.3 Schaefer-200 extraction → FC for ADNI

Run `DATA/DELCODE/src/processing/process_using_schaeffer_atlas.py` over the
staged ADNI flat BOLD. Parameterize cohort/paths rather than copying the script.
Expected output: **567 FC matrices over 237 subjects** — the current
postprocessed-flat contents, exactly.

The extra 13 subjects / 53 sessions sitting in ADNI's fmriprep-flat (250/620) are
*not* included in that figure and will not arrive by themselves: they need a
postprocessing pass first. Two options, and the choice matters for scheduling:

- **Proceed at 237 now.** The cohort intersection (§1.2) is already 162 subjects
  with ≥2 sessions, which is enough to decide the head-to-head. Recommended —
  do not block Phase C on stragglers.
- **Wait for the postprocessing pass**, then re-extract. Only worth it if the
  triage in §5 is being run anyway, since it is the same job.

Either way, re-run §7's count block after any postprocessing pass and update
§1.2 — do not assume the delta landed.

### A.4 Splits

Generate ADNI split CSVs with the identical stratified-by-label,
grouped-by-subject protocol as DELCODE, from the manifest. Same seed policy.

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

---

## 5. Phase C/D — external validation, re-scoped to the real numbers

**Phase C — ADNI (n=162; 51 converters, 31.5%).** This is where the head-to-head
is decided. Full CV, seeds 42–45, all arms + BrainTokenGT + Tier-C baselines,
per criterion 1. Three claims in increasing strength: zero-shot transfer with
the frozen DELCODE threshold; within-ADNI CV; pooled leave-one-cohort-out.

At n=162 vs the 34-subject DELCODE test split, this is roughly 5× the
evaluation set — the reason the pivot is worth it. Note the CIs will still be
wide; report them.

**Phase D — OASIS-3, demoted.** At 40 subjects / 19 converters it cannot support
a validation claim and should not be presented as a third validation cohort.
Two honest uses remain:

1. **Unlabeled pretraining data** — all 82 subjects × 152 sessions feed the
   GAAE regardless of conversion labels. This is now OASIS-3's primary value.
2. **Qualitative robustness probe** — single-vendor, three scanner generations;
   report with CIs and state explicitly that it is not powered for the
   head-to-head.

**Before either: run the 46-subject triage** (§1.3). Those subjects have
fMRIPrep output and no postprocessed BOLD. If the postprocessing is re-runnable,
OASIS-3 roughly doubles and Phase D may be worth restoring to validation status.
This is the highest-value-per-hour item in the whole plan — one triage pass
against a possible +46 subjects.

**Phase E — pretraining ablation at scale.** Unchanged in intent, but the
honest unlabeled pool is now DELCODE + ADNI(237) + OASIS-3(82), not the ~380 v1
assumed. Still a large multiple of DELCODE alone, so the test of whether
`none ≈ frozen` is a property of the method or of data volume remains valid.

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

Cohort intersection (ADNI IDs map `002_S_0729` → `ADNI002S0729`):

```
ADNI   converters  cohort=456   flat>=1= 62  flat>=2= 51
ADNI   stable      cohort=1285  flat>=1=175  flat>=2=111
OASIS3 converters  cohort= 73   flat>=1= 41  flat>=2= 19
OASIS3 stable      cohort= 86   flat>=1= 39  flat>=2= 21
```

ADNI's 62 + 175 = 237 exactly accounts for every flat subject — the ADNI flat
product *is* the MCI cohort, no strays.

**Stability check.** Re-run at 21:30 returned identical numbers. That is expected
and not a coincidence: the postprocessed→flat jobs for both cohorts have
**exited** (`process: NOT RUNNING`, settled short at ADNI 237/272 and
OASIS-3 128/142), so this product cannot change until something is re-run. The
job still logging progress is the *fmriprep*→flat one (ADNI 250/272), which
feeds a different directory — see the callout in §1.

So these counts are stable, not provisional. They will only move when either the
§5 triage or a postprocessing pass over the late ADNI arrivals is run. Re-run the
block above after either, and update §1.2.

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

---

## 8. Sequencing

| | track | gate |
|---|---|---|
| now | A.0 manifest + A.1 visit parsing | — |
| now, parallel | B (C1/C2/C3 fairness work) | — |
| now, parallel | OASIS-3 46-subject triage (§5) | cheap, high value |
| then | A.2 DELCODE reproduction | **AUC 0.8321 exactly** |
| then | A.3/A.4 ADNI FC + splits | manifest assertions pass |
| decision | A.5 dual gate (zero-shot × within-ADNI) | see §3 table |
| then | C — ADNI head-to-head | — |
| then | D/E/F | — |

Phase B is a few days and makes the DELCODE results defensible regardless of
what Phase A finds. Phase A is the higher-risk, higher-payoff track. A.2 is the
gate that protects everything already published.
