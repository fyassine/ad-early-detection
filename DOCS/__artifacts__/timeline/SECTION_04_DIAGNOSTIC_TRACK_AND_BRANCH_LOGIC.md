[← §3 — Execution track: Phases A–F](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md) | [Master Plan](MASTER_PLAN.md) | [§5 — Stability audit: BrainTokenGT vs GELSTM variance →](SECTION_05_STABILITY_AUDIT.md)

---

# §4 — Diagnostic track: Phases 0–3, and the branch logic

This section carries the reasoning behind the sequence, not just its content.

## The shared logic across Phase 0 and Phase 1: clean the instrument before measuring with it

Both phases exist to remove a possible confound **before the next step's output can be
trusted**, applied twice in a row:

| Phase | What it rules out | Why it has to go first |
|---|---|---|
| **0** | Two silent-fallback bugs + a seed-provenance bug in the code the diagnostic will run through | If a labeling bug were live, Phase 1's diagnostic would be measuring corrupted ground truth and its answer would be meaningless |
| **1** | Whether the at-chance conversion result reflects broken *features* vs. genuine *domain difficulty* | These two causes produce an identical symptom (an at-chance number) but demand opposite next actions — conflating them wastes days in the wrong direction |

## Phase 0 — the two defects (done, 22 Aug)

**Defect 1 — seed provenance.** `training_config.seed` recorded `42` in every run on disk
regardless of actual seed, because `build_config` never injected `exp["seed"]`. Confirmed
**provenance-only** — no training code reads that field, real seeding always ran correctly
off the papermill `SEED` parameter. Fixed by layering the registry seed as a final,
unshadowable step. The ~60 existing artifacts were not back-edited (see §1).

**Defect 2 — silent fallbacks in `GELSTM/dataset.py`.** The dead `cohorts_csv` load
(`except Exception: pass` around code whose result was never read) and the worse
label-resolution fallback (`else: label = 0`) — see §3, A.1. Both fixed, both
regression-tested.

**Verification:** `recon-ablation-gelstm-none` re-run twice after these fixes —
`run_summary.json` bit-identical to the 19 Aug baseline both times. Confirms both fixes (and
the seed fix) are true no-ops on DELCODE model *output*; only recorded metadata changed.

## Phase 1 — one diagnostic, before any other external work

**Why sex-decoding specifically, and why it had to run before touching the zero-shot arm.**
Zero-shot DELCODE→ADNI confounds two things: feature quality and domain shift. A null result
there is uninterpretable — you can't tell whether the model failed because the *features*
carry no signal in this cohort, or because the *task* (conversion) genuinely doesn't
generalize across cohorts even with good features. Sex is strongly, reliably decodable from
functional connectivity in the literature, and — critically — **depends on neither the
conversion label nor any domain-shift argument.** It's an orthogonal probe that shares the
exact pipeline but has ground truth nobody disputes. That's what makes it able to separate
the two explanations instead of restating the ambiguity.

**The decision rule was pre-registered — fixed before looking at the output** — specifically
to prevent picking a comfortable threshold after seeing the number:

| DELCODE sex-AUC | ADNI / OASIS-3 sex-AUC | Reading | Next |
|---|---|---|---|
| high | comparably high | Features carry real signal; at-chance conversion result is a finding | → **Branch A** |
| high | at chance | The A.3 extraction is broken for these cohorts; every external number is void | → **Branch B** |
| at chance | at chance | The probe or its harness is wrong | → re-run probe |

**Result, 23 Aug — Branch A fires.** 5-fold CV logistic regression on the vectorised upper
triangle of baseline FC (`DATA/manifest/probe_sex_decoding.py`):

| cohort | n | sex AUC (5-fold CV) |
|---|---|---|
| DELCODE | 167 | 0.6191 ± 0.0691 |
| ADNI | 192 | **0.7131 ± 0.0883** |
| OASIS-3 | 60 | 0.5368 ± 0.1129 |

ADNI decodes sex *better* than DELCODE — the reference line — so the A.3 extraction is not
broken for ADNI; the at-chance conversion result is a genuine finding, not a bug. OASIS-3
sits between chance and DELCODE with a wide CI at n=60: read as underpowered, not as a second
data point against ADNI's result.

**Honest caveat, stated once here because it matters for how hard this claim can be leaned
on:** DELCODE's own 0.619 is lower than sex-decoding AUCs typically reported in the
literature (usually 0.8+). The absolute floor this probe sets is soft. **The reading that
carries weight is the relative one** the decision rule was built on (no ADNI-specific fault)
— not an absolute signal-quality claim. Do not overstate this in the write-up.

## Phase 2 — branch on the answer, and why it's a fork rather than a checklist

The two branches allocate scarce pre-freeze time to mutually exclusive next steps: Branch A's
work is *wasted effort* if Branch B is true (you'd be building an interpretability story on
top of broken features); Branch B's cut is *actively wrong* if Branch A is true (you'd be
discarding a real, citable finding). That's why Phase 1 has to resolve before Phase 2 opens
either path — hedging by doing a bit of both would waste the very time this diagnostic exists
to save, four days from the freeze.

**Branch A — features are fine, the result is real.** *(This is the branch that fired.)*
In value order:

1. Metadata floor on ADNI/OASIS-3 — without it there's no reference line saying 0.50 is bad
   rather than typical for these cohorts. Cheap, CPU-only, same probe harness as Phase 1.
2. Zero-shot DELCODE→ADNI with the frozen DELCODE threshold — the missing half of the A.5
   dual gate, now interpretable because Phase 1 excluded the feature explanation.
3. Per-cohort tables with degeneracy counts — half the GELSTM external runs predict every
   subject positive; a mean that hides that is the exact failure this thesis audits in
   others' work.
4. **Do not** run the Δt ablation on ADNI — ablating an input inside a model that's at chance
   measures nothing. The Δt *contribution* claim stays DELCODE-only; the Δt
   *interval-distribution* table (D5) stands on its own regardless.

**Branch B — features are broken.** *(Did not fire — kept here because the decision to skip
it was itself deliberate and worth recording.)*

1. Cut the external block from the thesis entirely — Ch. 6 reports DELCODE only.
2. Do not attempt to fix the A.3 extraction before 26 Aug — unbounded-scope bug, four days
   out, the thesis doesn't depend on it.
3. D5 still ships — it comes from manifests and `allowed_days`, not the FC matrices.

Either branch keeps the Ch. 3 Δt argument and Ch. 5 baseline audit whole — that's the point
of running the probe instead of guessing.

## Phase 3 — DELCODE-side work, explicitly parallel to the branch above

**Why this sits outside the fork instead of downstream of it:** it touches neither ADNI nor
OASIS-3, so gating it behind Track 2's resolution would put genuinely unrelated work behind a
slower decision, for no reason. This is flagged in the source material as "the part that is
actually on the 28 Aug critical path" for exactly that reason.

1. **D2 byte-equality re-gate.** ✅ Done — see §3, A.2.
2. **`recon-ablation-gelstm-pretrained-finetuned-seed45` at HEAD.** ✅ Done — completes the
   single-commit-state table in §5, Finding 2.
3. **Phase B1/B2/B3 notebook corrections.** B1 done (§8). **B2 and B3 still open — the one
   remaining pre-Ch.6 blocker.**

## The eliminated hypotheses — why the original triage order was wrong

Before the sex probe, three hypotheses for the at-chance result were listed in triage order
(label harmonization → Δt normalization → domain difficulty). **That order was itself wrong**
and superseded the same evening the numbers came in — evidence collected already excluded the
first two, so working the list top-down would have wasted the days the freeze doesn't have:

| Hypothesis | Status | Evidence |
|---|---|---|
| Label harmonization fault | ❌ excluded | `converter_status` and `label` agree perfectly in both cohorts (ADNI 65/127, OASIS-3 31/29, zero off-diagonal); dataset takes `converter_status` first, `int64` 0/1 |
| Δt normalisation (`MAX_INTERVAL_MONTHS=108`) clipping | ❌ excluded as a within-cohort cause | ADNI's longest interval is 2031 d ≈ 66.8 mo, inside 108, so nothing clips; scale differs from DELCODE but train/test share the scale within a cohort |
| Visit-count / length shortcut | ❌ excluded in all 3 cohorts | AUC of `n_scans` alone: DELCODE 0.457, ADNI 0.477, OASIS-3 0.523 — nobody has a length shortcut |
| Data loading dropped/mangled subjects | ❌ excluded | Executed notebook logs exactly the split-CSV counts; `min=2 max=10 mean=3.1` scans/subject matches manifest |
| **Feature quality (FC matrices themselves)** | ❌ excluded, 23 Aug | Phase 1's sex probe (above) |
| **Genuine domain difficulty** | ✅ the finding | Every other row excluded → Branch A fires |

**Note on BrainTokenGT's within-ADNI CV of 0.705.** Read as evidence of *overfitting*, not
feature quality — the same runs score 0.427 on held-out test. CV 0.705 with test 0.427 is not
telling you the features are informative; it's telling you the model memorized the CV folds.

---

[← §3 — Execution track: Phases A–F](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md) | [Master Plan](MASTER_PLAN.md) | [§5 — Stability audit: BrainTokenGT vs GELSTM variance →](SECTION_05_STABILITY_AUDIT.md)
