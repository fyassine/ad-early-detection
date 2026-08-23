# Submission runway: 22 Aug → 3 Sept 2026

**Two deadlines, not one:**

| Date | Milestone |
|---|---|
| **Fri 28 Aug** | **Feature-complete draft to the supervisor.** Compiles, Ch. 6 filled, abstract written. Nothing after this date may change the argument. |
| **Thu 3 Sept** | Submission. 29 Aug – 2 Sept is the supervisor's reading window, not writing time. |

**Deliverable:** the Master's thesis at `DOCS/all_sections/` (compile with `./compile.sh`).
`DOCS/draft_paper/paper.tex` is a supervisor-facing summary and follows the thesis, never
the reverse.

**Consequence of the 28 Aug hand-off:** Chapter 6 is no longer deferrable to the end. It must
be filled by 27 Aug, which means the evidence base freezes on **26 Aug**. Anything not
collected by then is not in the draft the supervisor reads, and therefore not in the thesis —
see the side lane below.

**Decision already taken:** Chapters 1–5 and 7–9 are written to be readable without Ch. 6, so
a thin Ch. 6 is survivable. A late Ch. 6 is not.

---

## Status as of 22 Aug, end of day (rev. after the external-validation runs landed)

| Item | State |
|---|---|
| Ch. 1 Introduction | ✅ written (146 lines) |
| Ch. 2 Foundations + 8 research gaps | ✅ written (414 lines) — carries the deep research |
| Ch. 3 Data + Δt finding | ✅ written (283 lines) |
| Ch. 4 Methodology | ✅ written (297 lines) |
| Ch. 5 Baseline audit | ✅ written (275 lines) — strongest chapter |
| Ch. 6 Results | 🟡 data contract only — **now due 27 Aug**, the critical path |
| Ch. 7 Software | ✅ written |
| Ch. 8 Discussion | ✅ written |
| Ch. 9 Conclusion | ✅ written |
| Appendices A–E | ✅ scaffolded; E auto-collects all 31 annotations |
| References | ✅ ~32 real entries (verify against publisher records) |
| Metadata floor | ✅ computed: CV AUC 0.6157 ± 0.0653, test AUC 0.4929 |
| Δt / shuffle ablation | ✅ run — see finding below |
| Fine-tuned repeat sweep | ✅ **complete** — all 4 seeds at a verified single commit-state; mean 0.6924 ± 0.2000 |
| ADNI wiring (D1) | ✅ done — cohort-aware `visits.py` wired into `GELSTM/dataset.py`, tests added; **uncommitted** |
| D2 byte-equality gate | ⚠️ **not re-run against the current working tree** — see below |
| ADNI + OASIS-3 validation runs | ✅ 24 runs collected 22 Aug — **and the result is at chance**, see finding 3 |
| Notebook stat corrections | ⬜ not started — still the critical path |

**Compile is green.** 31 annotations (5 gaps, 3 grey areas, 10 edges, 13 exposures) collect
into Appendix E automatically.

---

## Three findings from today that change the argument

### 1. Δt-conditioning is inert on DELCODE — because DELCODE has no time variation

Ablating the Δt input changes held-out test AUC by **exactly zero** (0.9071 → 0.9071,
identical sensitivity and specificity). The cause is not a broken ablation — the plumbing
was traced and is correct. It is the cohort:

| Interval | Count | Share |
|---|---|---|
| 12 months | 242 | **90.0%** |
| 24 months | 19 | 7.1% |
| 36 months | 7 | 2.6% |
| 48 months | 1 | 0.4% |

DELCODE is protocol-driven; nine intervals in ten are identical. A near-constant feature
cannot contribute. By contrast, **shuffling visit order does cost** 0.9071 → 0.8486 ± 0.0258
— so the recurrence uses sequence *order*, just not sequence *timing*.

**Consequence:** this promotes ADNI from "nice replication" to "the only place a
contribution of this thesis can be demonstrated at all." ADNI and OASIS-3 encode actual
elapsed days (`ses-d0381`) and are genuinely irregular.

### 2. The metadata floor is comfortably cleared

Age + sex + visit timing through gradient boosting: **CV AUC 0.6157 ± 0.0653, test AUC
0.4929** (chance). The imaging pipeline clears this by a wide margin (GELSTM CV 0.944, test
0.907). This is a genuine strength and belongs in every results table.

---

### 3. External validation landed — and both cohorts sit at chance

All 24 registered runs in `CLASSIFIER/experiments/external_validation.yaml` completed on
22 Aug. Per-arm test AUC, mean ± SD over seeds 42–45 (never pool the cohorts):

| Cohort | Arm | CV AUC | **Test AUC** | range | degenerate runs |
|---|---|---|---|---|---|
| ADNI (test n=39, 13 conv.) | GELSTM `none` | 0.540 ± 0.030 | **0.496 ± 0.074** | 0.411–0.577 | 3/4 |
| ADNI | GELSTM `pretrained_frozen` | 0.579 ± 0.033 | **0.480 ± 0.129** | 0.322–0.636 | 2/4 |
| ADNI | BrainTokenGT stabilized | 0.705 ± 0.029 | **0.427 ± 0.053** | 0.373–0.497 | 0/4 |
| OASIS-3 (test n=13, 7 conv.) | GELSTM `none` | 0.565 ± 0.043 | **0.530 ± 0.273** | 0.238–0.786 | 1/4 |
| OASIS-3 | GELSTM `pretrained_frozen` | 0.632 ± 0.072 | **0.565 ± 0.119** | 0.476–0.738 | 0/4 |
| OASIS-3 | BrainTokenGT stabilized | 0.770 ± 0.035 | **0.536 ± 0.177** | 0.381–0.786 | 1/4 |

"Degenerate" = predicts every test subject positive (sensitivity 1.00, specificity 0.00).
**Half of the GELSTM external runs are degenerate.** OASIS-3's test split has 42 label pairs,
so its AUC moves in steps of 0.024 and a single run there carries almost no information.

**This fires the pre-registered A.5 gate in `comparison-plan-v2.md` §3, and it fires it on
the "stop and debug" row for GELSTM:** within-ADNI CV is 0.54–0.58, i.e. at chance, which
that table defines as a pipeline or label-harmonization fault rather than a finding. The
complication is that BrainTokenGT reaches within-ADNI CV 0.705 on the same splits — so the
data are not inert, and a uniform pipeline fault is unlikely. The live hypotheses, in order:

> **Superseded within the day — see `NEXT_STEPS.md`.** The three hypotheses first listed here
> (Δt normalisation, label harmonization, domain shift) were checked, and the first two are
> **excluded**: `converter_status` and `label` agree perfectly in both cohorts, and ADNI's
> longest interval (2031 d ≈ 66.8 mo) sits inside `MAX_INTERVAL_MONTHS = 108`, so nothing
> clips. Two further candidates were also excluded — no visit-count shortcut exists in any
> cohort (`n_scans` alone scores AUC 0.457 / 0.477 / 0.523 on DELCODE / ADNI / OASIS-3), and
> the executed notebooks confirm every split subject loaded (`CV pool {stable: 101,
> converter: 52}`, test 26/13, 3.1 scans/subject).
>
> **Resolved 23 Aug — Branch A.** Label-free positive control (`DATA/manifest/probe_sex_decoding.py`,
> sex decoded from baseline FC, 5-fold CV): DELCODE 0.6191 ± 0.0691 (n=167), **ADNI 0.7131 ±
> 0.0883 (n=192)**, OASIS-3 0.5368 ± 0.1129 (n=60), zero subjects missing FC anywhere. ADNI
> decodes sex *better* than DELCODE, so the A.3 extraction is not broken for ADNI — the
> at-chance conversion result is a genuine finding. OASIS-3 is inconclusive at n=60, read as
> underpowered rather than a second fault signal. `NEXT_STEPS.md` Phase 1/2 carries the full
> table and next actions (metadata floor, zero-shot arm, degeneracy-count tables).

Two things the gate needs that were never run: the **zero-shot DELCODE→ADNI arm** with the
frozen DELCODE threshold (the other half of A.5's dual gate), and the **metadata floor on
ADNI/OASIS-3** — without it there is no reference line to say 0.50 is bad rather than
typical for these cohorts.

**And the good news underneath it:** the interval distributions are exactly what the Δt
argument needed.

| Cohort | median interval | IQR | CV | modal month bucket |
|---|---|---|---|---|
| DELCODE | 12 mo | — | — | **90.0%** at 12 mo |
| ADNI | 371 d | 207–419 d | **0.647** | 21.9% at 12 mo |
| OASIS-3 | 1012 d | 708–1342 d | **0.574** | 8.1% at 36 mo |

ADNI's modal interval covers barely a fifth of its visit pairs against DELCODE's nine
tenths. This is the two-cohort version of the Ch. 3 Δt table (deliverable D5) and it stands
**independently of whether the classifiers work** — it is a property of the cohorts, not of
the models. Ship it in Ch. 3 regardless of how the gate resolves.

---

## Day-by-day — writing lane (23–28 Aug)

This lane is the thesis. It runs on the CPU of your attention and does not wait for any GPU.

### Sun 23 – Mon 24 Aug — corrections
- [x] **Phase B1** — `DOCS/per_section/results/GELSTM_VS_BRAINTOKENGT_MATCHED_COHORT.ipynb`.
  **Done, 23 Aug.** Globs all run dirs per experiment id, filtered to commit `5e33e2170`
  (4 GELSTM + 18 BrainTokenGT runs; the old hardcoded path picked one of 6 seed-42 candidates
  and happened to land near the middle — 0.6104 vs the corrected seed-mean 0.6163). The n=20
  fold-level t-test is deleted (Figure 1A now descriptive only); the n=4 test-AUC/F1 t-tests
  are replaced with a Wilcoxon signed-rank test (floors at p=0.125, n=4, reported as
  underpowered) plus a seed-cluster bootstrap 95% CI on the margin ([+0.158, +0.375],
  excludes zero). Criterion C2 rewritten to separate "doesn't crash" from "reproducible."
  Bit-reproducibility (within-seed SD 0.090 vs between-seed-mean SD 0.068) is now a result in
  its own right. All figures and hand-written prose cells recomputed against the corrected
  data; notebook re-executed end-to-end with zero errors. Corrected GELSTM margin: **+0.2619**
  (was +0.2110 against the cherry-picked subset).
- [ ] **Phase B2** — `DOCS/meetings/ninth-meeting/GELSTM_ABLATIONS_AGGREGATION.ipynb`: drop
  `stats.ttest_ind` on fold AUCs; standardise on `ddof=1`; qualify takeaway 3 (the arm used
  one shared LR for pretrained encoder and fresh head — it measures *naive* fine-tuning)
- [ ] **Phase B3** — update `DOCS/reconstruction-value-ablation.md`; its Results section still
  says "initial single-seed" while the pooled 4-seed table exists
- [ ] Collect the finished fine-tuned repeat sweep; report as repeats nested in seeds.
      **Filter by `git.short_commit` when globbing** — seed 42 moves 0.78 → 0.65 across
      commits with no seed change, so a naive pooled mean averages two code states
- [ ] **D2, now overdue** — the byte-equality gate was never re-run after the A.1 cohort-aware
      wiring landed in the working tree. The newest `recon-ablation-gelstm-none` run predates
      it (`a9e4cf2`, 19 Aug). Re-run that id at HEAD and diff `run_summary.json`. It is the
      deterministic arm, so it must be byte-identical. **Every DELCODE number in Ch. 6 rests
      on this, and it is one run**
- [ ] **A.5 triage on ADNI** — **revised, see `NEXT_STEPS.md`.** (a) and (b) of the original
      list are already excluded; do not spend time on them. Run the **label-free FC positive
      control** (Phase 1) first, then branch: metadata floor + zero-shot arm if it passes,
      cut the external block if it fails

### Tue 25 – Wed 26 Aug — fill Chapter 6
- [ ] Fill each table declared in the Ch. 6 data contract, in order, obeying the six reporting
  rules in §6.1
- [ ] **Confront the GEC exposure explicitly** — `gec-trajectory-whole-brain` scores test AUC
  0.964 against GE-LSTM 0.921. Either the recurrence is not contributing, or GEC is using the
  length shortcut its `append_visit_mask` makes available. The within-subject slope
  diagnostics answer this; the answer goes in §6.2 and not in a footnote. This is the single
  most likely thing to be challenged in a defence.
- [ ] Reproduce the pre-registered interpretation table verbatim and mark which row the
  observed pattern matches — *before* stating any conclusion

> **Evidence freeze: end of Wed 26 Aug.** Every number that appears in the handed-off draft
> must be on disk by now. A run still training on 27 Aug is a Ch. 9 sentence, not a table row.

### Thu 27 Aug — headline and abstract
- [ ] With the evidence frozen, choose the framing. `DOCS/draft_paper/SOTA_POSITIONING.md` §6
  has a defensible template.
- [ ] Write Abstract + Kurzfassung + Ch. 9 concluding remarks against the evidence
- [ ] Decide whether the EXPOSURE annotations stay in the handed-off document. **Recommend:
  keep them in for the supervisor** — they are the fastest way for a reader to see what you
  already know is weak, and they can be hidden from `main.tex` before submission without
  touching any chapter.

### Fri 28 Aug — hand-off ✅
- [ ] `python scripts/run_checks.py` — **once**, after all code changes are done
- [ ] Full compile; check ToC, Appendix E register, no stray `\draftnote` in a finished chapter
- [ ] Resync `DOCS/draft_paper/paper.tex` to the thesis
- [ ] **Send the PDF to the supervisor** with a short cover note naming: the deferred items,
  the abort decision on ADNI, and the three things you most want read (suggest Ch. 5, §6.2's
  GEC confrontation, and the Δt finding in Ch. 3)
- [ ] State a feedback deadline in the note: **comments by Mon 1 Sept** to be actionable

Bibliography verification is *not* on the hand-off critical path — a supervisor reads
arguments, not volume numbers. It moves to 1–2 Sept.

---

## Day-by-day — supervisor window (29 Aug – 3 Sept)

The draft is out. This lane is deliberately light: do not reopen the argument while it is
being read, or you will be revising against a document the supervisor is not holding.

### Sat 29 – Sun 30 Aug — off the thesis
- [ ] Rest, or run side-lane experiments (below). No chapter edits.

### Mon 31 Aug – Tue 1 Sept — incorporate feedback as it arrives
- [ ] Apply supervisor comments in order of severity: argument → evidence → wording
- [ ] Verify every bibliography entry against the publisher record (author lists and
  volume/page numbers are drafted from the survey, not all checked)
- [ ] **Feedback freeze end of 1 Sept.** Comments arriving after this are logged in
  `DOCS/draft_paper/POST_SUBMISSION_NOTES.md`, not applied.

### Wed 2 Sept — final hardening
- [ ] `python scripts/run_checks.py` if any code changed since 28 Aug
- [ ] Hide EXPOSURE annotations from `main.tex` if that is the decision
- [ ] Full compile from clean; check ToC, Appendix E, page count, no `\draftnote`
- [ ] Re-resync `paper.tex`

### Thu 3 Sept — submit
- [ ] Submit in the morning. Do not use the buffer day as a work day.

---

## Side lane — experiments running in parallel

Experiments run **on the side, on their own schedule**, and are wired into the thesis only if
they land before the freeze. The writing lane never blocks on them.

| Item | Freeze rule |
|---|---|
| Fine-tuned repeat sweep (seeds 42–45) | ✅ **done, 22 Aug 22:58.** All 4 seeds landed at a verified single commit-state (`STABILITY_AUDIT.md` Finding 2). No longer a freeze risk |
| ADNI / OASIS-3 (Phase D, below) | ✅ runs landed; **the freeze rule now applies to the A.5 triage, not to the runs.** Abort checkpoint **26 Aug** unchanged |
| Anything else you start | Ch. 9 future work by default. Assume it does not make the thesis. |

### ADNI + OASIS-3 — Phase D ⚠️ **abort checkpoint Wed 26 Aug (now a triage gate, not a wiring gate)**

**The wiring is done and the runs exist.** What is not done is the pre-registered A.5 gate
that decides whether the at-chance result is a *finding* or a *bug*. That is now what the
26 Aug checkpoint is about.

- [x] **D1** — cohort-aware `visit_index`/`protocol_month`/`delta_t_months` wired into
  `CLASSIFIER/model/GELSTM/dataset.py`; `cohort` field threaded through
  `adapters/{gelstm,gec,gep}.py` and `BRAINTOKENGT/adapter.py`, defaulting to `"delcode"`,
  never inferred silently; `common/visits.py::month_allowed` takes a `cohort` argument;
  `tests/test_dataset_month_filter.py` extended (+72 lines). **Still uncommitted.**
- [ ] **D1b — one code-quality fix before this is committed.** The new
  `cohorts_csv` load in `GELSTM/dataset.py` swallows every exception
  (`except Exception: pass`). That is the silent fallback `.claude/rules/errors.md` exists to
  forbid, and it sits directly upstream of the at-chance result. Raise instead.
- [ ] **D2 — byte-equality gate (non-negotiable, and still open).** Re-run one existing
  DELCODE experiment at HEAD; `run_summary.json` must be byte-identical. Use
  `recon-ablation-gelstm-none` — it is the arm verified to rerun byte-identically. **This has
  not been done since D1 changed `month_allowed`'s semantics.** If it is not identical, stop.
- [x] **D3** — `CLASSIFIER/experiments/external_validation.yaml` registers 24 experiments:
  ADNI × {gelstm-none, gelstm-pretrained-frozen, braintokengt-stabilized} × seeds 42–45, and
  the same for OASIS-3. Note this is **three** arms per cohort, not the two D3 originally
  specified, and it includes OASIS-3, which the "explicitly out of scope" list below had cut.
- [x] **D4** — splits built with the stratified patient-level protocol, zero subject overlap:
  **ADNI 192 subjects (115/38/39; 65 converters, 33.9%)**, **OASIS-3 60 (35/12/13; 31
  converters)**. Both carry cohort-native `subject_id`/`allowed_days` columns.
  ⚠️ These supersede `comparison-plan-v2.md`'s **ADNI 162 / 51 converters** everywhere.
- [x] **D5** — inter-visit interval distributions computed for both cohorts; see the table in
  finding 3. **This is the deliverable that makes the Δt finding a three-cohort argument, and
  it is already safe to write into Ch. 3.**
- [ ] **D6 (new, and now the gate)** — the A.5 triage listed in the 23–24 Aug block.

> **Abort rule (revised again):** the runs are collected, so there is nothing left to abort
> *running*. What is now conditional is the **claim**. If the A.5 triage has not produced an
> answer by end of **26 Aug**, Ch. 6 reports the external results as *"an at-chance transfer
> result whose cause we did not isolate,"* with the degeneracy counts and the CV/test gap
> shown, and Ch. 9 takes the triage as future work. **Do not report a domain-shift finding
> that the pre-registered gate has not cleared** — that is the one move here that would be
> genuinely indefensible in a defence.

> **Also do not bury it.** An at-chance external result is a legitimate, publishable outcome
> for the framing this thesis already uses (see `SOTA_POSITIONING.md` §6) — the honest version
> is stronger than the alternative of omitting the cohorts and claiming DELCODE-only.

## Cut-list, in cut order

1. **The A.5 *explanation* for the external result** — cut if the 26 Aug triage fails; the
   external tables themselves stay in Ch. 6 either way, reported as unexplained. The runs are
   done, so cutting them now would be hiding evidence, not saving time
2. ~~Fine-tuned repeat sweep — seed 45 at HEAD only.~~ **Done, 22 Aug.** All 4 seeds landed;
   the withdrawal is **lifted** (see `STABILITY_AUDIT.md` Finding 2). The claim narrows to
   *naive fine-tuning at a shared LR*, it does not disappear — no longer a cut-list item
3. **Appendix C auto-generation** — hand-build a reduced hyperparameter table
4. **`paper.tex` resync** — the thesis is the deliverable
5. **Bibliography verification** — slips from the hand-off to 1–2 Sept; a supervisor reads
   arguments, not volume numbers

**Never cut:** Chapters 1–5 (written), Ch. 6, the abstract, the Phase B corrections, the
**D2 byte-equality gate**, and the **D5 interval table**. The invalid *p*-values must not
reach a document anyone reads — supervisor included. D2 is one run and every DELCODE number
depends on it. D5 costs nothing and is the only Δt evidence that survives the external
result.

**Never cut, and never slip:** the 28 Aug hand-off itself. If the writing lane runs late, the
draft goes out thinner, not later. A Ch. 6 with three of five tables filled and the gaps
marked `\draftnote` is a readable draft; a Ch. 6 that arrives on 1 Sept is not a draft at all.

---

## Explicitly out of scope

- **Strict-determinism repair of BrainTokenGT.** `torch.use_deterministic_algorithms(True)`
  will raise on ops that then need rewriting inside a third-party port — unbounded work, 12
  days from submission and 6 from hand-off. **The 19-run distribution is the stronger finding
  anyway**, and it is already
  collected.
- **The parcellation ablation** (whole brain vs DMN vs DMN+limbic) — no runs exist; Ch. 9.
- **The imaging-based survival head** — only KM + clinical Cox exist; Ch. 9.
- ~~**OASIS-3** — ADNI first.~~ **Superseded 22 Aug:** OASIS-3 ran alongside ADNI and its 12
  runs are on disk. Report it as the underpowered secondary probe it is (test n=13, AUC steps
  of 0.024) — never as a co-equal third cohort, and never pooled with ADNI.
- **Multi-cohort pooled supervised training** (train DELCODE + ADNI → test OASIS-3) — added to
  the record 23 Aug from the supervisor's cross-cohort strategy; never registered and never
  run. It needs a pooled split CSV, an explicit cohort-covariate decision, and the zero-shot
  arm first as its baseline — none of which fits before the 26 Aug freeze. Full spec in
  `NEXT_STEPS.md` §"Supervisor cross-cohort request" A.2. Ch. 9 / paper. Note this is
  *training-data* pooling on a single named held-out cohort, not the pooled-metrics reporting
  that §"never pooled with ADNI" forbids.
- **The LR-scaled fine-tuning arm** — not implemented (`adapters/gelstm.py:274` is still one
  Adam group). Not worth opening 6 days from hand-off; it becomes a stated limitation on the
  fine-tuning claim instead. Ch. 9 / paper.

---

## Standing rules while running experiments

- **The writing lane never waits on a GPU.** If a run is not done, write the sentence that
  says it is not done and move on. This is the single rule that protects the 28 Aug date.
- Launch side-lane runs at the *start* of a writing block, not the end — they train while you
  write instead of while you sleep on the deadline.
- Check free GPU **memory** (not utilisation) before launching: `scripts/dispatch.sh --list`.
  fritz is shared with a colleague; frieda is usually idle.
- Split sweeps across both boxes **by experiment id**. Never run one id twice — `outputs/<id>/latest`
  is rewritten without locking, and downstream notebooks then read the wrong run.
- `--dry-run` before any new registry entry.
- Pin `checkpoint_path` in registry entries rather than relying on the notebook's
  "latest checkpoint" fallback, which resolves by sort order and changes silently when a new
  checkpoint appears.
- **`run_summary.json`'s `training_config.seed` is fixed as of 23 Aug** (commit after
  `b7e2560`) — `build_config` now layers the registry seed last, so it can no longer be
  shadowed by a JSON config or hyperparams block. **For any run recorded before that fix**
  (everything through 22 Aug, including the seed-43/44/45 sweeps and both D2 gate runs on
  22 Aug), the field still reads `42` regardless of actual seed and must not be trusted —
  read the seed from the experiment id or the papermill parameters cell instead, since real
  seeding always ran correctly off the `SEED` parameter (`set_seed(SEED)`), verified in the
  executed notebooks. The ~60 pre-fix artifacts were not back-edited; see `NEXT_STEPS.md`
  Phase 0.
- **Never glob runs across commits.** `git.short_commit` in `run_summary.json` is the field
  that makes a pooled mean meaningful; the fine-tuned arm moves 0.13 AUC between commits at a
  fixed seed.
