[← §7 — Positioning: SOTA, novelty, venue](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md) | [Master Plan](MASTER_PLAN.md) | [§9 — Cut list and out of scope →](SECTION_09_CUT_LIST_AND_OUT_OF_SCOPE.md)

---

# §8 — Remaining work, day by day

## Sun 23 – Mon 24 Aug — corrections (in progress)

- [x] **Phase B1** — `GELSTM_VS_BRAINTOKENGT_MATCHED_COHORT.ipynb` rewritten against real
  evidence. Done 23 Aug (§6).
- [ ] **Phase B2** — `GELSTM_ABLATIONS_AGGREGATION.ipynb`: drop `stats.ttest_ind` on fold
  AUCs, standardise `ddof=1`, rewrite takeaway 3 per §5 Finding 2 (no longer
  "provisional/withdrawn" — "naive fine-tuning at a shared LR").
- [ ] **Phase B3** — `DOCS/reconstruction-value-ablation.md`: recompute the pooled table at a
  single named commit (§6 has the numbers).
- [x] Fine-tuned repeat sweep collected, filtered by `git.short_commit` (§5).
- [x] D2 byte-equality re-gate — passed (§3, A.2).
- [x] Track-2 triage — resolved, Branch A fired (§4).

## Tue 25 – Wed 26 Aug — fill Chapter 6

- [ ] Fill each table declared in the Ch. 6 data contract, in order, per the six reporting
  rules in §6.1 of the contract (never pool cohorts, always show degeneracy, CIs everywhere).
- [ ] **Confront the GEC exposure explicitly** — `gec-trajectory-whole-brain` scores test AUC
  0.964 against GELSTM 0.921. Either the recurrence isn't contributing, or GEC is exploiting
  the length shortcut its `append_visit_mask` makes available. Within-subject slope
  diagnostics answer this; the answer goes in §6.2, not a footnote — single most likely
  challenge in a defence.
- [ ] Reproduce the pre-registered interpretation table verbatim and mark which row the
  observed pattern matches *before* stating any conclusion.

> **Evidence freeze: end of Wed 26 Aug.** Every number in the handed-off draft must be on
> disk by now.

## Thu 27 Aug — headline and abstract

- [ ] Choose the framing (§7's template is defensible today).
- [ ] Write Abstract + Kurzfassung + Ch. 9 concluding remarks against frozen evidence.
- [ ] Decide whether EXPOSURE annotations stay in the handed-off document (recommend: keep
  for the supervisor, hide before final submission).

## Fri 28 Aug — hand-off

- [ ] `python scripts/run_checks.py` — once, after all code changes are done.
- [ ] Full compile: ToC, Appendix E register, no stray `\draftnote` in a finished chapter.
- [ ] Resync `DOCS/draft_paper/paper.tex` to the thesis.
- [ ] Send PDF to supervisor with a cover note naming deferred items, the ADNI/OASIS-3 abort
  decision (kept, framed as a finding), and three things to read first: Ch. 5, §6.2's GEC
  confrontation, the Δt finding in Ch. 3.
- [ ] State a feedback deadline in the note: comments by Mon 1 Sept to be actionable.

## Sat 29 – Sun 30 Aug — off the thesis

Rest, or side-lane experiments only. No chapter edits.

## Mon 31 Aug – Tue 1 Sept — incorporate feedback

- [ ] Apply comments in order of severity: argument → evidence → wording.
- [ ] Verify every bibliography entry against the publisher record.
- [ ] Feedback freeze end of 1 Sept — later comments logged in
  `DOCS/draft_paper/POST_SUBMISSION_NOTES.md`, not applied.

## Wed 2 Sept — final hardening

- [ ] `run_checks.py` if any code changed since 28 Aug.
- [ ] Hide EXPOSURE annotations if that's the decision.
- [ ] Full clean compile.

## Thu 3 Sept — submit

Morning only. Don't use the buffer day as a work day.

## Side lane — A.2 pooled multi-cohort training (supervisor request, 23 Aug)

**Status: not started, explicitly below the 26 Aug freeze.** Post-hand-off / paper item.
Train DELCODE + ADNI → test held-out OASIS-3. What exists: time unification (§3 A.1, one
code path for all three cohorts already handles a pooled month scale for free), per-cohort
manifests with zero subject overlap. What's missing, in order:

1. A pooled split — build from the *downstream train/val* subjects of both cohorts only,
   never DELCODE's held-out test.
2. An explicit site/cohort covariate decision (pass cohort as a feature, or hold it out and
   report a cohort-decoding probe the way §4's sex probe was run) — do not leave it implicit.
3. Registry entries in `external_validation.yaml`, arms `none`/`pretrained_frozen`, seeds
   42–45, evaluated on held-out OASIS-3.
4. Same reporting rule as everywhere else: OASIS-3 test n=13 moves AUC in 0.024 steps — read
   the seed distribution, keep the degeneracy count.

**Precondition: do not run before the zero-shot arm (§4, Branch A item 2) exists** — it's the
interpretable baseline pooled training has to beat. This is *training-data* pooling on a
single named held-out cohort — compatible with the "never pool OASIS-3 metrics with ADNI"
rule (§3, Phase D), which is about pooling *metrics*, not training data. Don't read one as
licence for the other.

---

[← §7 — Positioning: SOTA, novelty, venue](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md) | [Master Plan](MASTER_PLAN.md) | [§9 — Cut list and out of scope →](SECTION_09_CUT_LIST_AND_OUT_OF_SCOPE.md)
