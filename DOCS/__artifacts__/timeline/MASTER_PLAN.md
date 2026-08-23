# Master plan — AD early detection thesis, Aug–Sept 2026

**Status as of:** 23 Aug 2026, evening. **Authoritative.** Where anything in
`DOCS/timeline/archive/` disagrees with this document, this document wins — it is the
reconciled, current state; the archive is the detailed working record it was built from.

**The two hard deadlines everything below is organised around:**

| Date | Milestone |
|---|---|
| **Wed 26 Aug** | **Evidence freeze.** Every number that appears in the handed-off draft must be on disk by end of this day. |
| **Fri 28 Aug** | **Feature-complete draft to the supervisor.** Compiles, Ch. 6 filled, abstract written. Nothing after this date may change the argument. |
| Thu 3 Sept | Submission. 29 Aug–2 Sept is the supervisor's reading window, not writing time. |

---

## Table of Contents

| Section | Title | Description | Link |
|---|---|---|---|
| **§0** | **How to read this document** | Parallel tracks (Track 1 Execution vs Track 2 Diagnostic), their relationship, and resolution | [SECTION_00_HOW_TO_READ_THIS_DOCUMENT.md](SECTION_00_HOW_TO_READ_THIS_DOCUMENT.md) |
| **§1** | **Deadlines and standing rules** | Calendar, write-first rule, GPU allocation etiquette, seed metadata provenance, commit filtering | [SECTION_01_DEADLINES_AND_STANDING_RULES.md](SECTION_01_DEADLINES_AND_STANDING_RULES.md) |
| **§2** | **Status dashboard** | Current state of thesis chapters, baseline audits, multi-cohort runs, and open blockers | [SECTION_02_STATUS_DASHBOARD.md](SECTION_02_STATUS_DASHBOARD.md) |
| **§3** | **Execution track: Phases A–F** | End-to-end engineering pipeline (manifests, FC extraction, gates, matched cohorts, head-to-head) | [SECTION_03_EXECUTION_TRACK_PHASES_A_F.md](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md) |
| **§4** | **Diagnostic track: Phases 0–3, and the branch logic** | Resolution of external at-chance results, sex-decoding positive control, and Branch A execution | [SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md) |
| **§5** | **Stability audit: BrainTokenGT vs GELSTM variance** | Analysis of BrainTokenGT non-determinism, GELSTM seed variance, and cross-commit sensitivity | [SECTION_05_STABILITY_AUDIT.md](SECTION_05_STABILITY_AUDIT.md) |
| **§6** | **Evidence tables** | Consolidated benchmark numbers: external validation, sex control, metadata floor, Δt, matched cohort | [SECTION_06_EVIDENCE_TABLES.md](SECTION_06_EVIDENCE_TABLES.md) |
| **§7** | **Positioning: SOTA, novelty, venue** | Research claims audit, competitive positioning against SOTA, venue shortlist, framing template | [SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md) |
| **§8** | **Remaining work, day by day** | Day-by-day execution checklist through submission, plus pooled multi-cohort side lane spec | [SECTION_08_REMAINING_WORK_DAY_BY_DAY.md](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md) |
| **§9** | **Cut list and out of scope** | Prioritized triage cut order if time runs short, non-negotiables, explicit out-of-scope boundaries | [SECTION_09_CUT_LIST_AND_OUT_OF_SCOPE.md](SECTION_09_CUT_LIST_AND_OUT_OF_SCOPE.md) |
| **§10** | **Provenance** | Mapping and reconciliation logic from archived planning docs in `DOCS/timeline/archive/` | [SECTION_10_PROVENANCE.md](SECTION_10_PROVENANCE.md) |

---

## Executive Summaries & Section Links

### [§0 — How to read this document](SECTION_00_HOW_TO_READ_THIS_DOCUMENT.md)
- **Track 1 (Execution, Phases A–F):** Built manifests, FC matrices, splits, and ran 24 experiment runs across DELCODE, ADNI, OASIS-3. Completed 22 Aug.
- **Track 2 (Diagnostic, Phases 0–3):** Resolved why all 24 external runs landed at chance. Determined this is a genuine finding, not a pipeline bug (sex decodes cleanly from ADNI FC). Completed 23 Aug with Branch A fired.
- [→ Open Section §0](SECTION_00_HOW_TO_READ_THIS_DOCUMENT.md)

### [§1 — Deadlines and standing rules](SECTION_01_DEADLINES_AND_STANDING_RULES.md)
- **Key Deadlines:** Evidence Freeze on **Wed 26 Aug**; Feature-complete draft to supervisor on **Fri 28 Aug**; Submission on **Thu 3 Sept**.
- **Golden Rule:** The writing lane never waits on a GPU.
- **Operational Rules:** Check free memory before launch; split sweeps by experiment ID; never trust pre-23 Aug `run_summary.json` seed field; never glob runs across commits.
- [→ Open Section §1](SECTION_01_DEADLINES_AND_STANDING_RULES.md)

### [§2 — Status dashboard](SECTION_02_STATUS_DASHBOARD.md)
- Chapters 1–5, 7–9 written; Appendices scaffolded; References prepared (~32 entries).
- Chapter 6 (Results) data contract established; due 27 Aug (critical path).
- Notebook statistics corrections (B2, B3) represent the only remaining pre-Ch. 6 blocker. Compile is green.
- [→ Open Section §2](SECTION_02_STATUS_DASHBOARD.md)

### [§3 — Execution track: Phases A–F](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md)
- **Phase A:** Manifest generation, visit parsing, DELCODE reproduction gate (passed), Schaefer-200 FC extraction, split creation, and transfer gate.
- **Phase B:** DELCODE fairness work (C1/C2/C3) and matched-cohort comparisons.
- **Phases C–F:** ADNI head-to-head, OASIS-3 probe (demoted by design), pretraining-scale ablation, and thesis write-up.
- [→ Open Section §3](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md)

### [§4 — Diagnostic track: Phases 0–3, and the branch logic](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md)
- **Phase 0:** Fixed two silent-fallback defects (visit metadata index alignment and demographic label fallback).
- **Phase 1:** Sex-decoding probe proved external FC matrices carry valid biological signal (ADNI sex AUC 0.887 vs DELCODE 0.771).
- **Phase 2:** Branch A selected (at-chance transfer is real domain shift); sets up zero-shot evaluation, metadata floor, and cross-cohort framing.
- **Phase 3:** DELCODE-side fairness and notebook corrections running in parallel.
- [→ Open Section §4](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md)

### [§5 — Stability audit: BrainTokenGT vs GELSTM variance](SECTION_05_STABILITY_AUDIT.md)
- **Finding 1:** BrainTokenGT "stabilized" means crash-free, not reproducible (within-seed run-to-run SD ±0.038 across seeds 42–45).
- **Finding 2:** Fine-tuned GELSTM reproduces within a commit (seed 43 collapse to 0.500 is real; shared Adam LR causes representational collapse).
- [→ Open Section §5](SECTION_05_STABILITY_AUDIT.md)

### [§6 — Evidence tables](SECTION_06_EVIDENCE_TABLES.md)
- Consolidated tables for: External validation (24 runs at chance), Sex-decoding positive control, Metadata baseline floor (CV AUC 0.616, Test 0.493), Δt time-conditioning ablation, DELCODE reconstruction ablation, and matched-cohort GELSTM vs BrainTokenGT comparison.
- [→ Open Section §6](SECTION_06_EVIDENCE_TABLES.md)

### [§7 — Positioning: SOTA, novelty, venue](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md)
- **Positioning:** Rigorous empirical audit and negative transfer result rather than benchmark claiming.
- **Novelty:** Multi-cohort evaluation protocol, temporal spacing diagnostics, and foundation model stability audit.
- **Venues:** Imaging Neuroscience, NeuroImage, or ML4H / CHIL.
- [→ Open Section §7](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md)

### [§8 — Remaining work, day by day](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md)
- **Schedule:** Sun 23–Mon 24 Aug (corrections); Tue 25–Wed 26 Aug (Ch. 6 tables & evidence freeze); Thu 27 Aug (abstract & framing); Fri 28 Aug (supervisor hand-off); Mon 31 Aug–Tue 1 Sept (feedback); Thu 3 Sept (submission).
- Specification for post-submission pooled multi-cohort training side lane.
- [→ Open Section §8](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md)

### [§9 — Cut list and out of scope](SECTION_09_CUT_LIST_AND_OUT_OF_SCOPE.md)
- Ordered cut list if time runs short (Appendix C auto-gen, paper.tex resync, bibliography verification).
- Non-negotiables: Chapters 1–6, abstract, Phase B corrections, D2 gate, 28 Aug hand-off deadline.
- Explicit out-of-scope boundaries (strict determinism porting, whole-brain parcellation ablations, imaging survival head).
- [→ Open Section §9](SECTION_09_CUT_LIST_AND_OUT_OF_SCOPE.md)

### [§10 — Provenance](SECTION_10_PROVENANCE.md)
- Mapping to historical planning documents in `DOCS/timeline/archive/` (`NEXT_STEPS.md`, `SUBMISSION_RUNWAY.md`, `comparison-plan-v2.md`, `STABILITY_AUDIT.md`, `SOTA_POSITIONING.md`).
- [→ Open Section §10](SECTION_10_PROVENANCE.md)
