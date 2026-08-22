# Submission runway: 22 Aug → 3 Sept 2026

**Deliverable:** the Master's thesis at `DOCS/all_sections/` (compile with `./compile.sh`).
`DOCS/draft_paper/paper.tex` is a supervisor-facing summary and follows the thesis, never
the reverse.

**Decision already taken:** Chapter 6 (Results) is deferred. Its inputs are still moving and
the headline framing is decided *after* they land, not before. Chapters 1–5 and 7–9 are
written to be readable without it.

---

## Status as of 22 Aug, end of day

| Item | State |
|---|---|
| Ch. 1 Introduction | ✅ written (146 lines) |
| Ch. 2 Foundations + 8 research gaps | ✅ written (414 lines) — carries the deep research |
| Ch. 3 Data + Δt finding | ✅ written (283 lines) |
| Ch. 4 Methodology | ✅ written (297 lines) |
| Ch. 5 Baseline audit | ✅ written (275 lines) — strongest chapter |
| Ch. 6 Results | 🟡 data contract only, by decision |
| Ch. 7 Software | ✅ written |
| Ch. 8 Discussion | ✅ written |
| Ch. 9 Conclusion | ✅ written |
| Appendices A–E | ✅ scaffolded; E auto-collects all 31 annotations |
| References | ✅ ~32 real entries (verify against publisher records) |
| Metadata floor | ✅ computed: CV AUC 0.6157 ± 0.0653, test AUC 0.4929 |
| Δt / shuffle ablation | ✅ run — see finding below |
| Fine-tuned repeat sweep | 🔄 running (seeds 42/43/44 in flight, 45 queued) |
| ADNI wiring + validation | ⬜ not started — **abort checkpoint 28 Aug** |
| Notebook stat corrections | ⬜ not started |

**Compile is green.** 31 annotations (5 gaps, 3 grey areas, 10 edges, 13 exposures) collect
into Appendix E automatically.

---

## Two findings from today that change the argument

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

## Day-by-day

### 23–25 Aug — corrections and the chapters that are done
- [ ] **Phase B1** — `DOCS/per_section/results/GELSTM_VS_BRAINTOKENGT_MATCHED_COHORT.ipynb`:
  - glob all run dirs per experiment id (cell `f23d84fb` currently hardcodes one path per
    seed — this is how 0.6104 became "seed 42" while six other seed-42 runs sat on disk)
  - **delete** the n=20 fold-level paired *t*-test (folds share training data)
  - **replace** the n=4 test-AUC *t*-test with distributions + a non-parametric interval
  - rewrite criterion **C2**: the arm does not crash, but is not reproducible at fixed seed
  - add the bit-reproducibility contrast as a result in its own right
  - update the hand-written prose in the summary/takeaway cells — they do not recompute
- [ ] **Phase B2** — `DOCS/meetings/ninth-meeting/GELSTM_ABLATIONS_AGGREGATION.ipynb`: drop
  `stats.ttest_ind` on fold AUCs; standardise on `ddof=1`; qualify takeaway 3 (the arm used
  one shared LR for pretrained encoder and fresh head — it measures *naive* fine-tuning)
- [ ] **Phase B3** — update `DOCS/reconstruction-value-ablation.md`; its Results section still
  says "initial single-seed" while the pooled 4-seed table exists
- [ ] Collect the finished fine-tuned repeat sweep; report as repeats nested in seeds

### 26–28 Aug — ADNI (the publishability lever) ⚠️ **abort checkpoint 28 Aug**
- [ ] **D1** — `CLASSIFIER/model/GELSTM/dataset.py:41`: replace the DELCODE-only `parse_month`
  import with the cohort-aware trio already implemented in `CLASSIFIER/common/visits.py`
  (`visit_index`, `protocol_month`, `delta_t_months`); replace the hardcoded
  `cohorts['visit'].str.replace('M','')`; add a typed `cohort` field defaulting to
  `"delcode"`, never inferred silently
- [ ] **D2 — byte-equality gate (non-negotiable).** Re-run one existing DELCODE experiment;
  `run_summary.json` must be byte-identical. For DELCODE `delta_t_months` equals the protocol
  month exactly, so the refactor must be a no-op. **If it is not, stop.**
- [ ] **D3** — register `ext-adni-gelstm-{none,pretrained-frozen}-seed{42..45}`; two arms only
- [ ] **D4** — build ADNI splits with the same patient-level stratified protocol; verify zero
  subject overlap before running
- [ ] **D5** — report the ADNI inter-visit interval distribution beside DELCODE's Table 3.2.
  This is what makes the Δt finding a two-cohort argument instead of a one-cohort curiosity.

> **Abort rule:** if D2 has not passed by end of 28 Aug, cut Phase D entirely, document the
> wiring as completed-but-unvalidated, and move ADNI to Ch. 9 future work. **The thesis does
> not depend on it. The paper does.**

### 29–31 Aug — fill Chapter 6
- [ ] Fill each table declared in the Ch. 6 data contract, in order, obeying the six reporting
  rules in §6.1
- [ ] **Confront the GEC exposure explicitly** — `gec-trajectory-whole-brain` scores test AUC
  0.964 against GE-LSTM 0.921. Either the recurrence is not contributing, or GEC is using the
  length shortcut its `append_visit_mask` makes available. The within-subject slope
  diagnostics answer this; the answer goes in §6.2 and not in a footnote. This is the single
  most likely thing to be challenged in a defence.
- [ ] Reproduce the pre-registered interpretation table verbatim and mark which row the
  observed pattern matches — *before* stating any conclusion

### 1 Sept — decide the headline, write the abstract
- [ ] With all evidence in, choose the framing. `DOCS/draft_paper/SOTA_POSITIONING.md` §6 has
  a defensible template.
- [ ] Write Abstract + Kurzfassung + Ch. 9 concluding remarks last, against the evidence
- [ ] Decide whether the EXPOSURE annotations stay in the submitted document (they can be
  hidden from `main.tex` without touching any chapter)

### 2 Sept — hardening
- [ ] `python scripts/run_checks.py` — **once**, after all code changes are done
- [ ] Verify every bibliography entry against the publisher record (author lists and
  volume/page numbers are drafted from the survey, not all checked)
- [ ] Full compile; check ToC, Appendix E register, no stray `\draftnote` in a finished chapter
- [ ] Resync `DOCS/draft_paper/paper.tex` to the thesis

### 3 Sept — buffer and submit

---

## Cut-list, in cut order

1. **Phase D (ADNI)** — cut if the 28 Aug gate fails
2. **Fine-tuned repeat sweep** — if incomplete, keep the fine-tuning conclusion *withdrawn*
   (this is already the correct state; it costs nothing to leave it withdrawn)
3. **Appendix C auto-generation** — hand-build a reduced hyperparameter table
4. **`paper.tex` resync** — the thesis is the deliverable

**Never cut:** Chapters 1–5 (written), Phase B corrections. The invalid *p*-values must not
reach a submitted document.

---

## Explicitly out of scope

- **Strict-determinism repair of BrainTokenGT.** `torch.use_deterministic_algorithms(True)`
  will raise on ops that then need rewriting inside a third-party port — unbounded work, 12
  days out. **The 19-run distribution is the stronger finding anyway**, and it is already
  collected.
- **The parcellation ablation** (whole brain vs DMN vs DMN+limbic) — no runs exist; Ch. 9.
- **The imaging-based survival head** — only KM + clinical Cox exist; Ch. 9.
- **OASIS-3** — ADNI first.

---

## Standing rules while running experiments

- Check free GPU **memory** (not utilisation) before launching: `scripts/dispatch.sh --list`.
  fritz is shared with a colleague; frieda is usually idle.
- Split sweeps across both boxes **by experiment id**. Never run one id twice — `outputs/<id>/latest`
  is rewritten without locking, and downstream notebooks then read the wrong run.
- `--dry-run` before any new registry entry.
- Pin `checkpoint_path` in registry entries rather than relying on the notebook's
  "latest checkpoint" fallback, which resolves by sort order and changes silently when a new
  checkpoint appears.
