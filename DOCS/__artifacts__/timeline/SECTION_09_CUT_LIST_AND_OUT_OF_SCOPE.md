[← §8 — Remaining work, day by day](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md) | [Master Plan](MASTER_PLAN.md) | [§10 — Provenance →](SECTION_10_PROVENANCE.md)

---

# §9 — Cut list and out of scope

**In cut order, if time runs short:**

1. The A.5 *explanation* for the external result, if the triage had failed — moot, it
   succeeded. (Kept for provenance: the tables themselves would have stayed regardless,
   reported as unexplained.)
2. ~~Fine-tuned repeat sweep~~ — done, no longer a cut candidate.
3. Appendix C auto-generation — hand-build a reduced hyperparameter table if needed.
4. `paper.tex` resync — the thesis is the deliverable, not the paper summary.
5. Bibliography verification — slips to 1–2 Sept; a supervisor reads arguments, not volume
   numbers.

**Never cut:** Chapters 1–5, Ch. 6, the abstract, the Phase B corrections, the D2
byte-equality gate, the D5 interval table. **Never cut, never slip: the 28 Aug hand-off
itself.** If the writing lane runs late, the draft goes out thinner, not later.

**Explicitly not attempted before hand-off:**

- Strict-determinism repair of BrainTokenGT — unbounded scope inside a third-party port.
- The parcellation ablation (whole-brain vs DMN vs DMN+limbic) — no runs exist, Ch. 9.
- The imaging-based survival head — only KM + clinical Cox exist, Ch. 9.
- Multi-cohort pooled supervised training — spec exists (§8 side lane), not run, Ch. 9/paper.
- The LR-scaled ("fair") fine-tuning arm — `adapters/gelstm.py:274` still one Adam group.
  Narrowed to a stated limitation in words instead (§5).
- The Δt ablation on external cohorts — measures nothing on an at-chance model (§4, Branch A
  item 4).
- Rewriting the ~60 pre-fix `run_summary.json` seed fields — back-editing provenance data
  destroys the record it exists to provide.

---

[← §8 — Remaining work, day by day](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md) | [Master Plan](MASTER_PLAN.md) | [§10 — Provenance →](SECTION_10_PROVENANCE.md)
