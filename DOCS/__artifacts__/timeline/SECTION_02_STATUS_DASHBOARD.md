[← §1 — Deadlines and standing rules](SECTION_01_DEADLINES_AND_STANDING_RULES.md) | [Master Plan](MASTER_PLAN.md) | [§3 — Execution track: Phases A–F →](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md)

---

# §2 — Status dashboard

**Superseded 2026-08-30.** The table below through 2026-08-23 described a different,
Overleaf-hosted draft (Chapters 1–5 and 7–9 marked written, 32 bibliography entries). That
draft is not present in this repository: `THESIS/` here started this session as a
scaffold — every chapter body was an `\outlinenote{}` placeholder and `bibliography.bib`
held one template entry (Lamport, 1994). The state below reflects this repository, not the
Overleaf draft, and is not carried over from it per the Chapter 3 writing plan's decision to
draft fresh against the revised `THESIS/OUTLINE.md` rather than reuse the prior 283-line
Chapter 3 text.

| Item | State |
|---|---|
| Ch. 1 Introduction | ⬜ scaffold — `\outlinenote{}` placeholders only |
| Ch. 2 Foundations and Related Work | ⬜ scaffold |
| Ch. 3 Data, Cohorts and Preprocessing | ✅ written 2026-08-30 — all seven sections (§3.1–§3.7) drafted and compiling; every `\outlinenote{}` removed |
| Ch. 4 Methodology | ⬜ scaffold |
| Ch. 5 Baselines | ⬜ scaffold |
| Ch. 6 Experimental Results and Ablations | ⬜ scaffold — gated on every number being verified against a results file (`.claude/rules/thesis.md` §2) |
| Ch. 7 Discussion and Limitations | ⬜ scaffold |
| Ch. 8 Conclusion and Outlook | ⬜ scaffold |
| Appendix A Mathematical Derivations | ⬜ scaffold |
| Appendix B Implementation Details | ⬜ scaffold |
| Appendix C Reproducibility Table | ⬜ scaffold |
| Appendix D Split Integrity and Overlap Verification Logs | ✅ drafted 2026-08-30 — audit-module and test-suite specification (no live console output; split/manifest files are not present in this repository) |
| Appendix E Complete Cross-Validation Metrics | ⬜ scaffold |
| Appendix F Software Dependencies and Hardware Environment | ⬜ scaffold |
| `bibliography.bib` | 🟡 14 entries (1 template + 13 verified for Chapter 3 citations); two citations named on the January statistics slide deck (Miao et al. 2022, Rechberger et al. 2022) could not be confirmed by search and are not yet in the file |
| `THESIS/figures/` | 🟡 directory created 2026-08-30, `\graphicspath` set; no figure files added yet |
| Metadata floor | ✅ CV AUC 0.6157 ± 0.0653, test AUC 0.4929 (experimental result, unrelated to the chapter-drafting work above) |
| Δt / shuffle ablation | ✅ run — Δt inert on DELCODE, order matters (experimental result) |
| Fine-tuned repeat sweep | ✅ complete, single-commit-state table (experimental result) |
| ADNI/OASIS-3 cohort-aware visit wiring (D1) | ✅ done, committed `b7e2560` (experimental result) |
| D2 byte-equality gate | ✅ passed, re-run 3× (experimental result) |
| ADNI + OASIS-3 validation runs (24 total) | ✅ collected 22 Aug 2026 (experimental result) |

**Chapter waves, as specified for the Chapter 3 drafting work:**

| Wave | Chapter | Start condition |
|---|---|---|
| 1 | Ch. 3 Data, Cohorts and Preprocessing | Done 2026-08-30 |
| 2 | Ch. 4 Methodology | First half ready; second half gated on the `DOCS/flipped/METHODS.md` phrase-migration pass (31 occurrences of "pre-registered"/"pre-registration" and several banned tone phrasings still present in that source document) |
| 3 | Ch. 5 Baselines | Ready; wording constrained by `THESIS/OUTLINE.md` claims-register rows 4–6 |
| 4 | Ch. 6 Experimental Results and Ablations | Gated: every number verified against a results file |
| 5 | Ch. 7 Discussion and Limitations | After Ch. 6 settles |
| 6 | Ch. 1 Introduction | Late; §1.2/§1.3 need the bibliography built out further |
| 7 | Ch. 8 Conclusion and Outlook | Last; summarises verified results only |
| Parallel | Ch. 2 Foundations | Blocked on `bibliography.bib` growing beyond the Chapter 3 citations |
| Last | Appendices A, B, C, E, F | Absorb detail once the main text stabilises |

---

[← §1 — Deadlines and standing rules](SECTION_01_DEADLINES_AND_STANDING_RULES.md) | [Master Plan](MASTER_PLAN.md) | [§3 — Execution track: Phases A–F →](SECTION_03_EXECUTION_TRACK_PHASES_A_F.md)
