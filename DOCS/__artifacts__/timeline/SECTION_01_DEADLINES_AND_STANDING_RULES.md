[← §0 — How to read this document](SECTION_00_HOW_TO_READ_THIS_DOCUMENT.md) | [Master Plan](MASTER_PLAN.md) | [§2 — Status dashboard →](SECTION_02_STATUS_DASHBOARD.md)

---

# §1 — Deadlines and standing rules

## Calendar

| Date | What must be true |
|---|---|
| 23–24 Aug | Notebook statistics corrections (Phase B2/B3, §8) — the only Track-2-adjacent item still open |
| 25–26 Aug | Chapter 6 tables filled, in the order the data contract declares |
| **26 Aug, end of day** | **Evidence freeze.** A run still training on 27 Aug is a Ch. 9 sentence, not a table row |
| 27 Aug | Headline framing chosen, Abstract + Kurzfassung + Ch. 9 written |
| **28 Aug** | `run_checks.py`, full compile, `paper.tex` resync, PDF sent to supervisor |
| 29–30 Aug | Off the thesis, or side-lane experiments only — no chapter edits |
| 31 Aug–1 Sept | Incorporate supervisor feedback, argument → evidence → wording, in that order |
| 2 Sept | Final hardening, re-compile |
| 3 Sept | Submit in the morning |

## The one rule that protects the 28 Aug date

**The writing lane never waits on a GPU.** If a run isn't done, write the sentence that says
it isn't done, and move on. Launch side-lane runs at the *start* of a writing block, not the
end, so they train while you write instead of while you sleep on the deadline.

## Standing operational rules

- Check free GPU **memory**, not utilisation, before launching (`scripts/dispatch.sh --list`
  or `gpus`). `fritz` is shared with a colleague; `frieda` is usually idle.
- Split sweeps across both boxes **by experiment id**, never run one id twice —
  `outputs/<id>/latest` is rewritten without locking; a repeated id corrupts it for every
  downstream reader on both hosts.
- `--dry-run` before any new registry entry. Pin `checkpoint_path` explicitly rather than
  relying on "latest checkpoint" sort-order resolution.
- **`run_summary.json`'s `training_config.seed` is trustworthy only for runs after 23 Aug**
  (commit after `b7e2560`). For anything recorded through 22 Aug — including the
  seed-43/44/45 sweeps and both D2 gate runs — that field reads `42` regardless of the actual
  seed used. Real seeding was always correct (it runs off the papermill `SEED` parameter,
  `set_seed(SEED)`, verified in every executed notebook); only the *recorded metadata* was
  wrong. Read the seed from the experiment id or the papermill parameters cell for pre-fix
  runs. The ~60 affected artifacts were **not** back-edited — back-editing provenance data
  destroys the record it exists to provide. Note the affected date range in the thesis
  Appendix instead.
- **Never glob runs across commits.** The fine-tuned arm moves 0.13 AUC between commits at a
  fixed seed (§5) — a pooled mean across commits is not a seed distribution, it's two code
  states averaged together. Filter every aggregation by `git.short_commit`.

---

[← §0 — How to read this document](SECTION_00_HOW_TO_READ_THIS_DOCUMENT.md) | [Master Plan](MASTER_PLAN.md) | [§2 — Status dashboard →](SECTION_02_STATUS_DASHBOARD.md)
