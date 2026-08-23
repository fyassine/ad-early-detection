[Master Plan](MASTER_PLAN.md) | [§1 — Deadlines and standing rules →](SECTION_01_DEADLINES_AND_STANDING_RULES.md)

---

# §0 — How to read this document

This project ran **two parallel tracks that each number their own phases**, and conflating
them is the single most common source of confusion when reading the history. This section
exists to prevent that.

## Track 1 — Execution (Phases A–F)

**Question it answers:** *how do we get external-cohort (ADNI, OASIS-3) numbers at all?*

Build manifests → extract FC matrices → build splits → run the models → write up. A
straightforward, if long, engineering pipeline. It ran from roughly 20 Aug and **completed
22 Aug**, producing 24 experiment runs.

## Track 2 — Diagnostic (Phases 0–3)

**Question it answers:** *the 24 runs all came back at chance — is that a real finding, or a
pipeline bug?*

This track exists *only because* Track 1's output was ambiguous. It has nothing to do with
"what comes after Phase F" — it is a magnifying glass held over one specific bad result,
opened the same evening Track 1 finished (22 Aug) and resolved the next day (23 Aug).

## How they relate

```
Track 1 (Phases A–F)                 Track 2 (Phases 0–3)
  A  manifests + FC + splits
  B  DELCODE fairness protocol
  C  ADNI head-to-head        ──┐
  D  OASIS-3 probe            ──┼──▶  all 24 runs land at chance, 22 Aug
                                 │
                                 └──▶  Phase 0  fix 2 silent-fallback bugs
                                       Phase 1  ONE diagnostic (sex-decoding probe)
                                       Phase 2  branch on the answer (A or B)
                                       Phase 3  DELCODE-only work, runs in PARALLEL
                                                (not gated by the branch)
  E  pretraining-scale ablation  ◀── depends on nothing above
  F  write-up                    ◀── depends on Track 2's branch resolving
```

**Important override:** Track 1 already contained its own decision gate for exactly this
situation — the "A.5 dual gate" (§3, Phase A.5). Its table only had three rows (everything
works / real domain shift / pipeline bug), keyed off "is within-cohort CV above chance."
When the real numbers came in, GELSTM's within-ADNI CV *was* at chance (0.54) but
BrainTokenGT's was *not* (0.705) — on the identical splits. A.5's table has no row for that
combination, so **Track 2 was opened specifically to build a diagnostic sharp enough to
resolve the case A.5 didn't anticipate.** Track 2 supersedes A.5's own inference for this
run; §4 below is authoritative over §3's A.5 table.

## Resolution, in one sentence

Track 2 finished 23 Aug: the at-chance external result is a **genuine finding**, not a bug
(sex decodes *better* from ADNI's FC matrices than from DELCODE's — see §4). That unlocked
the rest of the evidence in §6 and the framing in §7.

---

[Master Plan](MASTER_PLAN.md) | [§1 — Deadlines and standing rules →](SECTION_01_DEADLINES_AND_STANDING_RULES.md)
