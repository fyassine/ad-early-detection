[← §4 — Diagnostic track: Phases 0–3, and the branch logic](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md) | [Master Plan](MASTER_PLAN.md) | [§6 — Evidence tables →](SECTION_06_EVIDENCE_TABLES.md)

---

# §5 — Stability audit: BrainTokenGT vs GELSTM variance

**Question asked:** does the notebook evidence actually support the words "stabilized"
(BrainTokenGT) and the fine-tuning-instability conclusion (GELSTM)? **Both answers were no**,
and both have since been corrected.

## Finding 1 — BrainTokenGT's "stabilized" means "doesn't crash," not "reproducible"

The stabilization fix (`give_weight_decay=1e-3`, `give_lr_scale=0.1`) targets a run-to-run
**crash** (NaN scores in `TopK.forward`), diagnosed as GPU floating-point-order sensitivity —
`cudnn.deterministic` is set, but `torch.use_deterministic_algorithms` is never called
anywhere in the repo, so the scatter/gather backward in `TopKPooling` and the node-embedding
gather in `grcu.py` are nondeterministic on GPU. A comparison notebook's claim "converged
cleanly, without NaNs" was true about crashes and silently stood in for a false claim about
reproducibility.

**19 repeat runs, same config, confirm it:**

| seed | n | mean | SD |
|---|---|---|---|
| 42 | 7 | 0.5779 | 0.1167 |
| 43 | 4 | 0.6802 | 0.1024 |
| 44 | 4 | 0.5227 | 0.0849 |
| 45 | 4 | 0.6477 | 0.1002 |

Mean **within-seed** SD (**0.1011**) exceeds the **between-seed-mean** SD (**0.0706**) — the
"four seeds" a naive notebook reports are four draws from run-to-run noise, not four seed
effects. Grand mean over all 19 runs: **0.6025 ± 0.1123** — below an earlier
cherry-picked-4 figure of 0.6672 ± 0.1008 (which hardcoded one run path per seed; seed 42
alone had 7 candidates on disk and the old cell took whichever ran last).

By contrast, **GELSTM reruns byte-identically** at a fixed seed (verified twice, §3 A.2) —
the nondeterminism is BrainTokenGT-specific. That makes the defensible finding *"our pipeline
reproduces, theirs doesn't"* — stronger than the invalid p-values it replaces (an n=4 paired
t-test treating seed as the only variance source; an n=20 fold-level t-test treating
non-independent folds as independent samples — both had to go, not be caveated).

**Fixing the nondeterminism is explicitly cut from scope** —
`torch.use_deterministic_algorithms(True)` would raise inside a third-party port and require
rewriting ops with unbounded scope, days from submission. The 19-run distribution is the
stronger, already-collected finding; report it, don't chase determinism.

## Finding 2 — the fine-tuned GELSTM arm *does* reproduce within a commit; seed 43's collapse is real

**Single-commit-state table, per-seed means (repeats nested, not pooled as independent
draws):**

| seed | runs at one commit-state | within-seed spread | mean |
|---|---|---|---|
| 42 | 0.6464, 0.6464, 0.6679 | 0.021 | 0.6536 (n=3) |
| 43 | 0.4250, 0.4250 | 0.000 | 0.4250 (n=2) |
| 44 | 0.8393, 0.8357 | 0.004 | 0.8375 (n=2) |
| 45 | 0.8536 | — | 0.8536 (n=1) |

**Across-seed: mean 0.6924, SD 0.2000 (ddof=1), range [0.4250, 0.8536].**

Seeds 42–44 ran at `4b9f6d8`; seed 45's `run_summary.json` records `b7e2560` — not the same
hash, but the D2 byte-equality gate was re-run immediately after that commit and reproduced
seeds 42–44's arm exactly, so the two hashes are a verified no-op pair on DELCODE and are
treated as one commit-state here. Flagged explicitly rather than silently smoothed over.

Two consequences, in opposite directions:

1. **Seed 43's collapse is a genuine seed effect, not run-to-run noise.** It replicates
   exactly (0.4250 twice) and reproduces across commits (0.4214 at an earlier hash).
   Between-seed SD at one commit (**0.200**) is an order of magnitude above within-seed SD
   (**≈0.01**) — the **mirror image** of BrainTokenGT's profile in Finding 1.
2. **The arm is commit-sensitive, and a pooled sweep on disk silently mixes commits.** Seed
   42 moves 0.7821 → 0.6464 between two commits with no seed change; seed 44 moves 0.7964 →
   0.8393. Any table for this arm **must** be computed at one commit — the table above is.

**The LR confound is unfixed and still qualifies the claim.** `adapters/gelstm.py:274`
builds a single Adam group over `model.get_trainable_params()` at one learning rate, so the
unfrozen pretrained GATv2 encoder trains at the same 1e-3 as the fresh head — exactly the
treatment BrainTokenGT's own stabilization applied *only* to its newly-unfrozen params
(`give_lr_scale=0.1`). No `encoder_lr_scale` field exists anywhere. **The honest sentence is
"naive fine-tuning at a shared learning rate destabilizes optimization at some seeds,"** not
"fine-tuning destabilizes optimization."

**Three candidate explanations for the collapse, ranked, not merged:**

1. **Naive fine-tuning at a shared LR** — the falsifiable one (add `encoder_lr_scale`, re-run
   seed 43). Stays unrun; deferred past hand-off. Until it runs, it is a named mechanism, not
   a demonstrated cause.
2. **Capacity vs sample size** — ≈966k trainable weights against N≈76 per fold once the
   encoder unfreezes. Explains why *some* seed collapses, not *which*.
3. **Pretext–downstream objective mismatch** — GAAE optimises FC reconstruction, the head
   optimises binary conversion; unfreezing moves the encoder off a representation never
   selected for the downstream task. Also why `none ≈ pretrained_frozen` in the ablation.

(1) is the mechanism the caveat rests on; (2) and (3) explain why it bites at this scale
rather than in a large-cohort setting.

---

[← §4 — Diagnostic track: Phases 0–3, and the branch logic](SECTION_04_DIAGNOSTIC_TRACK_AND_BRANCH_LOGIC.md) | [Master Plan](MASTER_PLAN.md) | [§6 — Evidence tables →](SECTION_06_EVIDENCE_TABLES.md)
