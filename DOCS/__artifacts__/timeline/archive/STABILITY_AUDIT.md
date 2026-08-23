# BrainTokenGT / GELSTM stability audit — status, 22 Aug 2026 (rev. 22 Aug, evening)

Feeds `SUBMISSION_RUNWAY.md` Phase B1–B3 (notebook corrections, due 23–24 Aug) and
`SOTA_POSITIONING.md` §4 items 2 and 4 (variance decomposition as headline finding;
LR-scaled fine-tuning arm). This doc is the working record for that sub-thread; the
runway doc is the authority on dates and cut order.

**Question asked:** does the notebook evidence in
`DOCS/per_section/results/GELSTM_VS_BRAINTOKENGT_MATCHED_COHORT.ipynb` and
`DOCS/meetings/ninth-meeting/GELSTM_ABLATIONS_AGGREGATION.ipynb` actually support the words
"stabilized" (BrainTokenGT) and the fine-tuning-instability conclusion (GELSTM)? **Both
answers were no.** Full technical detail (per-run tables, code paths, root-cause diffs) is
in `~/.claude/plans/vast-marinating-whale.md`; this doc is the reconciled, current-state
summary for the thesis timeline.

---

## Status

| Phase | What | State |
|---|---|---|
| 0 | Determinism probe: is the GELSTM `pretrained_finetuned` arm reproducible? | ✅ done — **no** |
| 1 | Quantify BrainTokenGT's within-seed noise (repeat runs, seeds 42–45) | ✅ done — 19 runs |
| 2 | Make BrainTokenGT deterministic (`torch.use_deterministic_algorithms`) | ❌ **cut** — see below |
| B1 | Rewrite `GELSTM_VS_BRAINTOKENGT_MATCHED_COHORT.ipynb` against the real evidence | ✅ done, 23 Aug — see below |
| B2–B3 | Rewrite `GELSTM_ABLATIONS_AGGREGATION.ipynb` and `reconstruction-value-ablation.md` | ⬜ not started — critical path, due 23–24 Aug |
| 4a | Repeat sweep of the fine-tuned arm, seeds 42–44 at one commit | ✅ done — **it reproduces**, see Finding 2 |
| 4b | Repeat of seed 45 at HEAD | ✅ done, 2026-08-22 22:58 — **table complete, see Finding 2** |
| 4c | LR-scaled ("fair") fine-tuning arm | ❌ **not implemented** — `adapters/gelstm.py:274` still builds one Adam group |

---

## Finding 1 — BrainTokenGT's "stabilized" label means "doesn't crash," not "reproducible"

`git` history (`BRAINTOKENGT/experiments/longitudinal.yaml:82-88`) already says this
honestly: the "stabilized" fix (`give_weight_decay=1e-3`, `give_lr_scale=0.1`) targets a
run-to-run **crash** (NaN scores in `TopK.forward`), diagnosed as GPU floating-point-order
sensitivity — `cudnn.deterministic` is set, but `torch.use_deterministic_algorithms` is
never called anywhere in the repo, so the scatter/gather backward in `TopKPooling`
(`model/transformer.py`) and the node-embedding gather in `model/grcu.py:153` are
nondeterministic on GPU. The comparison notebook's criterion C2 ("converged cleanly …
without NaNs or numerical collapse") is a true claim about crashes, silently standing in for
a false claim about reproducibility.

**19 runs, same config, repeated per seed, confirm it:**

| seed | n | test AUC values | mean | SD |
|---|---|---|---|---|
| 42 | 7 | 0.6104, 0.7078, 0.6948, 0.5519, 0.5714, 0.5519, 0.3571 | 0.5779 | 0.1167 |
| 43 | 4 | 0.6364, 0.6883, 0.8182, 0.5779 | 0.6802 | 0.1024 |
| 44 | 4 | 0.5260, 0.5260, 0.4156, 0.6234 | 0.5227 | 0.0849 |
| 45 | 4 | 0.7208, 0.6169, 0.7338, 0.5195 | 0.6477 | 0.1002 |

Mean within-seed SD (**0.1011**) exceeds the between-seed-mean SD (**0.0706**) — the
"four seeds" the notebook reports are four draws from run-to-run noise, not four seed
effects. Grand mean over all 19 runs is **0.6025 ± 0.1123**, below the notebook's
cherry-picked-4 figure of 0.6672 ± 0.1008 (cell `f23d84fb` hardcodes one run path per seed;
seed 42 alone had 7 candidates on disk and the notebook took the one that happened to run
last). By contrast, GELSTM reruns **byte-identically** at a fixed seed
(`recon-ablation-gelstm-none`, verified twice) — the nondeterminism is BrainTokenGT-specific,
which makes the real, defensible finding *"our pipeline reproduces, theirs doesn't,"*
stronger than the invalid p-values currently in the notebook (n=4 paired t-test treating
seed as the only variance source; n=20 fold-level t-test treating non-independent folds as
independent samples — both must go, not be caveated).

**Phase 2 (fixing the nondeterminism) is explicitly cut** — see
`SUBMISSION_RUNWAY.md` §"Explicitly out of scope": `torch.use_deterministic_algorithms(True)`
will raise inside a third-party port and require rewriting ops with no bounded scope, 12
days from submission. The 19-run distribution above is the stronger and already-collected
finding; report it, don't chase determinism.

## Finding 2 — the fine-tuned arm *does* reproduce within a commit; the seed-43 collapse is real

**This reverses the provisional verdict recorded earlier today.** The repeat sweep has
landed for seeds 42–44 and the arm is reproducible to ~0.02 AUC when the commit is held
fixed:

| seed | runs at HEAD-equivalent* | within-seed spread | run at `a9e4cf2` |
|---|---|---|---|
| 42 | 0.6464, 0.6464, 0.6679 | 0.021 | 0.7821 |
| 43 | 0.4250, 0.4250 | 0.000 | 0.4214 |
| 44 | 0.8393, 0.8357 | 0.004 | 0.7964 |
| 45 | 0.8536 | — (1 run) | 0.8429 |

\* Seeds 42–44 ran at `4b9f6d8`. Seed 45 (launched 22:44, completed 22:58) landed mid-run
across the D1 commit and its `run_summary.json` records `b7e2560` — **not the same hash**,
but D2's byte-equality gate was re-run immediately after that commit and reproduced seeds
42–44's arm exactly, so `4b9f6d8` and `b7e2560` are a verified no-op pair on DELCODE. Treated
as one commit-state for this table; flagging the hash mismatch explicitly rather than
silently smoothing over it, per Finding 2 point 2's own rule.

**Single-commit-state table, per-seed means (repeats nested, not pooled as independent
draws):** seed 42 → 0.6536 (n=3), seed 43 → 0.4250 (n=2), seed 44 → 0.8375 (n=2), seed 45 →
0.8536 (n=1). **Across-seed: mean 0.6924, SD 0.2000 (ddof=1), range [0.4250, 0.8536].**

Two consequences, in opposite directions:

1. **The seed-43 collapse is a genuine seed effect, not run-to-run noise.** It replicates
   exactly (0.4250 twice) and reproduces across commits (0.4214). Between-seed SD across
   seeds 42–44 at HEAD (**0.198**) is an order of magnitude above within-seed SD (**≈0.01**)
   — the mirror image of BrainTokenGT's profile in Finding 1, and the contrast is now
   quantified on both sides rather than asserted. The earlier "not bit-reproducible"
   observation came from a run killed mid-fold-5 and does not survive the completed repeats.
   **The fine-tuning conclusion can be un-withdrawn** — with the caveat in point 4.
2. **The arm is commit-sensitive, which the seed sweep on disk silently mixes.** Seed 42
   moves 0.7821 → 0.6464 between `20b5957` and `4b9f6d8` with no seed change; seed 44 moves
   0.7964 → 0.8393. A pooled 4-seed mean built from the runs currently on disk averages
   across two code states. **Any table for this arm must be computed at one commit** — done,
   see the table above.
3. **The table is now complete — one commit-state, no gaps.** Between-seed SD (0.200) at
   this commit is even larger than the provisional 3-seed estimate (0.198), driven almost
   entirely by seed 43's collapse; within-seed SD stays nearly zero everywhere it was
   checked. The contrast with BrainTokenGT (Finding 1: within-seed SD 0.101 > between-seed
   SD 0.071) is sharper with the full table than the partial one.
4. **The LR confound is unfixed and therefore still qualifies the claim.**
   `CLASSIFIER/adapters/gelstm.py:274` still builds a single `torch.optim.Adam` group over
   `model.get_trainable_params()` at one `learning_rate`, so the unfrozen pretrained GATv2
   encoder trains at the same 1e-3 as the fresh head — exactly the treatment BrainTokenGT's
   own stabilization applied only to its newly-unfrozen params (`give_lr_scale=0.1`). No
   `encoder_lr_scale` field exists in any config or in any `run_summary.json` on disk. The
   honest sentence is therefore **"naive fine-tuning at a shared learning rate destabilizes
   optimization at some seeds,"** not "fine-tuning destabilizes optimization."

**The three candidate explanations for the collapse, and what each is worth (logged 23 Aug).**
Only the first is testable before hand-off, and it is cut; the other two are framing the
write-up already uses. Keep them ranked, not merged:

1. **Naive fine-tuning at a shared LR** — the mechanism above. The one *falsifiable* candidate
   (add `encoder_lr_scale`, re-run seed 43), and the one that stays unrun: `SUBMISSION_RUNWAY.md`
   §"Explicitly out of scope" and `SOTA_POSITIONING.md` §4.5 both put it after hand-off. Until
   it runs, it is a mechanism named in the caveat, not a demonstrated cause.
2. **Capacity vs sample size** — ≈966 k trainable weights against N ≈ 76 per training fold once
   the encoder unfreezes. Explains why *some* seed collapses; explains nothing about *which*.
   Carried in `SOTA_POSITIONING.md` §1–§2 as the small-N framing.
3. **Pretext–downstream objective mismatch** — the GAAE optimises FC reconstruction, the head
   optimises binary conversion, so unfreezing moves the encoder off a representation that was
   never selected for the downstream task. `SOTA_POSITIONING.md` §2.3 (the Brain-JEPA row) is
   this argument, and it is also why `none ≈ pretrained_frozen` in the ablation.

Do not write these as three co-equal causes. (1) is the mechanism the caveat rests on; (2) and
(3) are why it bites here rather than in a large-cohort setting.

---

## Next actions (owners match `SUBMISSION_RUNWAY.md` Phase B)

1. ~~**Phase B1**~~ **Done, 23 Aug.** `GELSTM_VS_BRAINTOKENGT_MATCHED_COHORT.ipynb` now globs
   every `run_summary.json` per experiment id (4 GELSTM + 18 BrainTokenGT runs, filtered to
   commit `5e33e2170` — not filtered on the recorded `dirty` flag, since many legitimate
   same-commit repeats were launched back-to-back without an intervening commit and
   Finding 1's own 19-run count doesn't filter on it either). Re-executed end-to-end, zero
   errors. Results, recomputed live (not hardcoded):
   - Test AUC: GELSTM **0.8782 ± 0.0256** (n=4, unchanged — no repeats exist for this arm)
     vs BrainTokenGT **0.6163 ± 0.0679** (seed means of 18 pooled runs, down from the
     previously-reported cherry-picked 0.6672 ± 0.1008). Margin widens to **+0.2619**.
   - The n=20 fold-level t-test is gone (Figure 1A is descriptive only, no test computed).
     The n=4 test-AUC/F1 t-tests are replaced by a Wilcoxon signed-rank test (floors at
     p=0.125, the minimum attainable at n=4, since GELSTM beats BrainTokenGT's seed mean on
     all 4 seeds — reported as underpowered, not as significance) plus a seed-cluster
     bootstrap 95% CI on the margin, **[+0.158, +0.375]**, which excludes zero and propagates
     BrainTokenGT's within-seed noise instead of discarding it.
   - Criterion C2 rewritten: "stabilized" is a claim about not crashing, not about
     reproducibility. Within-seed SD (0.090) vs between-seed-mean SD (0.068) is now its own
     reported result, not a caveat, with GELSTM's separately-verified byte-reproducibility
     (D2 gate) cited alongside it.
   - All 6 figures (CV/test distributions, ROC/PR, confusion matrices, calibration) rebuilt
     to pool every BrainTokenGT repeat run rather than one hardcoded/cherry-picked run.
2. **Phase B2** (`GELSTM_ABLATIONS_AGGREGATION.ipynb`) — drop `stats.ttest_ind` on fold
   AUCs; standardise `ddof=1`; **rewrite** takeaway 3 per Finding 2 (it is no longer
   "provisional/withdrawn" but "naive fine-tuning at a shared LR").
3. **Phase B3** — `DOCS/reconstruction-value-ablation.md`: recompute the pooled table at a
   single commit and say which one. Arm means at `a9e4cf2`/`20b5957`: `none` **0.831**,
   `pretrained_frozen` **0.781**, `random` **0.731**, `pretrained_finetuned` **0.711**.
4. ~~Run `recon-ablation-gelstm-pretrained-finetuned-seed45` once at HEAD (item 4b).~~
   **Done, 22:58.** Test AUC 0.8536. The single-commit-state table in Finding 2 is complete.

## Superseded

- *"Every claim about this arm is provisional until the repeat sweep lands"* (earlier
  revision, 22 Aug). The sweep landed; see Finding 2. The **withdrawal is lifted**, but the
  claim must be narrowed to naive/shared-LR fine-tuning rather than restated as written.
- *"4b (seed 45 at HEAD) is blocking"* (earlier revision, same day). No longer blocking —
  landed 22:58, table complete, mean 0.6924 ± 0.2000 (ddof=1) across seeds 42–45.

## Process note — duplicate session

This work was briefly run by two sessions sharing the same name and plan file
(`~/.claude/plans/vast-marinating-whale.md`), most likely two live continuations of one
session after a fork/resume point. Division agreed between them: this thread owns Phase
0–1 and B1–B3 (notebook rewrites); the other owns Phase 4 (fine-tuned repeat sweep +
LR-scale fix to `gelstm.py`). Recorded here so a future reader isn't confused by two sets of
"first-hand" run records for the same 19 BrainTokenGT runs.
