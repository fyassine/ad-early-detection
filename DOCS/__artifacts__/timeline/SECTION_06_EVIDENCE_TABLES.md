[← §5 — Stability audit: BrainTokenGT vs GELSTM variance](SECTION_05_STABILITY_AUDIT.md) | [Master Plan](MASTER_PLAN.md) | [§7 — Positioning: SOTA, novelty, venue →](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md)

---

# §6 — Evidence tables

## External validation — 24 runs, all at chance on held-out test

| Cohort | Arm | CV AUC | **Test AUC** | range | degenerate |
|---|---|---|---|---|---|
| ADNI (test n=39, 13 conv.) | GELSTM `none` | 0.540 ± 0.030 | **0.496 ± 0.074** | 0.411–0.577 | 3/4 |
| ADNI | GELSTM `pretrained_frozen` | 0.579 ± 0.033 | **0.480 ± 0.129** | 0.322–0.636 | 2/4 |
| ADNI | BrainTokenGT stabilized | 0.705 ± 0.029 | **0.427 ± 0.053** | 0.373–0.497 | 0/4 |
| OASIS-3 (test n=13, 7 conv.) | GELSTM `none` | 0.565 ± 0.043 | **0.530 ± 0.273** | 0.238–0.786 | 1/4 |
| OASIS-3 | GELSTM `pretrained_frozen` | 0.632 ± 0.072 | **0.565 ± 0.119** | 0.476–0.738 | 0/4 |
| OASIS-3 | BrainTokenGT stabilized | 0.770 ± 0.035 | **0.536 ± 0.177** | 0.381–0.786 | 1/4 |

"Degenerate" = predicts every test subject positive (sens 1.00, spec 0.00). **Half the
GELSTM external runs are degenerate.** OASIS-3's test split has 42 label pairs, so its AUC
moves in steps of 0.024 — a single run there carries almost no information.

## Sex-decoding positive control (§4, Phase 1)

| cohort | n | sex AUC (5-fold CV) |
|---|---|---|
| DELCODE | 167 | 0.6191 ± 0.0691 |
| ADNI | 192 | **0.7131 ± 0.0883** |
| OASIS-3 | 60 | 0.5368 ± 0.1129 |

## Metadata floor (age + sex + visit timing, gradient boosting)

**CV AUC 0.6157 ± 0.0653, test AUC 0.4929 (chance).** The imaging pipeline clears this by a
wide margin on DELCODE (GELSTM CV 0.944, test 0.907) — a genuine strength, belongs in every
results table.

## Δt-conditioning is inert on DELCODE — because DELCODE has almost no time variation

Ablating the Δt input changes held-out test AUC by **exactly zero** (0.9071 → 0.9071,
identical sensitivity and specificity) — traced and confirmed correct, not a broken ablation.
By contrast, **shuffling visit order costs** 0.9071 → 0.8486 ± 0.0258 — the recurrence uses
sequence *order*, just not sequence *timing*, on this cohort.

| Interval | Count | Share |
|---|---|---|
| 12 months | 242 | **90.0%** |
| 24 months | 19 | 7.1% |
| 36 months | 7 | 2.6% |
| 48 months | 1 | 0.4% |

This is why ADNI/OASIS-3 (irregular, continuous elapsed-day intervals) are "the only place a
contribution of this thesis can be demonstrated at all" for Δt — see the interval CVs in §3,
Phase A.4 (D5): ADNI 0.647, OASIS-3 0.574 vs DELCODE's near-zero variation.

## DELCODE reconstruction-value ablation, pooled at commit `a9e4cf2`/`20b5957`

| Arm | Mean test AUC |
|---|---|
| `none` | **0.831** |
| `pretrained_frozen` | 0.781 |
| `random` | 0.731 |
| `pretrained_finetuned` | 0.711 |

`none ≈ pretrained_frozen`, CV AUC 0.891 vs 0.921 — well inside fold-to-fold spread. The
graph encoder is not earning its place on DELCODE. (The fine-tuned arm's single-seed mean
here predates the completed 4-seed sweep in §5, Finding 2 — 0.6924 ± 0.2000 supersedes 0.711
as the more complete figure.)

## GELSTM vs BrainTokenGT matched-cohort (Phase B1, done 23 Aug)

Globs every `run_summary.json` per experiment id, filtered to commit `5e33e2170` (4 GELSTM +
18 BrainTokenGT runs — filtered on commit, not on the recorded `dirty` flag, since many
legitimate same-commit repeats were launched back-to-back with no intervening commit):

- Test AUC: GELSTM **0.8782 ± 0.0256** (n=4) vs BrainTokenGT **0.6163 ± 0.0679** (seed means
  of 18 pooled runs, corrected down from a previously cherry-picked 0.6672 ± 0.1008). Margin
  **+0.2619**.
- The n=20 fold-level t-test is gone (descriptive only, no test computed). The n=4 t-tests
  are replaced with a Wilcoxon signed-rank test (floors at p=0.125, the minimum attainable at
  n=4, reported as underpowered — GELSTM beats BrainTokenGT's seed mean on all 4 seeds) plus
  a seed-cluster bootstrap 95% CI on the margin: **[+0.158, +0.375]**, excludes zero,
  propagates BrainTokenGT's within-seed noise instead of discarding it.
- **Two distinct within/between-seed SD figures exist for BrainTokenGT — keep them
  separate, quote each with its population:**
  - Filtered notebook pair (18 runs after the `5e33e2170` commit filter): within-seed SD
    **0.090** vs between-seed-mean SD **0.068**.
  - Unfiltered audit pair (all 19 runs, §5 Finding 1): within-seed SD **0.1011** vs
    between-seed-mean SD **0.0706**.
  
  Both are correct; they're over different populations. Collapsing them into one number is
  the specific error to avoid in the write-up.

---

[← §5 — Stability audit: BrainTokenGT vs GELSTM variance](SECTION_05_STABILITY_AUDIT.md) | [Master Plan](MASTER_PLAN.md) | [§7 — Positioning: SOTA, novelty, venue →](SECTION_07_POSITIONING_SOTA_NOVELTY_VENUE.md)
