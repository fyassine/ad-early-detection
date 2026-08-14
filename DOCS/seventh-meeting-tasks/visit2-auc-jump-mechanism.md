# Why the AUC jumps so much from 1 to 2 visits

This is the detailed, mechanistic companion to
[`visit-count-auc-confound.md`](visit-count-auc-confound.md). That note framed the
N=1→N=2 jump as "mostly an evaluation confound." Re-running the numbers across all three
downstream models shows the emphasis was wrong: **the jump is predominantly real added
information, and it survives every control we have.** The confounds (cohort composition,
informative dropout, small-n) are real but secondary — they modulate the exact numbers,
they do not manufacture the jump.

All numbers below come from the 2026-06-21 visit-confound sanity runs, treated here as
ground truth:

- GELSTM — `sanity-visit-confound-gelstm/sunny-hill-2` (source `gelstm-trajectory-whole-brain/ancient-abyss-10`)
- GEGRU — `sanity-visit-confound-gegru/winter-flame-2` (source `gegru-trajectory-whole-brain/lucky-harbor-3`)
- GEC — `sanity-visit-confound-gec/graceful-wind-3` (non-recurrent baseline)

Machinery: `common/early_detection.py::early_detection_table` and
`common/visit_confound.py` (`early_detection_fixed_cohort`, `within_subject_prob_slopes`,
`prob_vs_visit_count`).

## TL;DR

1. **AUC is a ranking metric.** At N=1 all three models sit at chance (AUC ≈ 0.49–0.63):
   a single cross-sectional functional-connectivity snapshot does **not** rank-separate
   converters from stable MCI.
2. **At N=2 all three jump to 0.80–0.93.** The ingredient the second visit adds is the
   *change between visits* — the trajectory — and that is where the class signal lives.
3. **The jump survives the fixed cohort** (same 12 subjects scored at every N), so it is
   not the class-balance shift.
4. **The jump reproduces in a non-recurrent model (GEC),** so it is not an RNN
   out-of-distribution artifact from feeding a length-1 sequence.
5. The within-subject probability slopes show the model is reading *decline vs.
   stability*, not sequence length — this is evidence accumulation, not a count shortcut.

## The core evidence: AUC at each N, variable vs fixed cohort

`fixed` = only the 12 subjects with ≥4 visits, scored at every N (cohort held constant,
so any trend is pure information, not composition). `variable` = the standard
shrinking-cohort table.

| Model | cohort | N=1 | N=2 | N=3 | N=4 | **N=1→2 Δ** |
|---|---|---|---|---|---|---|
| **GELSTM** | variable | 0.554 | 0.801 | 0.863 | 0.971 | **+0.247** |
| | fixed (n=12) | 0.486 | 0.829 | 0.886 | 0.971 | **+0.343** |
| **GEGRU** | variable | 0.607 | 0.832 | 0.925 | 0.943 | **+0.225** |
| | fixed (n=12) | 0.629 | 0.800 | 0.886 | 0.943 | **+0.171** |
| **GEC** (non-recurrent) | variable | 0.564 | 0.929 | 0.950 | 0.914 | **+0.365** |
| | fixed (n=12) | 0.543 | 0.857 | 0.886 | 0.914 | **+0.314** |

Two facts fall straight out of this table:

- **N=1 is chance for every model.** 0.486–0.629 across the board. This is not a quirk of
  one architecture or one threshold — it is a property of the data. One baseline scan
  does not tell a mildly-abnormal-but-stable brain from a mildly-abnormal-and-declining
  one.
- **The jump is not the cohort confound.** Holding the cohort fixed at 12 subjects, the
  N=1→2 gap is still +0.34 (GELSTM), +0.17 (GEGRU), +0.31 (GEC). If the jump were an
  artifact of dropping stable-MCI subjects between N=1 and N=2, it would collapse in the
  fixed column. It does not.

## Mechanism 1 — one FC snapshot barely ranks conversion

The premise of the whole task: early detection is hard *cross-sectionally*. Converters
and stable MCI overlap heavily in a single functional-connectivity graph, because at
baseline both are, by construction, MCI. The fixed-cohort N=1 AUCs (0.486 / 0.629 /
0.543) quantify exactly this — on the identical 12 subjects, one snapshot ranks them no
better than a coin. There is nothing for the model to latch onto yet.

## Mechanism 2 — the second visit supplies a trajectory, and the trajectory carries the signal

Going from one graph to two lets the model encode the *direction and rate of change* of
the FC embedding between visits. The recurrent models (GELSTM/GEGRU) do this through the
hidden-state update with the real inter-visit `delta_t`; GEC does it by pooling the two
visits. All three convert a chance ranking into a strong one at N=2, which means the
discriminative content is in the delta, not in either endpoint alone.

The within-subject slope diagnostic (`within_subject_prob_slopes`) confirms *what* the
model reads from the trajectory — it reads stability:

| Model | non-converters (median slope, % negative) | converters (median slope, % negative) |
|---|---|---|
| GELSTM | −0.058, 86% negative | +0.015, 21% negative |
| GEGRU | −0.198, 86% negative | +0.064, 14% negative |

As a stable-MCI subject accumulates more clean visits, the model's P(converter) drifts
*down* (86% of non-converters have a negative slope in both models); converters stay
flat-to-rising. Between-subject, GELSTM's Spearman of final P(converter) vs. visit count
is r = −0.76 (p = 1e-4) **within** non-converters. A model keying on sequence length as a
shortcut would show flat within-subject slopes; the consistent negative drift is evidence
accumulation instead. So the second visit's job is mostly to let the model *rule out*
conversion by seeing a flat trajectory — which is precisely the signal a single snapshot
cannot provide.

## Mechanism 3 (secondary, threshold-level) — the N=1 default bias

AUC is threshold-free, so the story above stands on its own. But the *threshold-dependent*
metrics at N=1 are worth understanding because they explain why the jump looks even more
dramatic in a confusion matrix than in the AUC:

| Model | N=1 sensitivity | N=1 specificity | N=1 default behaviour |
|---|---|---|---|
| GELSTM | 0.857 | 0.050 | calls almost everyone **converter** |
| GEGRU | 0.643 | 0.400 | leans **converter** |
| GEC | 0.143 | 0.850 | calls almost everyone **stable** |

Fed a degenerate single-visit input (for the RNNs, a length-1 sequence with
`delta_t=[0.0]` — an input shape they rarely saw in training), each model collapses to its
own prior: the RNNs default toward "converter," GEC toward "stable." This is a real
effect and it does inflate the *apparent* jump in accuracy/specificity, but note it points
in **opposite directions** for the RNNs vs. GEC while the AUC jump is the same for both —
which is exactly why the AUC (rank-based) is the honest number and the sens/spec swing is
secondary color.

## What the confounds actually do (and don't)

The earlier note is right that these exist; they just aren't the cause of the jump.

- **Cohort composition / informative dropout.** Converters were followed longer in this
  split (test: converter mean 3.71 visits vs. stable-MCI 2.35, Mann-Whitney p = 0.011), so
  the shrinking variable-cohort table gets converter-enriched as N grows. This shifts the
  variable-cohort numbers but is *controlled out* by the fixed cohort — and the jump
  survives there. Use the fixed-cohort column as the number to trust.
- **Small-sample noise.** 12–34 subjects with ~14 positives; a few rank flips move AUC.
  This adds scatter to the exact values (and makes N≥5 single-class and uninterpretable),
  but cannot explain a +0.2–0.3 jump reproduced across three independent architectures.

## Bottom line

The 1→2 visit AUC jump is genuine and mechanistically clear: **a single functional-
connectivity snapshot ranks conversion at chance; a second visit turns that into a good
ranking because the discriminative signal is the between-visit trajectory (decline vs.
stability), not either snapshot alone.** It survives holding the cohort fixed and
reproduces in a non-recurrent model, so it is information, not the evaluation confound. The
right number to quote for "how much does the second visit help" is the **fixed-cohort**
delta (GELSTM 0.486→0.829, GEGRU 0.629→0.800, GEC 0.543→0.857), and the mechanism to cite
is the within-subject slope evidence that the model reads trajectory, not sequence length.
