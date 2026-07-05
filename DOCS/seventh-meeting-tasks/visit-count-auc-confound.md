# Why 1-visit vs 2-visit AUC differs so much

Context: the GELSTM early-detection table (`common/early_detection.py`) reports AUC
separately for subjects truncated to their first N visits. The jump from N=1 to N=2
looks like strong evidence that the second scan carries real signal. Most of that jump
is an evaluation confound, not a property of the model.

## The confound, in three parts

### 1. The evaluated cohort changes shape at every N

`early_detection_table` restricts to subjects with **≥ N** visits at each N, so N=1 and
N=2 are different populations, not the same subjects scored with less information.
Downstream (`mci`/`converter`) test split:

| N (visits) | n subjects | converters | frac converter |
|---|---|---|---|
| 1 | 34 | 14 | 0.412 |
| 2 | 28 | 14 | **0.500** |
| 3 | 18 | 10 | 0.556 |
| 4 | 12 | 7 | 0.583 |
| 5 | 5 | 5 | 1.000 |
| 6 | 2 | 2 | 1.000 |

Going from N=1 to N=2 drops exactly 6 subjects — **all 6 are stable MCI** (the
single-scan subjects). Class balance shifts from 41% to 50% converter purely from that
drop, before the model has done anything differently. Rows N≥5 are single-class and
uninterpretable (`early_detection_table`'s `min_subjects`/single-class guard exists for
exactly this reason).

### 2. Visit count is itself a label proxy (informative dropout)

Converters are followed for longer than stable-MCI subjects in this cohort — they don't
drop out as early. Test-split visit counts by diagnosis:

| diagnosis | n | mean n_scans | median | min | max |
|---|---|---|---|---|---|
| converter | 14 | 3.71 | 3.5 | 2 | 6 |
| mci (stable) | 20 | 2.35 | 2.0 | 1 | 4 |

(Same direction in train/val.) Because `n_scans` correlates with the label, any
deeper-N cohort is automatically converter-enriched. Part of the "N=2 lift" is scoring
a richer positive class, not the model reading a second scan's biology.

### 3. Small-sample instability

34 vs 28 subjects with 14 positives is a small enough n that a handful of rank flips
moves AUC substantially. Not a large effect on its own, but it stacks with 1 and 2.

### Secondary, model-side effect

At N=1 the LSTM/GRU sees a length-1 sequence with `delta_t = [0.0]` always
(`model/GELSTM/dataset.py`) — an out-of-distribution input shape relative to the
mostly multi-visit training sequences. This can depress N=1 AUC beyond the pure
information difference, independent of the cohort confound above.

## Existing tooling for disentangling this

`common/visit_confound.py` was already built for exactly this question, and
`SANITY_VISIT_COUNT_CONFOUND.ipynb` already exercises it end-to-end through the adapter
layer (GEC/GELSTM/GEGRU). Key routines:

- `cohort_composition_table` — per-N class balance, explains why the raw AUC-vs-N table
  isn't apples-to-apples (see table above).
- `early_detection_fixed_cohort` — recomputes AUC-vs-N on a cohort **held fixed**
  (only subjects observed at every N). If the N=1→2 gap survives here, it's real
  added-visit information; if it collapses, it was cohort composition.
- `prob_vs_visit_count` + `within_subject_prob_slopes` — separates "model reads visit
  count as a shortcut" (flat within-subject slope, strong between-subject correlation)
  from "evidence accumulation" (consistent within-subject drift, e.g. negative slopes
  for stable MCI as more clean visits accumulate).
- `require_full_window=True` on `LongitudinalSubjectDataset` — an alternative fix at
  the data-loading level: forces every kept subject to exactly `max_visits`, which the
  docstring calls out as neutralising "longer sequence = more likely converter" leakage.

## What to do with this

- Don't read the raw N=1→N=2 AUC delta from `early_detection_table` as "the second
  visit helps by X points" — it's confounded by class balance and cohort size.
- Use `early_detection_fixed_cohort` (fixed population across N) as the number to
  actually trust for "does an additional visit help."
- Cross-check with `prob_vs_visit_count` / `within_subject_prob_slopes` to make sure
  any real fixed-cohort gain is evidence accumulation and not the model keying on
  sequence length itself.

## GEGRU: same cohort confound, independent architecture

`sanity-visit-confound-gegru` (source run `gegru-trajectory-whole-brain/lucky-harbor-3`)
exercises the identical downstream test split as the GELSTM numbers above — its
`cohort_composition` block matches the table in part 1 exactly (34/28/18/12/5/2 subjects,
same converter counts at every N). That's expected (same `SPLITS/downstream/test.csv`,
same `require_full_window` machinery), but it's a useful cross-check: the composition
confound is a property of the cohort, not of any one model architecture.

GEGRU's own raw AUC-vs-N (`GELSTMAdapter` with `rnn_type: "gru"`) reproduces the same
shape as GELSTM's, with different absolute numbers (different recurrent cell):

| N (visits) | n subjects | raw AUC (variable cohort) | fixed-cohort AUC (n=12 throughout) |
|---|---|---|---|
| 1 | 34 | 0.607 | 0.629 |
| 2 | 28 | 0.832 | 0.800 |
| 3 | 18 | 0.925 | 0.886 |
| 4 | 12 | 0.943 | 0.943 |

Compare to GELSTM's own fixed-cohort numbers from the same tooling
(`sanity-visit-confound-gelstm/sunny-hill-2`): 0.486 → 0.829 → 0.886 → 0.971. Both models
show the bulk of the raw N=1→2 jump surviving in the fixed-cohort version too (GEGRU:
0.629→0.800; GELSTM: 0.486→0.829) — so for both architectures, part of the N=1→2 gap is
genuine added-visit signal, not purely the cohort-composition artifact from part 1. The
`within_subject_slopes` diagnostic for GEGRU shows the same evidence-accumulation pattern
used to sanity-check this: non-converters trend negative (`median_slope=-0.198`,
86% of subjects with a negative slope) while converters trend flat/slightly positive
(`median_slope=+0.064`) — consistent with "probability drifts down as more clean visits
accumulate for stable subjects," not the model just keying on sequence length.

## Follow-up added this cycle

Added a "visit-count confound" check to
`CLASSIFIER/notebooks/SANITY/SANITY_LONGITUDINAL_GELSTM.ipynb` (check 6, appended after
the existing 5 ablations) that runs the fixed-cohort comparison and the
shortcut-vs-evidence diagnostics against whichever GELSTM checkpoint the notebook has
loaded, without requiring the full adapter/experiment-runner pipeline that
`SANITY_VISIT_COUNT_CONFOUND.ipynb` uses.
