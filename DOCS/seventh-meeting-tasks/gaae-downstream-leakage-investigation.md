# GAAE pretrain/downstream leakage: what actually moved the AUC

Context: a supervisor flagged that patients used to train the GAAE encoder also appeared
in a downstream classifier's test/val split, and questioned why fixing that would move
AUC at all, since GAAE never sees labels. After the fix, downstream AUC dropped by
roughly 0.1-0.15. This doc traces what the "before" and "after" numbers actually are,
what changed between them, and what's attributable to the leak versus other confounds
bundled into the same period.

## The "before" and "after" images, identified

Two comparison images were assumed to come from `PROGNOSER/CROSS_NETWORK_COMPARISON.ipynb`.
They don't — that notebook produces survival C-index/Brier/AUC-at-time-t heatmaps. The
AUROC leaderboard bar chart and ROC curves are from
[`CLASSIFIER/notebooks/BASELINE/BASELINE_MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb`](../../CLASSIFIER/notebooks/BASELINE/BASELINE_MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb)
(confirmed by grepping the exact plot title strings). The "after" ROC pair is from
[`CLASSIFIER/notebooks/LONGITUDINAL/LONGITUDINAL_COMMON_DELCODE.ipynb`](../../CLASSIFIER/notebooks/LONGITUDINAL/LONGITUDINAL_COMMON_DELCODE.ipynb)'s
`plot_oof_test_roc` cell.

| | Run | GAAE checkpoint | Code path | CV AUC | Test AUC |
|---|---|---|---|---|---|
| "Before" | `CLASSIFIER/notebooks/checkpoints/checkpoints_gelstm_whole_brain/gelstm_2026-05-20_09-54-16` | `bright-disco-4_2026-05-07` | old hardcoded GELSTM script (pre-adapter) | 0.944 | 0.967 |
| "After" | `CLASSIFIER/outputs/gelstm-trajectory-whole-brain/runs/earnest-stream-9-d9655b0db-2026-06-21_01-10-19` | `ethereal-planet-16_2026-06-10` | `LONGITUDINAL_COMMON_DELCODE.ipynb` (adapter framework) | 0.944 (fold-mean) / 0.860 (pooled OOF) | 0.807 |

The exact numbers in both images match these two runs' `run_summary.json` to 4 decimal
places, so this is not a guess. Neither of these runs' configs are directly comparable,
though — see below.

## Confound 1: the GAAE encoder itself changed, not just the split

`bright-disco-4` (2026-05-07) and `ethereal-planet-16` (2026-06-10) are two different
trained encoders, a month apart. `bright-disco-4` has **no `run_config.json`** anywhere
in the current repo — zero recorded provenance (no hidden_dim, no split sizes, nothing).
`ethereal-planet-16` has full provenance. The "before" run's `run_summary.json` itself
predates the current schema entirely (a bare `hyperparams` dict, no `training_config`/
`dataset_info` keys), which independently confirms it comes from a materially older
version of the whole pipeline, not a same-code/different-split A/B.

`bright-disco-4`'s weight file does still physically exist, preserved in a legacy
snapshot: `/mnt/e/fyassine/_ad-early-detection/__CLASSIFIER__/CLASSIFIER_v1/checkpoints/
checkpoints_gaae_whole_brain/bright-disco-4_2026-05-07_20-17-11/model_bright-disco-4_2026-05-07_20-17-11.pth`
— so a true "same downstream code, old vs new encoder" comparison is possible later if
needed, without retraining anything. Not run as part of this investigation (out of
scope: investigate + explain only).

## Confound 2: downstream hyperparameters shrank in the same refactor as the split fix

Diffing `training_config` between a legacy-notebook run using the *same* GAAE checkpoint
as "after" (`copper-river-6-349c3823d-2026-06-19_23-38-25`, commit `349c3823d`, old
hardcoded script) against the "after" run (`earnest-stream-9`, commit `d9655b0db`, new
adapter):

| field | old (`349c3823d`) | new (`d9655b0db`) |
|---|---|---|
| `lstm_hidden` | 128 | 32 |
| `lstm_layers` | 2 | 1 |
| `classifier_hidden` | 64 | 32 |
| `standardize_features` | *(absent)* | `true` |
| `weight_decay` | *(absent)* | `0.0001` |
| `rnn_type` | *(absent)* | `lstm` |

Model capacity was cut substantially (roughly 4x fewer LSTM hidden units, half the
layers) and feature standardization + weight decay were added, all in the same commit
that introduced the adapter framework. Any of these can move AUC well beyond 0.1 on
their own. Notably, for *this specific pair*, test AUC went 0.53 → 0.81 (up, not down),
and the old-code/new-checkpoint run (`copper-river-6`) itself looks partially degenerate
(sensitivity=1.0, specificity=0.25 — a near-single-class classifier). That result should
not be read as "the honest pre-fix number" either; it needs its own explanation
(possible feature-scale mismatch against the newer encoder's latent distribution, before
`standardize_features` existed to correct for it) before being used as evidence of
anything.

## Confound 3: the historical split content is unrecoverable, but a concrete historical leak was found elsewhere

`DATA/DELCODE/__metadata__/SPLITS/` is gitignored; the on-disk CSVs were regenerated
2026-07-03 — two weeks after every GELSTM run examined here. `run_summary.json` for both
the old and new runs cites the same (now nonexistent) path
`DATA/DELCODE/SPLITS/downstream/*.csv` (missing `__metadata__/`), so matching path
strings prove nothing about matching subject content. This repo's git history was
squashed on 2026-06-22, so there's no commit to diff for "the fix" itself.

However, a sibling checkout at `/mnt/e/fyassine/_ad-early-detection` (separate `.git`,
last touched ~Feb 2026) contains the direct ancestor of today's split generators:
`data/Data-Delcode/create_gaae_data_splits.py` (predecessor of
`create_pretrain_data_splits.py`) and `create_gec_data_splits.py` (predecessor of
`create_downstream_data_splits.py`). Its docstring says *"GAAE test includes GEC test
patients for consistent holdout evaluation"* — reading the code confirms it reserves
GEC's **test** patients out of GAAE's free 60/20/20 split, but never reserves GEC's
**validation** patients. Verified directly against the on-disk split JSONs in that
directory (see `verify_legacy_split_overlap.py` alongside this doc):

```
GAAE train ∩ GEC test  = 0     (protected)
GAAE train ∩ GEC val   = 69 / 106   (65% of the downstream validation cohort was
                                      also in the encoder's own training set)
```

This is a real, on-disk, directly-verified leak — a validation-side leak specifically,
not the test-side one originally remembered, in what's most likely an earlier iteration
of the same lineage (script names, structure, and 60/20/20-with-reservation design are
identical to the current generators). Today's `create_pretrain_data_splits.py` fixes
exactly this class of bug by reserving **both** downstream val and downstream test into
pretrain's val and test respectively, and hard-asserting the disjointness — verified on
current on-disk CSVs: `pretrain_test ⊇ downstream_test`, `pretrain_val ⊇ downstream_val`,
zero overlap of either into `pretrain_train`.

The exact commit/date at which the *current* repo's `create_pretrain_data_splits.py`
started reserving val (not just test) could not be pinned down further — that file is
gitignored in every copy checked (`_ad-early-detection`, `ad-early-detection-backup`,
current repo), and `ad-early-detection-backup` (2026-06-22 snapshot) contains no `DATA/`
directory at all, only code.

## A second, independent, currently-live leak-adjacent bug (unrelated to the split question)

`CLASSIFIER/model/GELSTM/models.py`'s `freeze_encoder()` sets `requires_grad_(False)` on
the encoder but never puts its BatchNorm submodules into `.eval()`.
`CLASSIFIER/model/GELSTM/train.py` calls whole-model `.train()` every epoch, so
`encoder_bn1`/`encoder_bn2`'s running statistics keep updating on downstream CV batches
even though gradients are frozen. `CLASSIFIER/adapters/gec.py` and `gep.py` both call
`.eval()` after freezing; GELSTM's adapter doesn't. This doesn't touch the held-out test
set directly (CV pool is train+val only), but it means the "frozen" encoder is never
truly static across separate downstream runs — an extra, uncontrolled noise source when
comparing any two GELSTM runs, on top of confounds 1 and 2 above. Not fixed as part of
this investigation (documentation only, per scope).

## GEGRU: no "before" data point exists

GEGRU isn't a separate model — `LONGITUDINAL_COMMON_DELCODE.ipynb` covers it as `rnn_type:
"gru"` on the same `GELSTMAdapter`/`GELSTMClassifier` (`model_type` in `run_summary.json`
is still `"GELSTMClassifier"`). Checking every `gegru-trajectory-whole-brain` run
(`magic-stream-1-2e7594f8d`, `upbeat-water-2-d9655b0db`, `lucky-harbor-3-d9655b0db`) and
every `run_summary.json` anywhere under `CLASSIFIER/outputs/`: **none reference the old
`bright-disco-4` checkpoint** — every GEGRU run uses `ethereal-planet-16` (the "after"
encoder) and the adapter-framework code path. There is no pre-adapter, old-hardcoded-script
GEGRU run analogous to "before" GELSTM or to `copper-river-6` (old-code/new-checkpoint).

This means GEGRU cannot be used to isolate confound 2 (the split leak) at all — it never
ran under the leaky split regime, so there's no leak-era GEGRU number to compare against.
What it does confirm: the corrected `ethereal-planet-16` encoder is consistently shared
across all three downstream adapters (GEC/GELSTM/GEGRU) post-fix — all three known GEGRU
runs report byte-for-byte identical CV fold AUCs (`[0.932, 0.909, 0.972, 0.969, 0.921]`,
`best_fold=3`, `best_val_auc=0.9716`), i.e. GEGRU's own hyperparameters were never changed
across those three runs. This is *not* identical to GELSTM's "after" CV numbers
(`earnest-stream-9`: `[0.955, 0.920, 0.943, 0.981, 0.921]`, `best_fold=4`) — expected, since
`rnn_type` changes the recurrent cell itself (GRU vs LSTM), which is a legitimate source of
AUC variation distinct from the split/encoder confounds above, not a data point that speaks
to the leak question either way.

## Attribution

Given the above, the observed ~0.1-0.15 AUC drop between the "before" and "after" images
cannot be attributed to the split-leakage fix alone — it is a mix of at minimum three
factors that all changed in the same window (2026-05-20 to 2026-06-21):

1. **A genuine encoder swap** (`bright-disco-4` → `ethereal-planet-16`), plausibly
   itself motivated by the leak (retraining GAAE on a corrected pretrain split), but not
   verifiable as such from any artifact still on disk.
2. **A real leakage mechanism, confirmed to have existed in this lineage's earlier code**
   (val-side, not test-side): subjects used to train GAAE were also in the downstream
   validation fold in a directly-inspected historical artifact (69/106 overlap).
3. **A simultaneous model-capacity/regularization change** in the downstream adapter
   rewrite (smaller LSTM, added standardization and weight decay) bundled into the same
   commit — capable of moving AUC by a similar magnitude independent of any split issue.

Isolating factor 2's exact contribution from factors 1 and 3 would require the
controlled re-run described in the plan (same downstream code, old vs. new GAAE
checkpoint) — not executed here, since this pass was scoped to investigation and
write-up only.

## What to tell the supervisor

GAAE's loss never uses labels — that's correct. But reconstruction quality is not
label-dependent, and it is measurably better for subjects the encoder was directly
optimized on than for genuinely held-out ones (the standard transductive/memorization
gap in autoencoders). If a subject sits in both the GAAE training pool and a downstream
val/test fold, that subject's embedding is systematically "easier" for any downstream
classifier — purely a byproduct of reconstruction optimization, not label exposure. We
found this exact bug in an earlier version of our split-generation code: it protected
the downstream **test** set from encoder-training overlap, but not the downstream
**validation** set — 65% of the validation cohort was also in the encoder's training
data in that version. That alone is enough to inflate CV/validation-derived metrics even
with the test set nominally clean. That said, the ~0.1 drop we're seeing between our
specific before/after screenshots isn't a clean A/B on the split fix alone — the GAAE
checkpoint and the downstream model's hyperparameters both changed in the same window,
so the honest answer is "the leak is real and would inflate metrics, but this particular
before/after comparison also bundles in an encoder swap and a capacity/regularization
change, and we can't yet quote a number for the leak's isolated contribution."

## Hardening recommendations (documented, not implemented)

- `SHARED/sanity.py::run_full_audit()` only checks disjointness *within* whichever split
  dict it's given; every notebook calls it with `downstream` only, so a
  pretrain/downstream overlap regression would pass the mandatory audit silently today.
  `assert_cohort_policy(policy="disjoint")` already exists in the same file but isn't
  wired into that mandatory path.
- `BASELINE_MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb`'s `load_latest_summary()` picks
  "latest" via `sorted(glob(...))[-1]` on folder-name strings. With inconsistent naming
  (`gelstm_...` vs `gelstm_wholebrain_...`), this doesn't reliably pick the actual most
  recent run.
- No GAAE checkpoint or downstream run currently records a split-file hash or subject-ID
  list — this investigation would have been far more tractable if one did, and any
  future recurrence of this exact question will hit the same wall.
- `CLASSIFIER/model/GELSTM/models.py::freeze_encoder()` should put encoder submodules in
  `.eval()` the way `adapters/gec.py` and `adapters/gep.py` already do.
