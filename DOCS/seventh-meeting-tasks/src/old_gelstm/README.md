# Frozen legacy GELSTM code (pre-adapter)

Copied verbatim from `/mnt/e/fyassine/_ad-early-detection/__CLASSIFIER__/CLASSIFIER_v1/`
(a separate checkout, not part of this repo's git history). This is the exact,
hardcoded, pre-adapter script that produced the "before" checkpoint
(`checkpoints_gelstm_whole_brain/gelstm_2026-05-20_09-54-16`), trained against the
`bright-disco-4_2026-05-07` GAAE checkpoint and the historical leaky pretrain split
(see `../splits/legacy/`).

Confirmed match: `GELSTM_DELCODE_WHOLE_BRAIN.ipynb`'s hardcoded hyperparameter globals
(`LSTM_HIDDEN=128`, `LSTM_LAYERS=2`, `LSTM_DROPOUT=0.3`, `CLASSIFIER_HIDDEN=64`,
`FREEZE_ENCODER=True`, `GRAPH_POOL="mean"`, `LEARNING_RATE=0.001`, `EPOCHS=50`,
`BATCH_SIZE=16`, `N_FOLDS=5`) are byte-identical to the `hyperparams` dict saved in the
"before" run's `run_summary.json` (see `../run_summaries/before.json`).

**Not part of the active pipeline.** Do not adapt this into new code (see
`.claude/rules/architecture.md` — legacy code is not a pattern to copy). Kept only so the
"before" numbers in `gaae-downstream-leakage-investigation.md` have a citable, inspectable
source. Not wired into `scripts/run_checks.py`; not linted/tested as part of CI.
