# PROGNOSER

Time-to-conversion (MCI → AD) longitudinal survival analysis pipeline.

Predicts *when* an MCI subject will convert — complements the binary CLASSIFIER.
Supports both converters and non-converters with the **same longitudinal logic**,
restricted to each subject's at-risk window.

## Quick Start

```bash
# 1. Install packages (one-time)
pip install -r PROGNOSER/requirements.txt

# 2. Reprocess all follow-up fMRI visits into __v3__/matrices/
python -m CLASSIFIER.src.processing.run_all_processing --reprocess-followups

# 3. Population-level KM (no model fitting)
jupyter notebook PROGNOSER/notebooks/KAPLAN_MEIER_BASELINE.ipynb

# 4. Clinical-only Cox baseline
#    Set method='cox_clinical' in PROGNOSER_RUNNER.ipynb, run all cells

# 5. Precompute longitudinal GAAE embeddings (once per combo × strategy)
python -m PROGNOSER.src.build_subject_embeddings --all --strategy all_aggs

# 6. Sweep methods × combos via PROGNOSER_RUNNER.ipynb, then view leaderboard:
jupyter notebook PROGNOSER/notebooks/CROSS_NETWORK_COMPARISON.ipynb
```

See [DOCS/prognosis_pipeline.md](../DOCS/prognosis_pipeline.md) for the complete guide.

## At-Risk Window (symmetric handling)

Both converters and non-converters are processed with the **same code path**:

```
window_end = first_AD_visit_month      (converter, event=1)
           OR last_MCI_visit_month     (non-converter, event=0)
```

All feature extraction (clinical slopes, GAAE embeddings, visit sequences) uses
only visits within `[M0, window_end)` — no look-ahead bias, no asymmetric treatment.

## Methods

| Method | `method` key | Features |
|---|---|---|
| Kaplan-Meier | `km` | None |
| Cox clinical | `cox_clinical` | age, sex, MMSE, CDR, ApoE4 |
| Cox clinical + trajectories | `cox_clinical_longitudinal` | + MMSE/CDR slope/delta |
| Cox embedding | `cox_embedding` | 64-dim GAAE (strategy-selected) |
| Cox combined | `cox_combined` | clinical + embedding |
| Time-varying Cox | `cox_time_varying` | per-visit long-format |
| Random Survival Forest | `rsf` | any feature set |
| DeepSurv | `deepsurv` | any (requires pycox) |
| LSTM sequence model | `lstm_surv` | visit-sequence features |

## Embedding Strategies

| Strategy | Visits used |
|---|---|
| `baseline` | M0 only |
| `last` | Latest visit < window_end |
| `mean` | Mean across window |
| `slope` | last − baseline |
| `all_aggs` | Concat of all four |
| `sequence` | Time series for LSTM |

## Layout

```
PROGNOSER/
├── common/
│   ├── survival_table.py      # build (T, E, covariates, longitudinal aggregates)
│   ├── longitudinal.py        # at-risk window, aggregator, long-format builder
│   ├── embeddings.py          # multi-visit GAAE embedding extraction
│   ├── metrics.py             # C-index, IBS, time-dependent AUC
│   └── io.py                  # run_summary.json helpers
├── model/
│   ├── base.py                # SurvivalModel ABC
│   ├── kaplan_meier.py
│   ├── cox.py                 # CoxPHWrapper (3 factories)
│   ├── cox_time_varying.py    # CoxTimeVaryingWrapper (lifelines)
│   ├── rsf.py
│   ├── lstm_surv.py           # LSTMSurvWrapper (PyTorch)
│   └── deepsurv.py            # DeepSurvWrapper (pycox, optional)
├── src/
│   ├── build_subject_embeddings.py    # CLI: compute + cache embeddings per strategy
│   └── build_baseline_embeddings.py   # legacy shim (deprecated)
├── notebooks/
│   ├── KAPLAN_MEIER_BASELINE.ipynb
│   ├── PROGNOSER_RUNNER.ipynb          # parameterized: combo × method × strategy
│   └── CROSS_NETWORK_COMPARISON.ipynb  # leaderboard heatmap
└── requirements.txt
```
