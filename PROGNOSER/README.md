# PROGNOSER

Time-to-conversion (MCI → AD) survival analysis pipeline. Predicts when an MCI subject will convert, not just *whether* — complements the binary classifier in `CLASSIFIER/`.

## Quick Start

```bash
# 1. Install survival packages (one-time)
pip install -r PROGNOSER/requirements.txt

# 2. Population-level Kaplan-Meier (no model fitting)
jupyter notebook PROGNOSER/notebooks/KAPLAN_MEIER_BASELINE.ipynb

# 3. Clinical-only Cox PH (no GAAE embeddings needed)
#    Edit EXPERIMENT["method"] = "cox_clinical" in PROGNOSER_RUNNER.ipynb, run all
jupyter notebook PROGNOSER/notebooks/PROGNOSER_RUNNER.ipynb

# 4. Precompute GAAE embeddings (once per network combo)
python -m PROGNOSER.src.build_baseline_embeddings --combo dmn_hippo
# or all 8 combos:
python -m PROGNOSER.src.build_baseline_embeddings --all

# 5. Sweep methods × combos via PROGNOSER_RUNNER.ipynb, then:
jupyter notebook PROGNOSER/notebooks/CROSS_NETWORK_COMPARISON.ipynb
```

See [DOCS/prognosis_pipeline.md](../DOCS/prognosis_pipeline.md) for the full guide.

## Layout

```
PROGNOSER/
├── common/
│   ├── survival_table.py    # build (subject_id, T, E, covariates) from cohorts.csv
│   ├── metrics.py           # C-index, integrated Brier score, time-dependent AUC
│   └── embeddings.py        # extract baseline GAAE embeddings → 64-dim per subject
├── model/
│   ├── base.py              # SurvivalModel ABC
│   ├── kaplan_meier.py      # population KM
│   ├── cox.py               # CoxPH (lifelines), with 3 feature factories
│   ├── rsf.py               # Random Survival Forest (sksurv)
│   └── deepsurv.py          # DeepSurv neural Cox (pycox, optional)
├── src/
│   └── build_baseline_embeddings.py    # CLI to precompute & cache embeddings
├── notebooks/
│   ├── KAPLAN_MEIER_BASELINE.ipynb
│   ├── PROGNOSER_RUNNER.ipynb          # parameterized: choose combo × method
│   └── CROSS_NETWORK_COMPARISON.ipynb  # leaderboard
└── requirements.txt
```

## Data

Reuses `DATA/DELCODE/__v3__/metadata/`:
- `cohorts.csv` — longitudinal visits, `diagnosis` column tracks MCI → AD over time
- `splits_gec/{train,val,test}.csv` — same subject splits as the classifier (consistent comparison)

## Methods

| Method | Library | Notes |
|---|---|---|
| `km` | lifelines | Population baseline, no covariates |
| `cox_clinical` | lifelines | age, sex, MMSE, CDR, ApoE4 |
| `cox_embedding` | lifelines | 64-dim GAAE embedding (PCA→16) |
| `cox_combined` | lifelines | clinical + embedding |
| `rsf` | scikit-survival | non-linear baseline |
| `deepsurv` | pycox (stretch) | neural Cox |
