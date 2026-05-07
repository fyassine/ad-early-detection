# Prognosis Pipeline

End-to-end guide for running the time-to-conversion (MCI → AD) survival pipeline at `PROGNOSER/`.

---

## 1. Goal

CLASSIFIER answers *will this subject convert?* PROGNOSER answers *when, and how does prognosis depend on the network we use?*

Specifically, the pipeline:
1. Builds per-subject `(duration, event_observed)` tuples from `cohorts.csv` longitudinal visits
2. Trains baseline survival models (Kaplan-Meier, Cox PH, Random Survival Forest, DeepSurv) on (clinical) + (optional GAAE network embeddings)
3. Compares the 8 network combinations (DMN, hippocampus, limbic, DAN, and combinations) by **test C-index**, **integrated Brier score**, and **time-dependent AUC at 24/36/60 months**

---

## 2. Data

| File | Used for |
|---|---|
| `DATA/DELCODE/__v3__/metadata/cohorts.csv` | Per-visit `diagnosis` to compute (T, E) per subject |
| `DATA/DELCODE/__v3__/metadata/splits_gec/{train,val,test}.csv` | Same subject splits as the classifier — for consistent comparison |
| `DATA/DELCODE/__v{4-11}__/matrices/*.npz` | Baseline FC matrices for GAAE embedding extraction |
| `CLASSIFIER/notebooks/checkpoints_gaae_{combo}/{run_name}/model_*.pth` | Trained GAAE encoders (prerequisite) |

### Censoring rules

A subject is included if their **earliest non-NaN diagnosis** in `cohorts.csv` is `mci` or `converter`. Then:
- `event_observed = 1`, `duration = months_to_first_AD_visit` if any visit has `diagnosis == 'ad'`
- `event_observed = 0`, `duration = months_to_last_visit` otherwise (right-censored)

Implementation: `PROGNOSER/common/survival_table.py:build_survival_table()` (ported and extended from `DASHBOARD/app/services/survival.py:39-92`).

---

## 3. Methods

| Method | Library | Features | When to use |
|---|---|---|---|
| `km` | lifelines KaplanMeierFitter | none | Population reference, stratified plots |
| `cox_clinical` | lifelines CoxPHFitter | age, sex, MMSE, CDR, ApoE4 | Clinical baseline (literature C-index ~0.65–0.72) |
| `cox_embedding` | lifelines + PCA | 64-dim GAAE → PCA(16) | Test if connectivity carries prognosis signal alone |
| `cox_combined` | lifelines + PCA | clinical + 64-dim GAAE | Should beat clinical-only if H2 holds |
| `rsf` | scikit-survival RandomSurvivalForest | clinical / embedding / combined | Non-linear baseline; handles interactions |
| `deepsurv` | pycox CoxPH + MLPVanilla | any | Stretch — install `pycox torchtuples` first |

---

## 4. End-to-End Run Order

### Prerequisite — Train GAAE encoders (already covered by the network framing pipeline)

For each combo, run `CLASSIFIER/notebooks/NETWORK_GAAE_RUNNER.ipynb` once. Each run produces `CLASSIFIER/notebooks/checkpoints_gaae_{combo}/{run_name}/model_{run_name}.pth`. Without these, only the clinical-only methods will work.

### Step A — Install survival packages

```bash
pip install -r PROGNOSER/requirements.txt
```

### Step B — Population-level Kaplan-Meier

```bash
jupyter notebook PROGNOSER/notebooks/KAPLAN_MEIER_BASELINE.ipynb
# Run all cells. Outputs:
#   _artifacts_/km_overall.png
#   _artifacts_/km_by_apoe4.png
#   _artifacts_/km_by_sex.png
#   _artifacts_/km_by_baseline_diagnosis.png
#   _artifacts_/km_curves.json
```

### Step C — Clinical-only Cox baseline (no embeddings needed)

Edit `PROGNOSER/notebooks/PROGNOSER_RUNNER.ipynb` config cell:

```python
EXPERIMENT = {
    "network_combo": "dmn",          # arbitrary placeholder; ignored for clinical-only
    "data_version": "__v4__",
    "file_suffix": "...",
    "method": "cox_clinical",
    "feature_set": "clinical",
    ...
}
```

Run all cells. Test C-index in `_artifacts_/run_summary.json` is your reference baseline for the network-augmented runs to beat.

### Step D — Precompute GAAE embeddings (once per combo)

```bash
# One combo:
python -m PROGNOSER.src.build_baseline_embeddings --combo dmn_hippo

# All 8 (skips combos with no GAAE checkpoint):
python -m PROGNOSER.src.build_baseline_embeddings --all
```

Cached parquets land in `PROGNOSER/notebooks/_embeddings_cache_/{combo}_baseline_embeddings.parquet`. The runner notebook auto-loads these.

### Step E — Method × combo sweep

For each (combo, method) pair, edit `EXPERIMENT` in `PROGNOSER_RUNNER.ipynb` and run all cells. Each run saves:

```
PROGNOSER/notebooks/checkpoints_prognoser_{combo}/{run_name}/
├── run_summary.json     # all metrics + config
├── model_{run_name}.joblib
└── predictions_test.csv
```

### Step F — Cross-network leaderboard

```bash
jupyter notebook PROGNOSER/notebooks/CROSS_NETWORK_COMPARISON.ipynb
```

Walks all checkpoint dirs, ranks (combo × method) by test C-index, and writes `_artifacts_/leaderboard_*.csv` and `_artifacts_/leaderboard_barplot.png`.

---

## 5. Reading `run_summary.json`

```json
{
  "run_name": "cox_clinical_clinical_2026-05-08_14-30-00",
  "method": "cox_clinical",
  "feature_set": "clinical",
  "experiment": { "network_combo": "...", "data_version": "...", "...": "..." },
  "n_features": 5,
  "feature_columns": ["age", "sex", "mmstot", "cdrglobal", "apoe4"],
  "n_train": 51,  "n_val": 10,  "n_test": 8,  "n_events_test": 6,
  "metrics": {
    "train": {"c_index": 0.66, "ibs": 0.18, "auc": {"24": 0.71, "36": 0.69, "60": 0.65}},
    "val":   {...},
    "test":  {...}
  },
  "eval_times": [12, 24, 36, 48, 60, 72]
}
```

The leaderboard notebook tabulates these into a tidy DataFrame for ranking.

---

## 6. Known Limitations

- **Small test set (n=8 events).** Test C-index is high-variance — interpret with confidence intervals via bootstrap if you need formal claims.
- **Baseline-only features.** Time-varying covariates (multi-visit MMSE/CDR trajectories) are not used. This rules out joint longitudinal-survival, RNN/LSTM, and DeepHit dynamic-risk models.
- **GAAE encoders are unsupervised.** Trained on autoencoder reconstruction, not conversion prediction. A supervised encoder (or fine-tuning the GAAE on the survival loss) could add signal.

---

## 7. Deferred Methods (Future Work)

| Method | Why deferred | What it needs |
|---|---|---|
| DeepHit | Discrete-time multi-event; needs interval discretization + competing-risks setup | Same data + binning strategy |
| Joint longitudinal-survival | Per-visit feature trajectories + landmarking | Multi-visit feature tensors |
| RNN/LSTM | Sequence model over visits | Per-visit feature sequences |
| Neural ODE (BrainODE, DISCLOSE, LaTiM) | Continuous-time trajectory modeling | Multi-visit imaging features in a structured tensor |
| Subtyping/kernel (DCSM, MCI-CPS) | Subtype discovery + kernel survival | Multi-modal features (PET + MRI + CSF) |

These all share a common prerequisite: a **per-visit feature pipeline** (not just baseline). When that's built, swap in the corresponding model wrapper at the same `SurvivalModel` interface used here — the runner notebook architecture is designed to absorb new methods.
