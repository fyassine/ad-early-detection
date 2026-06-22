# Prognosis Pipeline

End-to-end guide for running the time-to-conversion (MCI → AD) survival pipeline at `PROGNOSER/`.

---

## 1. Goal

CLASSIFIER answers *will this subject convert?* PROGNOSER answers *when, and how does prognosis depend on which brain network we use?*

The pipeline supports longitudinal data for **both converters and non-converters** using a symmetric at-risk window:

```
window_start = M0
window_end   = months_to_first_AD_visit   (converter, event=1)
             OR months_to_last_MCI_visit  (non-converter, event=0)
```

All feature extraction (clinical aggregates, embeddings, visit sequences) is restricted to visits within `[window_start, window_end)` — the same code path for both groups.

---

## 2. Data

| File | Used for |
|---|---|
| `DATA/DELCODE/__v1__/fmri/sub-*/` | Raw BOLD NIfTIs for all visits (M0–M108) |
| `DATA/DELCODE/__v3__/matrices/` | Per-visit FC matrices (Schaefer 200, all visits after reprocessing) |
| `DATA/DELCODE/__v3__/metadata/cohorts.csv` | Longitudinal visits, `diagnosis` tracks MCI → AD |
| `DATA/DELCODE/__v3__/metadata/splits_gec/{train,val,test}.csv` | Subject splits (same as CLASSIFIER) |
| `DATA/DELCODE/__v{4-11}__/matrices/` | Per-visit FC matrices for each network combo |
| `CLASSIFIER/notebooks/checkpoints_gaae_{combo}/` | Trained GAAE encoders (prerequisite) |

### Censoring rules

Include subjects whose **earliest non-NaN diagnosis** is `mci` or `converter`. Then:
- `event_observed = 1`, `duration = months to first 'ad' visit`
- `event_observed = 0`, `duration = months of last MCI visit` (right-censored)

Both groups follow the **same code path** in `build_survival_table()`.

---

## 3. Methods

| Method | Library | Features | Notes |
|---|---|---|---|
| `km` | lifelines | None | Population KM — no covariates |
| `cox_clinical` | lifelines | age, sex, MMSE, CDR, ApoE4 at M0 | Literature baseline |
| `cox_clinical_longitudinal` | lifelines | clinical + MMSE/CDR slope/delta | Extends Cox with trajectories |
| `cox_embedding` | lifelines + PCA | 64-dim GAAE (strategy-selected) | Connectivity signal alone |
| `cox_combined` | lifelines + PCA | clinical + embedding | Main comparison |
| `cox_time_varying` | lifelines CoxTVF | per-visit (start, stop) long-format | Proper longitudinal survival |
| `rsf` | scikit-survival | any feature set | Non-linear baseline |
| `deepsurv` | pycox (optional) | any feature set | Neural Cox |
| `lstm_surv` | PyTorch LSTM | visit-sequence clinical/embedding | Full longitudinal sequence model |

### Embedding strategies

For methods that use GAAE embeddings, the `embedding_strategy` controls which visit(s) are encoded:

| Strategy | Which visit(s) | Notes |
|---|---|---|
| `baseline` | M0 only | Backward-compatible |
| `last` | Latest visit < `window_end` | Closest to event/censoring |
| `mean` | All visits in window | Average connectivity profile |
| `slope` | `last - baseline` vector | Connectivity change direction |
| `all_aggs` | Concat of all four | 4×latent_dim features |
| `sequence` | All visits as time series | For LSTM training |

---

## 4. End-to-End Run Order

### Prerequisites — train GAAE encoders

For each network combo, run `CLASSIFIER/notebooks/NETWORK_GAAE_RUNNER.ipynb` once (already covered by the network-framing pipeline).

### Step A — Install packages

```bash
pip install -r PROGNOSER/requirements.txt
```

### Step B — Reprocess all follow-up visits

This populates `__v3__/matrices/` with M12/M24/M36... FC matrices (currently mostly M0):

```bash
python -m CLASSIFIER.src.processing.run_all_processing --reprocess-followups
```

After this runs, re-run subset scripts to propagate follow-ups to `__v6__`, `__v7__`, `__v9__`:

```bash
python -m CLASSIFIER.src.processing.run_all_processing --skip-schaefer  # just update subsets
# or simply:
python -m CLASSIFIER.src.processing.run_all_processing  # runs both in sequence
```

**Verification**: confirm `DATA/DELCODE/__v3__/matrices/sub-95c562d3e_ses-01_M12_*` exists.

### Step C — Population-level Kaplan-Meier

```bash
jupyter notebook PROGNOSER/notebooks/KAPLAN_MEIER_BASELINE.ipynb
```

### Step D — Clinical-only Cox baseline (no GAAE needed)

Set `method='cox_clinical'`, `feature_set='clinical'` in `PROGNOSER_RUNNER.ipynb` and run.
This gives the reference C-index (~0.65–0.72) for network-augmented runs to beat.

### Step E — Precompute GAAE embeddings

```bash
# One combo, last-visit strategy:
python -m PROGNOSER.src.build_subject_embeddings --combo dmn_hippo --strategy last

# All combos, all aggregations (takes longer but maximises feature options):
python -m PROGNOSER.src.build_subject_embeddings --all --strategy all_aggs

# Sequence embeddings for LSTM:
python -m PROGNOSER.src.build_subject_embeddings --all --strategy sequence
```

Cached to `PROGNOSER/notebooks/_embeddings_cache_/{combo}_{strategy}_embeddings.parquet`.

### Step F — Method × combo sweep

For each `(method, network_combo)` combination, edit `EXPERIMENT` in `PROGNOSER_RUNNER.ipynb`
and run all cells. Each run saves to `checkpoints_prognoser_{combo}/{run_name}/`.

Suggested sweep order:
1. `cox_clinical` (baseline, no embeddings)
2. `cox_clinical_longitudinal` (adds MMSE/CDR trajectories)
3. `cox_combined` with `strategy='last'` per combo
4. `cox_time_varying` (per-visit long format)
5. `rsf` with `feature_set='clinical_longitudinal'`
6. `lstm_surv` (requires sequence embeddings)

### Step G — Cross-network leaderboard

```bash
jupyter notebook PROGNOSER/notebooks/CROSS_NETWORK_COMPARISON.ipynb
```

Produces `_artifacts_/leaderboard_heatmap.png` and `_artifacts_/leaderboard_all_runs.csv`.

---

## 5. Reading `run_summary.json`

```json
{
  "run_name": "cox_time_varying_clinical_longitudinal_last_2026-05-08_...",
  "method": "cox_time_varying",
  "feature_set": "clinical_longitudinal",
  "embedding_strategy": "last",
  "n_features": 13,
  "feature_columns": ["age", "sex", "mmstot", "cdrglobal", "apoe4",
                      "mmstot_baseline", "mmstot_last", "mmstot_slope", "mmstot_delta",
                      "cdrglobal_baseline", "cdrglobal_last", "cdrglobal_slope", "cdrglobal_delta"],
  "n_train": 51, "n_val": 10, "n_test": 8, "n_events_test": 6,
  "metrics": {
    "train": {"c_index": 0.72, "ibs": 0.16, "auc": {"24": 0.74, "36": 0.71, "60": 0.68}},
    "val":   {...},
    "test":  {...}
  }
}
```

---

## 6. Symmetric Longitudinal Handling

The key change from the baseline implementation: **all feature extraction uses the at-risk window**, computed identically for both groups:

```python
# In survival_table.py — same code path, different endpoint
window_end = first_AD_visit_month     # converter (event=1)
window_end = last_MCI_visit_month     # non-converter (event=0)

# All aggregates use only visits < window_end
mmstot_slope = agg.slope(subject_id, 'mmstot', window_end)  # ← same function call
z_last       = embedding(subject_id, last_visit_before=window_end)
```

This avoids:
- **Look-ahead bias**: no post-event data leaks into features for converters
- **Asymmetric information**: non-converters get the same window treatment as converters

---

## 7. Deferred Methods

| Method | Why deferred | What it needs |
|---|---|---|
| DeepHit | Multi-event competing risks | Competing-risks formulation |
| Joint longitudinal-survival | Shared latent process | Per-visit latent state model |
| Neural ODE (BrainODE, DISCLOSE) | Continuous-time trajectories | Irregular-time ODE solver |
| Subtyping/kernel (DCSM, MCI-CPS) | Unsupervised subtype discovery | Multi-modal features |

All share the prerequisite of a complete per-visit feature pipeline, which is now in place after Phase 0 reprocessing — these can be added as new `SurvivalModel` subclasses without changing the runner architecture.
