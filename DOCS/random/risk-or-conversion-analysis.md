# Plan: Survival & Risk Analysis for Converter Trajectories

## Context

The user wants to extend the longitudinal analysis beyond trajectory visualization into quantitative risk and survival analysis. Key data facts (from `DATA/DELCODE/__v3__/metadata/cohorts.csv`):

- **68 converters** (MCI → AD), 58 with M0 baseline scans, longitudinal fMRI through M0–M84
- **99 MCI non-converters** (MCI-NC), baseline only — these are the right-censored controls
- Clinical data available at M0: MMSE 100%, CDR 100%, Aβ42/tau/p-tau ~64%, ApoE 100%
- fMRI metrics per visit: Global FC, DMN FC, Modularity Q, Density (from `__v3__/matrices`)
- Visits: M0, M12, M24, M36, M48, M60, M72, M84, M96, M108

**Core constraint**: only converters have longitudinal fMRI. MCI-NC is a static reference cloud. This is fine for survival analysis — the event (conversion) happens only to converters; MCI-NC are censored.

---

## Survival Analysis Framework

### 1. Outcome Definition

| Subject | event | time (months) |
|---------|-------|---------------|
| Converter | 1 | last_visit_num (approx. conversion time) |
| MCI-NC | 0 | last_observed_visit_num (censored) |

For converters, "last visit as converter" is the available proxy for conversion time (exact conversion date would be better but isn't in the CSV). The visit numbering (M12 = 12 months, etc.) maps directly to months.

### 2. Kaplan-Meier Survival Curves

Stratify converters + MCI-NC into groups by **baseline** fMRI metrics and plot time-to-conversion:

- **High vs. low baseline DMN FC** (split at median among converters)
- **High vs. low Modularity Q**
- **Aβ42 positive vs. negative** (threshold ≈ 500 pg/ml or ratio < 0.1)
- **ApoE ε4 carrier vs. non-carrier**

Test group differences with log-rank test. This directly answers: "do patients with lower baseline DMN FC convert faster?"

### 3. Cox Proportional Hazards Model

Baseline covariates → hazard of conversion:

```python
from lifelines import CoxPHFitter

covariates = [
    'global_fc',      # fMRI baseline
    'dmn_fc',
    'modularity',
    'density',
    'mmse_total',     # clinical baseline
    'cdr_global',
    'abeta42',        # CSF (subset, handle missing)
    'total_tau',
    'apoe_e4',        # binary: carries ε4
    'age',
    'sex',
]
# Output: hazard ratios + 95% CI + p-values
# Concordance index (C-statistic) measures discriminative power
```

**Important**: with only 68 converters, keep the model sparse. Univariate Cox first per biomarker, then a final multivariate model with ≤ 5–6 predictors (rule of thumb: 1 variable per ~10 events).

### 4. Trajectory Slope as Time-Varying Risk Predictor

For converters with ≥ 2 visits: compute per-patient slope of each fMRI metric (linear regression of metric vs. visit month). The slope measures **rate of decline**:

```python
# For each converter:
slope_dmnfc = np.polyfit(visit_months, dmn_fc_values, 1)[0]  # negative = declining
slope_mod   = np.polyfit(visit_months, modularity_values, 1)[0]
```

Include slopes as additional covariates in Cox model (for converters with ≥ 2 visits). This answers: "does faster DMN FC decline predict faster conversion?"

### 5. Progression Score (Manifold-Based)

Already described in the user's framework:

```python
# Axis from MCI-NC centroid → AD centroid in UMAP space
# Project each converter visit onto this axis
# Score at M0 is the "baseline risk position"
# Rate of score increase is the "progression velocity"
```

This score can replace or complement the Cox model as a single interpretable risk number.

### 6. Risk Score per Patient (Dashboard Integration)

From the Cox model coefficients, compute a scalar risk score per converter at M0:

```
risk_score = β₁·GlobalFC + β₂·DMNFC + β₃·Modularity + β₄·MMSE + ...
```

Show this in the patient modal: a risk gauge or percentile relative to the converter cohort.

---

## What Is NOT Feasible (Given Constraints)

- **Joint longitudinal-survival model** (e.g., `JMbayes2`): requires longitudinal measurements for BOTH converters AND non-converters. MCI-NC only has baseline → skip.
- **Competing risks analysis**: only one event type (AD conversion), no competing events coded in the data.
- **Exact conversion time**: the CSV has last-visit-as-converter, not the actual clinical conversion date. This introduces interval censoring. For now, treat last visit as conversion time.

---

## Implementation Plan

### Phase A — Data Preparation (`DATA/src/` or a new `MODEL/src/survival.py`)

1. Load CSV, filter to converters + MCI-NC
2. Compute `time` (last visit number) and `event` (1/0)
3. Extract M0 fMRI metrics from `__v3__/matrices` by subject ID
4. Merge into a single survival DataFrame
5. Compute per-converter fMRI slopes (for patients with ≥ 2 scans)

### Phase B — Analysis Notebook (`MODEL/notebooks/`)

1. Kaplan-Meier plots (stratified by fMRI quartile and CSF status)
2. Log-rank tests
3. Univariate Cox per biomarker (forest plot of HRs)
4. Multivariate Cox (sparse, guided by univariate selection)
5. Progression score computation and trajectory plots

### Phase C — Dashboard Integration (optional, later)

- Add risk score to patient modal (computed from Cox coefficients)
- Add population-level K-M curve to main dashboard

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `MODEL/src/survival_analysis.py` | Data prep + Cox/KM utilities |
| `MODEL/notebooks/survival_analysis.ipynb` | Full analysis notebook |
| `DASHBOARD/app/static/app.js` | (Phase C only) risk score in patient modal |
| `DASHBOARD/app/biomarkers.py` | (Phase C only) serve pre-computed scores |

---

## Verification

1. K-M curve should show converters with lower baseline DMN FC surviving (time-to-conversion) shorter — i.e., converting faster.
2. Cox model C-statistic should be > 0.6 to be useful (chance = 0.5).
3. Aβ42 and ApoE ε4 should show significant HRs (literature-established predictors).
4. Progression scores should increase monotonically on average across converter visits.
