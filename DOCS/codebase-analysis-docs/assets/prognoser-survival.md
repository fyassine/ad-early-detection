# PROGNOSER Survival Diagrams

Supplemental diagrams for [`../CODEBASE_KNOWLEDGE.md`](../CODEBASE_KNOWLEDGE.md).
Paths are relative to the repository root.

---

## 1. Symmetric at-risk window (no look-ahead)

Source: `PROGNOSER/common/survival_table.py` (module docstring).

```mermaid
flowchart LR
    subgraph Converter["Converter (event = 1)"]
        C0["baseline visit"] --> Cend["first AD visit<br/>= window_end, T"]
    end
    subgraph NonConverter["Non-converter (event = 0, censored)"]
        N0["baseline visit"] --> Nend["last MCI visit<br/>= window_end, T"]
    end
    note["ALL features (clinical slopes, GAAE embeddings,<br/>sequences) computed strictly within [baseline, window_end)"]
```

---

## 2. Survival pipeline

```mermaid
flowchart TB
    META["cohorts.csv"] --> ST["build_survival_table()<br/>→ (T, E, covariates)"]
    EMB["GAAE embeddings (cached)<br/>src/build_subject_embeddings.py"] --> FEAT["feature assembly<br/>by embedding strategy"]
    ST --> FEAT
    FEAT --> MODELS

    subgraph MODELS["model/ (SurvivalModel ABC in base.py)"]
        KM["kaplan_meier.py"]
        COX["cox.py · cox_time_varying.py"]
        RSF["rsf.py"]
        LSTM["lstm_surv.py"]
        DS["deepsurv.py (optional)"]
    end

    MODELS --> METRICS["common/metrics.py<br/>C-index · IBS · time-dependent AUC"]
```

---

## 3. Embedding strategies (which visits feed the model)

```mermaid
flowchart LR
    V["visits in [baseline, window_end)"] --> B["baseline: M0 only"]
    V --> L["last: latest visit"]
    V --> M["mean: average over window"]
    V --> S["slope: last − baseline"]
    V --> A["all_aggs: concat of the four"]
    V --> Q["sequence: ordered series → LSTM-Surv"]
```
