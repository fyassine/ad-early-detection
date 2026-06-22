# DASHBOARD Architecture Diagrams

Supplemental diagrams for [`../CODEBASE_KNOWLEDGE.md`](../CODEBASE_KNOWLEDGE.md).
Paths are relative to the repository root.

---

## 1. Request/response topology

```mermaid
flowchart TB
    subgraph Browser["Vite frontend (DASHBOARD/frontend/)"]
        UI["3 tabs: Population · Cohort · Patient"]
    end
    subgraph Backend["FastAPI app (DASHBOARD/app/)"]
        MAIN["main.py<br/>routers + lifespan + watchdog"]
        ROUTES["routes/*.py"]
        SERVICES["services/*.py<br/>(gelstm, graph_metrics, atn, ebm, ...)"]
        STATS["cohort_stats.py<br/>(UMAP, biomarkers, EBM, brain-age)"]
    end
    subgraph Disk["Cache + artifacts"]
        CACHE["DASHBOARD/.cache/<br/>cohort_stats · gelstm · jobs · dfc"]
        CKPT["CLASSIFIER/model/GELSTM/checkpoints/<br/>best_model_fold*.pth · gaae_encoder.pth"]
        DATAR["DATA/DELCODE/ (matrices + cohorts.csv)"]
    end

    UI -->|GET /api/...| ROUTES
    ROUTES --> SERVICES
    ROUTES --> STATS
    SERVICES --> CKPT
    STATS --> DATAR
    SERVICES --> CACHE
    STATS --> CACHE
    MAIN -->|serves built SPA| UI
```

---

## 2. Detached precompute job lifecycle

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant API as routes/cohort.py
    participant JM as services/job_manager.py
    participant PC as precompute.py (subprocess)
    participant WD as watchdog thread

    FE->>API: GET /api/cohort/... (cache miss)
    API->>JM: start_job(csv, folders)
    JM->>PC: spawn detached (start_new_session=True)
    Note over PC: stages: cohort_stats → graph_metrics<br/>→ gelstm → qc → dynamic_fc
    PC->>PC: write .cache/jobs/<id>.json (progress)
    FE->>API: poll job status
    WD->>JM: every WATCHDOG_INTERVAL_S sweep_runaway_jobs()
    Note over WD: kill if age > MAX_JOB_AGE_S<br/>or stale > STALL_THRESHOLD_S
```

---

## 3. GELSTM ensemble inference path

```mermaid
flowchart LR
    M["FC matrices per visit<br/>(.npz)"] --> SVC["services/gelstm.py"]
    GA["gaae_encoder.pth"] --> SVC
    F1["best_model_fold1..5.pth"] --> SVC
    SVC -->|model_version = SHA1(checkpoints)| CACHE["gelstm/predictions_*.pkl"]
    SVC -->|mean ± CI over folds| OUT["P(MCI→AD) per subject"]
    note["Reuses project-root .venv<br/>(torch / torch_geometric / nilearn)"]
    SVC -.-> note
```

> **Note:** The deep internals of the 5-stage precompute and the GELSTM service
> (exact function names, dFC k-means parameters) are documented from secondary
> reading; verify against `DASHBOARD/app/precompute.py` and
> `DASHBOARD/app/services/gelstm.py` before relying on specific signatures.
