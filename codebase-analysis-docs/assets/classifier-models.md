# CLASSIFIER Model Diagrams

Supplemental diagrams for [`../CODEBASE_KNOWLEDGE.md`](../CODEBASE_KNOWLEDGE.md).
Paths are relative to the repository root.

---

## 1. Model lineage: GAAE → GEC / GELSTM

```mermaid
flowchart TB
    subgraph GAAE["GAAE — model/GAAE/ (pretrained once, frozen)"]
        ENC["GATv2 encoder (3 layers)<br/>+ FiLM conditioning (age, sex)"]
        DECF["feature decoder"]
        DECA["InnerProductDecoder (adjacency)"]
        ENC --> DECF
        ENC --> DECA
    end

    ENC -->|encoder weights loaded + frozen| GEC
    ENC -->|encoder weights loaded + frozen| GELSTM

    subgraph GEC["GEC — model/GEC/ (static / flattened)"]
        GENC["GAAE encoder"] --> GPOOL["global_mean_pool"] --> GHEAD["MLP head → logit"]
    end

    subgraph GELSTM["GELSTM — model/GELSTM/ (longitudinal)"]
        LENC["GAAE encoder per visit"] --> LSEQ["[z_t ‖ Δt_t] sequence"]
        LSEQ --> LLSTM["LSTM (hidden=128, layers=2)"] --> LHEAD["MLP head → logit"]
    end
```

Key files:
- `CLASSIFIER/model/GAAE/models.py`, `train.py`, `losses.py`
- `CLASSIFIER/model/GEC/models.py`, `train.py` (`train_classifier`, `evaluate_classifier`)
- `CLASSIFIER/model/GELSTM/models.py`, `train.py` (`train_model`, `evaluate`)
- Encoder transfer: `CLASSIFIER/common/utils.py::load_frozen_encoder_from_gaae`

---

## 2. Training + evaluation flow (leakage-safe threshold)

```mermaid
sequenceDiagram
    participant NB as Notebook
    participant SP as common/splits.py
    participant TR as model/**/train.py
    participant EV as evaluate(_classifier)
    participant CK as common/provenance.py

    NB->>SP: make_splits(subject_ids, labels, seed)
    SP-->>NB: {train, val, test} (subject-disjoint)
    NB->>TR: train_model(..., cfg, eval_cfg, rng)
    loop each epoch
        TR->>EV: evaluate(val) → AUC, best_threshold (Youden/F1)
    end
    TR->>CK: save_full_checkpoint(best by val AUC,<br/>incl. best_threshold + rng_state)
    NB->>EV: evaluate(test, threshold=best_threshold)
    Note over EV: raises if threshold is None<br/>(never derived from test data)
```

---

## 3. Experiment families (notebook prefixes)

```mermaid
flowchart LR
    STATIC["STATIC_<br/>per-scan: GAAE pretrain, LogReg"]
    BASELINE["BASELINE_<br/>model comparison tables"]
    LONG["LONGITUDINAL_<br/>GELSTM, GELSTM+FDR, GEC trajectory"]
    SANITY["SANITY_<br/>split hygiene, metadata floor, ablations"]
    COMPARISON["COMPARISON_<br/>cross-region stats"]

    STATIC -->|encoder| LONG
    STATIC -->|embeddings| BASELINE
    LONG --> COMPARISON
    SANITY -.->|gates trust in| LONG
```
