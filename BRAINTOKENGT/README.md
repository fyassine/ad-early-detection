# BRAINTOKENGT — competitor baseline

Brain-TokenGT (Dong et al., *Beyond the Snapshot: Brain Tokenized Graph Transformer
for Longitudinal Brain Functional Connectome Embedding*, MICCAI 2023;
[arXiv:2307.00858](https://arxiv.org/abs/2307.00858)) wired into the same
experiment / notebook / registry machinery as `CLASSIFIER/` and `PROGNOSER/`, so it
can be compared head-to-head with GELSTM on the DELCODE MCI-vs-converter task under
an identical protocol.

```
BRAINTOKENGT/
├── model/
│   ├── transformer.py   # port of upstream model_transformer.py (GIVE + BIGTR)
│   ├── grcu.py          # port of upstream model_grcu.py (EvolveGCN-H / INE)
│   └── sequences.py     # DELCODE record -> (A_list, Nodes_list)
├── adapter.py           # BrainTokenGTAdapter — the six-hook contract
├── configs/             # as-released / repaired / smoke
├── experiments/         # registry (same schema as CLASSIFIER/experiments)
├── notebooks/           # LONGITUDINAL_BRAINTOKENGT_DELCODE.ipynb
├── tests/               # upstream-equivalence regression test
├── outputs/             # runs, RESULTS.csv (gitignored)
└── run_experiment.py
```

`Brain-TokenGT/` at the repo root is the **pristine upstream checkout**. Do not edit
it — it is the reference the equivalence test compares against, and it is
snapshotted into every run's `source/`.

---

## Quick start

```bash
source .venv/bin/activate

# 1. Prove the port reproduces the authors' implementation (needs a GPU).
pytest BRAINTOKENGT/tests/test_upstream_equivalence.py -q

# 2. Wiring smoke test — 12 subjects, 3 folds, 3 epochs (~1 min).
cd BRAINTOKENGT && python run_experiment.py --id braintokengt-smoke --no-wandb

# 3. Full cohort, primary competitor row.
python run_experiment.py --id braintokengt-delcode-whole-brain

# 4. Secondary row with the released-code defects repaired.
python run_experiment.py --id braintokengt-delcode-whole-brain-repaired

python run_experiment.py --status     # run table
python run_experiment.py --collect    # rebuild outputs/RESULTS.csv
```

---

## What can be claimed in the thesis

> **An edit is a *port* if it provably yields the identical computation at the
> authors' original settings. An edit is a *modification* if it changes the
> computation there.**

Everything applied unconditionally here is on the port side, and the equivalence
test is the evidence:

```
pytest BRAINTOKENGT/tests/test_upstream_equivalence.py -q     # 20 passed
```

It instantiates upstream `EvolveGCNH_Transformer` and this package's
`BrainTokenGT` at the authors' published configuration (M=90 AAL ROIs, T=3 visits,
binary edge weights), copies upstream's weights across parameter-for-parameter, and
asserts the forward passes agree — across three architectures and five input draws.

Defensible wording:

> We evaluated Brain-TokenGT (Dong et al., MICCAI 2023) using the authors' released
> implementation of the GIVE and BIGTR modules. The model definition was ported
> unchanged apart from (i) generalising the hardcoded ROI count and sequence length
> from (M=90, T=3) to arbitrary (M, T), and (ii) removing hardcoded CUDA device
> placement. A regression test confirms the ported model reproduces the original
> implementation's output at the authors' published configuration. The released
> training script trains on a single subject with no test split, so training and
> evaluation used our own harness, identical to that used for every other model in
> this comparison.

**Not** defensible: "we reproduced the authors' reported results." Their 87.14 AUC
is OASIS-3, n=60, AAL-90, and did not come from this code path (see below).

---

## What was changed

### Ports — unconditional, verified no-ops at (M=90, T=3)

| # | Upstream | Here |
|---|---|---|
| 1 | `max_num_nodes=270`, `time_steps=3`, `in_channels=90` (`model_transformer.py:11-14`) and `np.eye(270,270,±90)` (`:126-127`) | `(M, T)` are constructor/call-time values; T is read from `len(A_list)` per subject, so visit counts may vary without padding |
| 2 | `.cuda()` in ~15 places; `parameters()` overridden to return an `nn.ParameterList` assigned over `nn.Module`'s own `_parameters` registry | Normal module registration; follows `.to(device)` |
| 3 | `edge_attr` read row-major off `adjs_all` while `hyperedge_index` is `[static \| temporal]`, so feature *i* is not edge *i* (`:131-137`) | Features gathered at the concatenated edge index. A **no-op** upstream — every released edge weight is exactly 1.0, and permuting an all-ones vector changes nothing — observable only under `edge_weight_mode="weighted"` |

### Flagged behavioural differences — **defaults reproduce the released code**

Four upstream behaviours contradict the paper. Each is a config flag whose default
is the upstream behaviour, so the faithful run is the default run.

| Flag | Default (= upstream) | Repaired | What upstream does |
|---|---|---|---|
| `edge_weight_mode` | `"binary"` | `"weighted"` | `main_optuna.py:84` passes a binarised adjacency and never forwards `edge_attr`, so VEE's hypergraph conv is fed an all-ones vector and **never sees FC weights** — contradicting paper §2.2 |
| `train_give` | `false` | `true` | `Parameter(...).to(device)` returns a plain tensor, so `GRCU.parameters()` is **empty**: the whole INE/EvolveGCN module is frozen at random init and the optimiser never receives it (asserted in `test_upstream_give_is_untrainable`) |
| `readout` | `"mean"` | `"graph_token"` | `:178` averages **all** tokens; paper Fig. 1 reads out the `[graph]` token |
| `force_single_head` | `true` | `false` | `:76` sets `self.nhead = nhead` then immediately `self.nhead = 1`, discarding the configured head count |

**Report both rows** (`braintokengt-delcode-whole-brain` and
`…-repaired`) and take the better as the competitor's score. That is
simultaneously faithful and immune to "you handicapped the baseline".
`run_summary.json → model_config.upstream_faithful` records which a run was.

### Preserved upstream quirks — not flagged, kept unconditionally

* Node identifiers are appended as M extra **tokens** rather than concatenated to
  each token's features (paper Eq. 4), and there are M of them for M·T node tokens.
* The static/temporal split index is `num_edge − M(T−1)` while the temporal block
  spans `2·M(T−1)` columns, so half the temporal edges are pooled as "static".
* `self.projection` is built but never used in `forward` (kept for parameter parity).
* HyperDrop / the paper's interpretability (Eq. 3, Fig. 2) is **absent upstream** and
  is not added here.

### Not ported at all

`Brain-TokenGT/datasets.py` — a synthetic-data loader. Line 24 does
`FC[keys].astype('int64')`, harmless only because the shipped `.mat` matrices are
integer 0/1; on real z-transformed FC (every |r| < 1) it zeroes the entire matrix.
Its edge budget is hardcoded to 1216 entries, meaningful only at 90×90.

`Brain-TokenGT/main_optuna.py` — line 60 is `val_index = list(range(9)); train_index = [9]`:
it trains on **one subject** and validates on nine, with `StratifiedKFold` commented
out and no test split. The value returned to Optuna is a max-over-epochs on that
same validation set.

Both are replaced by `model/sequences.py` + `adapter.py` + the shared
`CLASSIFIER.common` harness.

---

## Fair-comparison contract

What is held identical between Brain-TokenGT and GELSTM:

1. **Data** — the adapter's `prepare_data` builds its Bundle from the *same*
   `LongitudinalSubjectDataset` object the GELSTM adapter uses, with the same
   `data_root`, `cohorts_csv`, `file_variant` and `allowed_months` filtering. Both
   models see byte-identical FC matrices for an identical subject set.
2. **Splits** — `SPLITS/downstream` train+val as the CV pool, test held out;
   `StratifiedGroupKFold(5)` via the shared `run_kfold_cv`.
3. **Threshold policy** — `select_oof_threshold` on pooled out-of-fold predictions
   (Best-F1 default), applied unchanged to test. Never re-optimised on test.
4. **Metrics** — AUC / sensitivity / specificity / F1 from the shared
   `binary_metrics`.
5. **Artifacts** — same `save_run` schema, same `RESULTS.csv` ledger.

What differs, deliberately, because it is part of each *method*:

* **Adjacency** — Brain-TokenGT uses a global top-k over the FC matrix (upstream's
  1216/90² density, transferred as a density); GELSTM uses kNN (k=8).
* **Encoder** — Brain-TokenGT trains end-to-end; GELSTM uses a pretrained frozen
  GAAE. State this as pipeline-vs-pipeline in the write-up.
* **Δt** — GELSTM consumes normalised inter-visit intervals; Brain-TokenGT has no
  such notion (its temporal edges are weighted 1).

### ⚠ The cohort window you must match

The authors' cohort was *"2–3 time points, zero-padded to three"*. DELCODE spans
**1–6 visits**, and **47 of 167 subjects have a single scan** — a T=1 sequence has no
temporal edges, a path upstream never executes. The configs therefore default to
`min_visits=2, max_visits=3`, restricting to the authors' regime rather than forcing
their model into ours.

**This means the GELSTM entry must be re-run with the same window before the two
numbers go in a table together.** Otherwise the models are scored on different
cohorts. The excluded single-scan subjects are themselves a reportable finding —
GELSTM handles them, Brain-TokenGT structurally cannot.

### Equal tuning budget

Upstream runs 200 Optuna trials; these configs are fixed. Either leave both fixed
(current state) or give both the same search budget **on the CV pool only**. Do not
let one search and not the other.

---

## Capacity note

At M=200, T=3 with the default config the token sequence is ~1.4k tokens per
subject (1 graph + 600 node + 180 pooled spatial-edge + 400 temporal-edge + 200
identifier tokens). Parameter counts are printed by `build_model()` and recorded in
`run_summary.json`; put them in the results table — the CV pool is ~120 subjects.
