# GEGRU cross-dataset generalization audit + simulated scanner-drift sweep

Context: the trained `gegru-trajectory-whole-brain` model (full-trajectory GRU
over frozen GAAE embeddings, see
[`CLASSIFIER/experiments/longitudinal.yaml`](../CLASSIFIER/experiments/longitudinal.yaml))
is evaluated on DELCODE only — ADNI and OASIS3 have no preprocessed
Schaefer-200 FC graphs yet (raw DICOM/NIfTI only). Two notebooks document what
*can* be measured honestly today and set up what activates automatically once
ADNI/OASIS3 preprocessing lands:

| Notebook | Question it answers | Training? |
|---|---|---|
| [`COMPARISON_GEGRU_CROSS_DATASET.ipynb`](../CLASSIFIER/notebooks/COMPARISON/COMPARISON_GEGRU_CROSS_DATASET.ipynb) | Real DELCODE performance + how different are ADNI/OASIS3's scanners, quantitatively? | No — reloads a saved run |
| [`SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT.ipynb`](../CLASSIFIER/notebooks/SANITY/SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT.ipynb) | How much does the model degrade if DELCODE graphs are perturbed to *simulate* ADNI-/OASIS3-like acquisition drift? | No — reloads the same run |

**Neither notebook claims the model has been validated on ADNI or OASIS3.**
That claim requires real FC matrices for those cohorts, which don't exist yet.
What exists instead: a real DELCODE number, a quantitative scanner-heterogeneity
inventory, and a drift simulation anchored to that inventory — not to an
assumption like "more vendors = more noise."

---

## 1. Real DELCODE evaluation

The saved run (`outputs/gegru-trajectory-whole-brain/latest`,
`lucky-harbor-3-d9655b0db-2026-06-21_13-50-00`) is reloaded with
`adapter.load_state()` / `read_run_threshold()` — no retraining, no
re-derivation of the decision threshold (Best-F1 on OOF, per
[`evaluation.md`](../.claude/rules/evaluation.md)) — and re-scored on the
34-subject DELCODE test split:

| Metric | Value |
|---|---|
| AUC | **0.8321** |
| Sensitivity | 0.6429 |
| Specificity | 0.7500 |
| F1 | 0.6429 |
| Threshold | 0.6594 (`oof_f1`) |

This matches `run_summary.json`'s `test_auc` exactly, which is the notebook's
built-in consistency check: if the reload ever drifts (wrong GAAE checkpoint,
wrong `adjacency_k`, wrong feature-norm buffers), this number would move and
flag it immediately.

> **Caught during validation:** the first draft of the reload cell hardcoded
> `adjacency_k=8` for the GAAE kNN-graph construction. The run was actually
> trained with `adjacency_k=16` (`configs/gaae_delcode_whole_brain.json`).
> That single wrong default silently produced AUC 0.72 instead of 0.83 —
> same code path, same checkpoint, wrong graph topology. Fixed by reading
> `gaae_hp` from the GAAE config file instead of reconstructing it from
> `run_summary.json`'s `model_config` (which records the encoder's output
> shape, not how its input graph was built).

---

## 2. Scanner / site heterogeneity — what ADNI and OASIS3 actually look like

DELCODE's `cohorts.csv` carries no scanner/site columns (single-protocol
cohort, untracked). ADNI and OASIS3 do:

| | ADNI | OASIS3 |
|---|---|---|
| fMRI/MR records | 18,616 | 30,339 (5,124 are resting-state BOLD) |
| Manufacturer | SIEMENS 14,056 / Philips 2,416 / GE 2,144 | Siemens-only |
| Model generations | TrioTim, Prisma(_fit), Skyra, Verio, DISCOVERY MR750, Achieva, Ingenia, ... (10+) | TrioTim, Biograph_mMR, MAGNETOM_Vida, Vision, Sonata, Avanto (6) |
| Field strength | 3.0 T (18,614) / 1.5 T (2) | 3.0 T / 1.5 T / 1.494 T |

ADNI is genuinely multi-vendor; OASIS3 is single-vendor but spans three
materially different Siemens scanner generations (PET/MR `Biograph_mMR`,
`TrioTim`, `MAGNETOM_Vida`) — different coil hardware and SNR profiles despite
sharing a manufacturer.

> **Caught during validation:** `OASIS3_MR_json.csv` is a dump of *every* MR
> series acquired (T1, DTI, ASL, fieldmaps, ...), not just resting-state BOLD.
> Computing the TR coefficient of variation over the whole file mixes "TR
> varies because the sequence type varies" with genuine fMRI protocol drift,
> inflating the CV to 8.6 — an order of magnitude too high. Restricting to
> `SeriesDescription` matching `bold|rsfmri` (5,124 of 30,339 records) brought
> it down to 0.358, in line with ADNI's 0.381.

---

## 3. Calibration anchor — turning "scanners differ" into a number

Two real quantities, computed in the comparison notebook and persisted to
[`outputs/comparison-gegru-cross-dataset/standalone/site_heterogeneity_stats.json`](../CLASSIFIER/outputs/comparison-gegru-cross-dataset/standalone/site_heterogeneity_stats.json):

| Quantity | Value | Meaning |
|---|---|---|
| `delcode_consecutive_visit_fc_delta_std` | 0.2636 | std of edge-wise FC differences between consecutive same-subject DELCODE visits — the "drift" the model already tolerates from normal longitudinal variability |
| `typical_fc_std` | 0.2655 | typical per-graph FC value spread |
| `base_noise_level` | 0.9925 | `delcode floor / typical_fc_std` — DELCODE's natural drift expressed in the same units as `perturb_graph`'s `noise_level` parameter (it scales injected noise by each graph's own std) |
| `adni_tr_cv` | 0.3815 | std/mean of `fmri_tr` across ADNI's scanner records |
| `oasis3_tr_cv` | 0.3585 | std/mean of `RepetitionTime` across OASIS3's resting-state BOLD records |

The drift notebook sets each profile's noise level as
`noise_level = base_noise_level * (1 + CV_<site>)` — i.e. "DELCODE's own
natural drift, scaled up by how much more dispersed that cohort's real
acquisition parameters are." This is the bridge between "ADNI has 3 vendors"
(qualitative) and an actual `perturb_graph` magnitude (quantitative), instead
of picking a noise level by hand.

---

## 4. Simulated drift sweep — results

[`SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT.ipynb`](../CLASSIFIER/notebooks/SANITY/SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT.ipynb)
perturbs the real DELCODE test trajectories with
[`common.robustness.perturb_graph`](../CLASSIFIER/common/robustness.py)
(`feature_noise` + `edge_perturbation`) at the calibrated levels and a
surrounding sweep, always scoring at the *same* validation-derived threshold
(0.6594 — never re-tuned on perturbed data):

* **adni_like** — `feature_noise` + `edge_perturbation` both at the full
  calibrated level (multi-vendor ⇒ amplitude *and* structural drift).
* **oasis3_like** — `feature_noise` at the calibrated level,
  `edge_perturbation` at half that (single-vendor, but 3 distinct model
  generations, so some structural drift is plausible — just less than ADNI's).

| Profile | Noise level | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| — (baseline) | 0.000 | 0.832 | 0.643 | 0.750 | 0.643 |
| adni_like | 0.100 | 0.879 | 0.714 | 0.750 | 0.690 |
| adni_like | 0.300 | 0.879 | 0.714 | 0.800 | 0.714 |
| adni_like | 0.500 | 0.871 | 0.714 | 0.800 | 0.714 |
| **adni_like** | **1.371 (calibrated)** | **0.700** | **0.071** | **1.000** | **0.133** |
| oasis3_like | 0.100 | 0.854 | 0.643 | 0.750 | 0.643 |
| oasis3_like | 0.300 | 0.879 | 0.714 | 0.750 | 0.690 |
| oasis3_like | 0.500 | 0.868 | 0.714 | 0.800 | 0.714 |
| **oasis3_like** | **1.348 (calibrated)** | **0.721** | **0.071** | **1.000** | **0.133** |

Full sweep: [`drift_sweep.csv`](../CLASSIFIER/outputs/sanity-gegru-synthetic-scanner-drift/standalone/drift_sweep.csv).
Summary JSON: [`drift_summary.json`](../CLASSIFIER/outputs/sanity-gegru-synthetic-scanner-drift/standalone/drift_summary.json).

**Reading the curve:** at low-to-moderate noise (0.1–0.5) the model is
essentially flat or even nudges up slightly — within the noise of a 34-subject
test set. At the calibrated real-world operating point (~1.35–1.37, i.e.
roughly 1.35× DELCODE's own natural inter-visit drift), AUC drops from 0.832
to ~0.70–0.72 and sensitivity collapses to 0.071 — the model starts predicting
almost everyone as `stable_mci` (specificity saturates at 1.0). This is the
same qualitative failure mode the cover-letter draft describes for the real
DELCODE→ADNI generalization gap: a model that looks strong in-cohort loses
most of its converter-detection ability once acquisition-protocol drift
exceeds what it saw during training — here demonstrated on simulated rather
than real ADNI/OASIS3 data, because that data isn't preprocessed yet.

**Sanity check:** `noise_level=0.0` reproduces the real DELCODE AUC (0.8321)
exactly in both notebooks, confirming `perturb_graph` is a true no-op at that
setting and the two notebooks share a consistent reload.

---

## 5. Histopathology analogy (portfolio framing)

The same failure mode shows up in digital pathology: a model trained on
whole-slide images from one scanner/stain protocol can look excellent
in-house and degrade sharply on a second site's slides, driven by
scanner-vendor color response, stain-batch variation, and tissue-prep
differences — the histopathology analogue of the MRI
manufacturer/field-strength/TR heterogeneity quantified above. The
methodology transfers directly: stratify performance by site/scanner, trace
failures back to acquisition metadata, and make the degradation explicit
(the calibration anchor + drift sweep above) rather than reporting a single
flattering AUC.

---

## 6. Limitations

- **n=34 test subjects** — every metric above has wide confidence intervals;
  treat the drift curve as illustrative of *direction and rough magnitude*,
  not a precise degradation estimate.
- **Single run, single seed** — no variance estimate across GEGRU training
  seeds or CV folds; the saved run is whichever was promoted to `latest`.
- **Simulated, not real, drift** — `perturb_graph`'s Gaussian/edge-rewiring
  noise is a proxy for scanner heterogeneity, not a physical acquisition
  simulator. It does not model stain-equivalent effects like systematic bias,
  geometric distortion, or motion artifacts specific to a given scanner.
- **OASIS3's `RepetitionTime` CV** still mixes several rs-fMRI sequence
  variants (`ep2d_bold_connect`, `Axial_rsfMRI_MB4`, ...) that may differ in
  more than just TR; the filter removes structural/DTI/ASL contamination but
  doesn't fully disentangle fMRI sub-protocols.
- **The ADNI/OASIS3 arms are stubs**, not results — `comparison_summary.json`
  reports `"status": "pending_preprocessing"` for both. Once
  `DATA/src/processing` produces their Schaefer-200 matrices and a
  DELCODE-shaped `cohorts.csv`, re-running
  `COMPARISON_GEGRU_CROSS_DATASET.ipynb` scores them with no code changes.

---

## File locations

| File | Role |
|---|---|
| [`CLASSIFIER/notebooks/COMPARISON/COMPARISON_GEGRU_CROSS_DATASET.ipynb`](../CLASSIFIER/notebooks/COMPARISON/COMPARISON_GEGRU_CROSS_DATASET.ipynb) | Real DELCODE eval + scanner-heterogeneity characterization + calibration anchor + ADNI/OASIS3 stubs |
| [`CLASSIFIER/notebooks/SANITY/SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT.ipynb`](../CLASSIFIER/notebooks/SANITY/SANITY_GEGRU_SYNTHETIC_SCANNER_DRIFT.ipynb) | Calibrated drift sweep + degradation curves + ROC overlay |
| [`CLASSIFIER/common/robustness.py`](../CLASSIFIER/common/robustness.py) | `perturb_graph()` — shared perturbation primitive (see also [`noise_perturbation_methods.md`](noise_perturbation_methods.md)) |
| [`CLASSIFIER/outputs/comparison-gegru-cross-dataset/standalone/`](../CLASSIFIER/outputs/comparison-gegru-cross-dataset/standalone/) | `site_heterogeneity_stats.json`, `comparison_summary.json`, figures |
| [`CLASSIFIER/outputs/sanity-gegru-synthetic-scanner-drift/standalone/`](../CLASSIFIER/outputs/sanity-gegru-synthetic-scanner-drift/standalone/) | `drift_sweep.csv`, `drift_summary.json`, figures |
| [`CLASSIFIER/outputs/gegru-trajectory-whole-brain/latest/`](../CLASSIFIER/outputs/gegru-trajectory-whole-brain/latest/) | The reloaded GEGRU run (`run_summary.json`, checkpoint) |
