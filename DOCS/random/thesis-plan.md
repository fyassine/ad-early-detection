# Thesis Implementation Plan — Apr 28 → Aug 4, 2026

> **Goal:** Pivot the existing GNN-only thesis into a *Bayern-medtech-ready* multimodal time-to-event Alzheimer's prognosis project, while submitting on time and applying for industry roles in parallel.
>
> **Submission target:** early August 2026 (~14 weeks).
>
> **Repo baseline:** GNN family (GAAE, DenseGAAE, GEC, CostWeightedGEC) under `MODEL/model/`, trained on DELCODE fMRI with whole-brain and DMN parcellations; W&B logging in place; binary `is_converter` label.

---

## North-star deliverables

By August 4, the repo + thesis should demonstrate:

1. **Time-to-event head** — discrete-time survival prediction of MCI → AD conversion using DELCODE longitudinal visits (M0–M60), reported with concordance index (C-index) and time-dependent AUC.
2. **Three encoder families benchmarked** — your existing GNN, a MONAI-based 3D-CNN, a 3D Vision Transformer (UNETR encoder or ViT3D). Same data splits, same survival head, fair comparison.
3. **Multimodal fusion** — imaging encoder + tabular branch (CSF biomarkers + cognitive scores + APOE + age + sex), fused via cross-attention or concatenation.
4. **Productionized artifact** — Dockerized FastAPI endpoint that takes a NIfTI input + tabular biomarkers and returns a hazard curve and a 24-month conversion risk; Streamlit demo on top.
5. **Public-grade repo** — README, results table, W&B link, tests, CI, a 2-page summary PDF.
6. **Thesis written** — clinical motivation, methods, results, discussion. ~60–80 pages depending on TUM template.

---

## High-level timeline

| Week | Dates | Thesis work | Job-search work |
|------|-------|-------------|-----------------|
| W0 | Apr 28 – May 4 | Phase 0 — Setup & branching | Lebenslauf v1; portfolio repo skeleton |
| W1 | May 5 – May 11 | Phase 1a — Time-to-event labels | Anschreiben template; LinkedIn refresh |
| W2 | May 12 – May 18 | Phase 1b — Survival head on existing GNN | First 5–8 applications (Roche Penzberg, Siemens HC, deepc, Brainlab) |
| W3 | May 19 – May 25 | Phase 2a — MONAI install + dataloader + 3D CNN | First-round screening calls |
| W4 | May 26 – Jun 1 | Phase 2b — 3D ViT/UNETR encoder | Broaden applications to 15–20 |
| W5 | Jun 2 – Jun 8 | Phase 3 — Multimodal fusion | Technical interview prep |
| W6 | Jun 9 – Jun 15 | Phase 4a — Docker + FastAPI | Continued interviews |
| W7 | Jun 16 – Jun 22 | Phase 4b — Streamlit demo + tests + CI | On-site / deep technical rounds |
| W8 | Jun 23 – Jun 29 | Phase 5a — Evaluation: stratified k-fold, bootstrap | Reference checks, offer negotiation |
| W9 | Jun 30 – Jul 6 | Phase 5b — Ablations + interpretability (Captum / Integrated Gradients) | Sign offer (target) |
| W10 | Jul 7 – Jul 13 | Phase 6a — Thesis: intro, methods | Notify employer of submission date |
| W11 | Jul 14 – Jul 20 | Phase 6b — Thesis: results, discussion | — |
| W12 | Jul 21 – Jul 27 | Phase 6c — Thesis: revisions, figures | — |
| W13 | Jul 28 – Aug 3 | Phase 7 — Final pass, supervisor sign-off | — |
| W14 | Aug 4 | Submit | — |

If any phase slips: cut Phase 5b (interpretability) and/or shrink Phase 4b (drop Streamlit, keep FastAPI). Phases 1–4a are non-negotiable.

---

## Phase 0 — Setup (W0: Apr 28 – May 4)

### 0.1 Repo hygiene
- Create branch `thesis/pivot-survival-multimodal` off `main`.
- Add `docs/`, `tests/`, `infra/` directories at repo root. (`docs/` already houses this file.)
- Pin Python to 3.11; freeze current venv (`pip freeze > requirements.lock.txt`) before adding new deps so you can roll back.

### 0.2 Dependency additions

Add to a new `requirements-thesis-pivot.txt`:

```
monai==1.5.1
nibabel>=5.2
pydicom>=2.4
pycox>=0.2.3
scikit-survival>=0.23
torchsurv>=0.1.4
fastapi>=0.110
uvicorn[standard]>=0.29
streamlit>=1.34
captum>=0.7
pytest>=8.1
pytest-cov>=5.0
pre-commit>=3.7
ruff>=0.4
black>=24.4
```

Notes:
- **MONAI 1.5+** — required for nnU-Net bundle support and current Model Zoo.
- **pycox + scikit-survival + torchsurv** — three survival libraries; pycox for DeepHit/Logistic Hazard, scikit-survival for the C-index utility and Kaplan-Meier baselines, torchsurv for differentiable Cox PH if you want a continuous-time variant.
- **captum** — for Integrated Gradients on the 3D-CNN to make a saliency-map figure for your thesis.

### 0.3 Configuration scaffolding
Mirror existing `MODEL/configs/gaae_*.json` for the new models:

```
MODEL/configs/
  gaae_delcode_dmn.json           (existing)
  gaae_dancer.json                (existing)
  cnn3d_delcode_whole_brain.json  (NEW — Phase 2a)
  vit3d_delcode_whole_brain.json  (NEW — Phase 2b)
  multimodal_delcode_whole_brain.json (NEW — Phase 3)
```

Keep one config per model × dataset to keep notebook-driven training reproducible.

### 0.4 Project tracking
- Create a single W&B project: `delcode-mci-conversion-survival`. Use `group=` field per encoder family so GNN/CNN/ViT runs are visually grouped.
- Add a `notes/EXPERIMENT_LOG.md` where every W&B run-id gets one line of human commentary. This is the artifact you'll show interviewers.

---

## Phase 1 — Time-to-event reframing (W1–W2: May 5 – May 18)

This is the highest-leverage week. Don't touch architectures yet.

### 1.1 Build the survival labels (W1, ~2 days)

DELCODE has visits at M0, M12, M24, M36, M48, M60. For each MCI subject at baseline (M0):

- `t_event` = month of first visit at which they meet AD criteria (using `prmdiag` column referenced in your README).
- `event` = 1 if conversion observed, 0 if censored (last visit was MCI or SCD).
- For non-converters who exited the study at e.g. M24, `t_event = 24`, `event = 0` (right-censored).

Implementation:
- New module `MODEL/src/labels/survival_labels.py` with `build_survival_labels(metadata_df, max_horizon_months=60) -> pd.DataFrame[subject_id, t_event, event]`.
- Unit-test on a synthetic mini cohort in `tests/test_survival_labels.py` — verify three censoring patterns (early dropout, full follow-up no event, full follow-up with event).

### 1.2 Wire labels into the existing dataset (W1, ~1 day)

In each of `MODEL/model/{GAAE,DenseGAAE,GEC,CostWeightedGEC}/dataset.py`:
- Replace / supplement `is_converter` with `(t_event, event)` in the data graph attributes (e.g. `data.t_event`, `data.event`).
- Keep `is_converter` available as a fallback so old notebooks still run.

### 1.3 Add a discrete-time survival head (W2, ~3 days)

Pattern: any encoder produces a fixed-length embedding `z`; a discrete-time survival head outputs hazards over K bins (K = 6 → bins 0–12, 12–24, 24–36, 36–48, 48–60 months).

New file `MODEL/src/heads/survival_head.py`:

```python
import torch
import torch.nn as nn

class DiscreteTimeSurvivalHead(nn.Module):
    """
    Logistic-Hazard head over K time bins (Gensheimer & Narasimhan, 2019).
    Output: hazard h_k = sigmoid(logit_k).
    Survival S_k = prod_{j<=k}(1 - h_j).
    """
    def __init__(self, embed_dim: int, num_bins: int = 6, hidden: int = 128):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_bins),
        )

    def forward(self, z):
        return self.net(z)  # logits over bins

    @staticmethod
    def survival_from_logits(logits):
        hazards = torch.sigmoid(logits)
        return torch.cumprod(1.0 - hazards, dim=1)
```

Loss (Logistic-Hazard / negative log-likelihood with right-censoring), in `MODEL/src/losses/survival_loss.py`:

```python
def logistic_hazard_loss(logits, t_idx, event):
    """
    logits: [B, K]
    t_idx:  [B]   bin index of last observed time
    event:  [B]   1 if event observed at t_idx, 0 if censored
    """
    B, K = logits.shape
    log_h = torch.nn.functional.logsigmoid(logits)
    log_1mh = torch.nn.functional.logsigmoid(-logits)
    bin_range = torch.arange(K, device=logits.device).unsqueeze(0)
    t = t_idx.unsqueeze(1)
    surv_part = (bin_range < t).float() * log_1mh
    event_part = (bin_range == t).float() * (event.unsqueeze(1) * log_h + (1 - event.unsqueeze(1)) * log_1mh)
    return -(surv_part + event_part).sum(dim=1).mean()
```

### 1.4 Retrofit one GNN model first

Pick `CostWeightedGEC` (it's the one you've iterated most recently per `git status`).
- New trainer `MODEL/model/CostWeightedGEC/train_survival.py` mirroring `train.py` but using `logistic_hazard_loss` and tracking C-index instead of AUC.
- C-index via `sksurv.metrics.concordance_index_censored`.
- Time-dependent AUC at 24 and 36 months via `sksurv.metrics.cumulative_dynamic_auc`.

W&B charts: hazard heatmap per fold, predicted survival curves stratified by predicted risk quartile.

**Definition of done for Phase 1:** GAAE/GEC retrained with survival head; W&B reports C-index and tdAUC; results in a markdown table inside `notes/EXPERIMENT_LOG.md`.

---

## Phase 2 — MONAI 3D-CNN and 3D ViT baselines (W3–W4: May 19 – Jun 1)

### 2.1 MONAI 3D dataloader (W3, ~2 days)

DELCODE has functional connectivity matrices and parcellated time-series; for a 3D CNN you need *volumetric* fMRI inputs. Two options:

**Option A (recommended): use mean-fMRI volumes** — collapse 4D resting-state series to a 3D map per subject (mean signal, ALFF, fALFF, ReHo, or seed-based connectivity maps). Put one map per channel; e.g. 4-channel 3D volume per subject.

**Option B: full 4D** — too heavy for a master's thesis on this timeline, skip.

New file `MODEL/src/data/monai_volumetric_dataset.py`:

```python
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    ScaleIntensityd, RandFlipd, RandRotate90d, ToTensord
)

train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
    RandRotate90d(keys=["image"], prob=0.3, max_k=3),
    ToTensord(keys=["image"]),
])
```

Reuse the same train/val/test split that the GNN dataset uses (load it from `DATA/DELCODE/__v5__/metadata/splits.csv` once that's defined).

### 2.2 3D-CNN encoder (W3, ~2 days)

Use MONAI's `DenseNet121` 3D variant or `EfficientNetBN` 3D as the encoder. Strip the classification head; expose a `features` method that returns a [B, 1024] embedding, then plug `DiscreteTimeSurvivalHead` on top.

New `MODEL/model/CNN3D/{models.py, train.py, dataset.py}`:

```python
from monai.networks.nets import DenseNet121

class CNN3DSurvivalEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dim: int = 1024, num_bins: int = 6):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=embed_dim)
        self.head = DiscreteTimeSurvivalHead(embed_dim, num_bins=num_bins)

    def forward(self, x):
        z = self.backbone.features(x)
        z = torch.flatten(F.adaptive_avg_pool3d(z, 1), 1)
        return self.head(z), z
```

**Compute reality check:** 3D DenseNet on whole-brain MNI 91×109×91 with 4 channels and batch size 4 fits on a 16GB GPU; with batch size 2 fits on 12GB. Use mixed precision (`torch.cuda.amp`) from day one.

### 2.3 3D ViT encoder (W4, ~3 days)

Use MONAI's `UNETR` encoder (just the ViT trunk, drop the U-Net decoder) **or** `monai.networks.nets.ViT` 3D directly.

```python
from monai.networks.nets import ViT

class ViT3DSurvivalEncoder(nn.Module):
    def __init__(self, in_channels=4, img_size=(96, 96, 96), patch_size=16,
                 hidden_size=768, num_bins=6):
        super().__init__()
        self.backbone = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=False,
        )
        self.head = DiscreteTimeSurvivalHead(hidden_size, num_bins=num_bins)

    def forward(self, x):
        z, _ = self.backbone(x)
        z = z.mean(dim=1)  # mean over patch tokens
        return self.head(z), z
```

Note image size constraints: `img_size` must be divisible by `patch_size`. Pad/crop volumes to (96,96,96) in the transform pipeline.

### 2.4 Training parity checklist

For a fair comparison across GNN / CNN / ViT:
- Same subject splits (load once from CSV, never re-shuffle per encoder).
- Same number of training epochs OR same convergence criterion (early stop on val C-index, patience=20).
- Same survival head, same loss, same optimizer family (AdamW), each with its own LR (GNNs typically 5e-4, CNN/ViT typically 1e-4 to 5e-5 for transformer).
- Same evaluation protocol (5-fold stratified CV by `prmdiag` strata).

**Definition of done for Phase 2:** three encoders × one survival head, all logged to W&B group `phase2_baselines`, results table updated.

---

## Phase 3 — Multimodal fusion (W5: Jun 2 – Jun 8)

### 3.1 Tabular branch

DELCODE has CSF biomarkers (Aβ42, t-tau, p-tau181), neuropsych battery (MMSE, CERAD subscales, ADAS-Cog if collected), demographics (age, sex, education years), genetics (APOE ε4 carrier status).

New file `MODEL/src/data/tabular_features.py`:
- Function `load_tabular_features(metadata_df) -> Tuple[np.ndarray, List[str]]` returning a `(N, F)` matrix + feature names.
- Z-score continuous features using **train-fold statistics only** (compute in trainer, not globally).
- One-hot or binary code categorical (sex, APOE).
- Median-impute missing biomarkers; add a missingness indicator column per CSF feature (clinical-ML convention).

### 3.2 Tabular encoder + fusion

```python
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class CrossAttentionFusion(nn.Module):
    """Imaging tokens cross-attended by tabular query."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, img_tokens, tab_token):
        q = tab_token.unsqueeze(1)     # [B, 1, D]
        out, _ = self.attn(q, img_tokens, img_tokens)
        return self.norm(out.squeeze(1) + tab_token)
```

For the GNN branch, use concatenation (simpler, GNN embeddings are graph-pooled vectors not token sequences). For CNN/ViT, use cross-attention from the tabular query into the imaging token sequence.

Run **three multimodal variants** (one per encoder family) so the thesis can answer: *"does adding biomarkers help equally across architectures?"* — that's a publishable-quality question.

### 3.3 Sanity check
Train a tabular-only baseline (no imaging) — biomarkers + demographics → survival head. This is the most important baseline in the entire thesis: if a small MLP on CSF biomarkers reaches similar C-index to your imaging models, your thesis is honest about it. (Spoiler: it often does on AD prognosis. That's a *finding*, not a failure.)

**Definition of done for Phase 3:** four model rows in your results table (tabular-only, GNN+tab, CNN+tab, ViT+tab) plus the three imaging-only rows from Phase 2 = 7 total.

---

## Phase 4 — MLOps wrapper (W6–W7: Jun 9 – Jun 22)

### 4.1 Inference module (W6)

New file `infra/inference.py`:

```python
from typing import Dict
import torch, nibabel as nib, numpy as np
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, ToTensor, Resize

class SurvivalInferenceEngine:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        self.model = torch.load(ckpt_path, map_location=device).eval()
        self.tx = Compose([
            EnsureChannelFirst(),
            Resize(spatial_size=(96,96,96)),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            ToTensor(),
        ])

    @torch.inference_mode()
    def predict(self, nifti_path: str, tabular: Dict[str, float]):
        vol = nib.load(nifti_path).get_fdata().astype(np.float32)
        x = self.tx(vol).unsqueeze(0).to(self.device)
        tab = self._tabular_to_tensor(tabular).to(self.device)
        logits, _ = self.model(x, tab)
        S = self.model.head.survival_from_logits(logits)[0].cpu().numpy()
        return {
            "hazard_per_bin": torch.sigmoid(logits)[0].cpu().tolist(),
            "survival_per_bin": S.tolist(),
            "risk_24mo": float(1.0 - S[1]),
            "risk_36mo": float(1.0 - S[2]),
        }
```

### 4.2 FastAPI service (W6)

New file `infra/api.py`:

```python
from fastapi import FastAPI, UploadFile, File, Form
from infra.inference import SurvivalInferenceEngine
import json, tempfile, os

app = FastAPI(title="MCI-AD Conversion Risk API")
engine = SurvivalInferenceEngine(ckpt_path=os.environ["CKPT"])

@app.post("/predict")
async def predict(image: UploadFile = File(...), tabular: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(await image.read()); path = tmp.name
    return engine.predict(path, json.loads(tabular))

@app.get("/healthz")
def healthz(): return {"status": "ok"}
```

### 4.3 Docker (W6)

New file `infra/Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements-thesis-pivot.txt .
RUN pip install --no-cache-dir -r requirements-thesis-pivot.txt

COPY MODEL ./MODEL
COPY infra ./infra
EXPOSE 8000
CMD ["uvicorn", "infra.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

`infra/docker-compose.yml`:

```yaml
services:
  api:
    build: { context: .., dockerfile: infra/Dockerfile }
    environment:
      CKPT: /app/checkpoints/best_multimodal.pt
    volumes:
      - ../MODEL/notebooks/checkpoints_cost_weighted_gec_whole_brain:/app/checkpoints:ro
    ports: ["8000:8000"]
```

Test locally: `docker compose up --build`, then `curl localhost:8000/healthz`, then a sample inference with `curl -F image=@sample.nii.gz -F 'tabular={"age":72,...}' localhost:8000/predict`.

### 4.4 Streamlit demo (W7)

New file `infra/app.py`: 30 lines, sidebar for tabular fields, file uploader for NIfTI, line chart of survival curve, KPI tiles for 24-mo and 36-mo risk. This is what you link in your CV / cover letter.

### 4.5 Tests + CI (W7)

`tests/`:
- `test_survival_head.py` — shape checks, monotonicity of survival, gradient flows.
- `test_survival_loss.py` — known-value test on synthetic batch.
- `test_inference_engine.py` — load a tiny mock model, run predict on a small synthetic NIfTI.
- `test_tabular_features.py` — missingness indicator behavior.

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements-thesis-pivot.txt
      - run: ruff check .
      - run: pytest -q --cov=MODEL --cov=infra
```

`.pre-commit-config.yaml`: ruff, black, end-of-file-fixer, trailing-whitespace.

**Definition of done for Phase 4:** a) `docker compose up` produces a working API; b) Streamlit demo loads a NIfTI and shows a survival curve; c) GitHub Actions green on push.

---

## Phase 5 — Evaluation, ablations, interpretability (W8–W9: Jun 23 – Jul 6)

### 5.1 Robust metrics (W8)
- **5-fold stratified CV** (already in your repo per `_STRATIFIED_K_FOLD` artifact) — re-use the same fold assignments across all encoders.
- **Bootstrap 95% CI** for C-index and time-dependent AUC: 1000 resamples on the test fold.
- **Calibration:** plot predicted vs. observed survival via the IPCW-corrected Brier score (`sksurv.metrics.brier_score`).
- **Decision-curve analysis:** net benefit at 24-month threshold for clinical relevance — this is a *clinician-language* figure that pharma interviewers love.

### 5.2 Ablations (W8)
- Imaging-only vs. tabular-only vs. fused.
- Whole-brain vs. DMN parcellation (you already have both).
- Image-only encoder choice ablation: GNN, CNN, ViT.
- Loss ablation: Logistic-Hazard vs. DeepHit (pycox) — use the better one in the headline result.

### 5.3 Interpretability (W9)
- **For CNN/ViT:** Captum's `IntegratedGradients` on the imaging input, averaged within parcels — produce a brain-region importance map. One headline figure.
- **For GNN:** node-level attribution via `GNNExplainer` from `torch_geometric.explain` — show which connectome edges drive the prediction.
- **For tabular branch:** SHAP values over biomarkers. Match against neurology literature (p-tau and Aβ should dominate).

### 5.4 Comparison to known literature
Update `notes/EXPERIMENT_LOG.md` with a table comparing your numbers to the recent literature (3D-CNN on ADNI fMRI ~92% AUC; multimodal sMRI+rs-fMRI for MCI converters ~97%). Be honest about gaps; DELCODE is smaller than ADNI so absolute numbers will be lower — emphasize the C-index / time-dependent AUC angle that ADNI papers often skip.

**Definition of done for Phase 5:** 7-row results table with bootstrap CIs, calibration plot, decision-curve plot, three interpretability figures. These are your thesis Results chapter.

---

## Phase 6 — Thesis writing (W10–W12: Jul 7 – Jul 27)

### Chapter map (target ~70 pages)
1. **Introduction** (~8 pages) — clinical motivation: lecanemab/donanemab era, why prognosis matters now, fMRI's role; thesis question.
2. **Background** (~12 pages) — fMRI preprocessing, parcellation, GNN basics, CNN/ViT for medical imaging, survival analysis fundamentals (KM, Cox, discrete-time).
3. **Related work** (~6 pages) — MCI conversion prediction lineage (Frontiers 2025, Nature 2024 multimodal review), survival-with-imaging (PMC review), foundation models in medical imaging (VISTA3D, MONAI Model Zoo).
4. **Data** (~6 pages) — DELCODE cohort description, M0–M60 visit structure, biomarker assays, splits.
5. **Methods** (~12 pages) — encoder families (GNN, CNN, ViT), survival head, multimodal fusion, training protocol.
6. **Experiments & Results** (~16 pages) — 7-row table, calibration, ablations, interpretability.
7. **Discussion** (~6 pages) — what the encoder comparison reveals, why biomarkers may dominate, clinical translation considerations including EU AI Act high-risk classification.
8. **Conclusion + future work** (~3 pages).
9. **Appendix** — extra tables, hyperparameters, repo + W&B links.

### Writing strategy
- W10: drafts of 1, 2, 3, 4 (the parts that don't depend on final results).
- W11: drafts of 5, 6, 7.
- W12: revisions, figures polished, supervisor sign-off cycle.

Use a **figure-first** workflow: lock figures in W10, write captions, then write text around them. Cuts writing time roughly in half.

---

## Phase 7 — Submission (W13–W14: Jul 28 – Aug 4)

- Final supervisor pass.
- Plagiarism / format check per TUM template.
- Print + bind if your examiner requires hardcopy.
- Submit.
- Push the tagged commit `v1.0-thesis-submission` and freeze the branch.

---

## Parallel job-search activities (running W0 → W9)

### Application packet (W0–W1)
- **Lebenslauf** (German tabular CV): photo top-right, contact block, *Berufserfahrung*, *Ausbildung*, *Veröffentlichungen* (if any), *Skills*, *Sprachen*. 2 pages max.
- **Anschreiben** template per segment: 1-page, addressed to a named manager when possible, opens with the *clinical* problem they work on (not your skills), closes with the one-sentence pitch.
- **GitHub README** (the thesis repo): 1-paragraph clinical motivation → architecture diagram → results table → "Run the demo" section linking the Streamlit URL.

### Target shortlist (apply by end W2)
1. Roche Diagnostics, Penzberg — Data Scientist Neurology / Biomarker DS / ML Engineer Diagnostics
2. Siemens Healthineers, Erlangen — AI Solution Engineer / Research Scientist Medical Imaging
3. deepc, Munich — ML Engineer / Research Engineer
4. Brainlab, Munich — Software Engineer ML / Computer Vision
5. Boehringer Ingelheim — CNS / neurodegeneration AI roles
6. mediaire (Berlin, hybrid) — brain-MRI AI ML Engineer
7. Avelios Medical, Munich — clinical software ML
8. Helmholtz Munich — *engineer* (not scientist) titles only
9. Bayer / Merck KGaA — pharma data-science teams, often remote-DE
10. Quibim / Floy / Smart Reporting — radiology AI startups (remote-DE)

### Interview prep topics (curate over W3–W6)
- **Medical imaging foundation models:** VISTA3D, MONAI Model Zoo, BiomedCLIP, MedSAM — name + purpose + one limitation each.
- **EU AI Act:** high-risk classification for medical AI; conformity assessment; relationship to MDR/IVDR.
- **Lecanemab / donanemab era:** patient-stratification as commercial driver in 2026.
- **Plasma biomarkers (p-tau217 in particular):** clinical deployability since 2024–2026.
- **nnU-Net:** still the segmentation baseline; know architecture-as-design-system idea.
- **Survival analysis fundamentals:** KM, Cox, discrete-time hazard, C-index, IPCW.

### Single-sentence pitch (memorize)
> "I'm finishing a TUM master's thesis on multimodal time-to-event prediction of MCI-to-Alzheimer's conversion, combining fMRI and clinical biomarkers, with a benchmark of 3D-CNN, ViT, and GNN encoders — built end-to-end with MONAI, containerized and served via FastAPI, and tracked in Weights & Biases."

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 3D CNN/ViT memory blow-up on whole-brain volumes | Medium | Mixed precision from day one; downsample to (96,96,96); batch size 2 with gradient accumulation |
| DELCODE longitudinal coverage too sparse for survival | Medium | Fall back to coarser bins (3 bins: 0–24, 24–48, 48–60); include censoring indicator analysis |
| ViT3D doesn't converge in time | Low-Med | Pre-initialize from MONAI Model Zoo VISTA3D encoder if compatible; otherwise drop ViT and keep CNN+GNN |
| Tabular-only baseline beats imaging | Medium | This is a *finding*, not a failure — write it up honestly; pharma interviewers respect this |
| Job offer arrives before thesis submission | Possible | Negotiate Sep–Nov start date; many DE employers accept "thesis defense pending" |
| Thesis writing slips past Aug 4 | Med-Low | Cut Phase 5b (interpretability), Phase 4b (Streamlit) first; Phase 4a (Docker+API) is the must-keep |

---

## Reading list (15 papers, prioritized)

Read top 5 by W2; rest by W6.

1. Gensheimer & Narasimhan, *A scalable discrete-time survival model for neural networks* (PeerJ 2019).
2. Lee et al., *DeepHit: A deep learning approach to survival analysis with competing risks* (AAAI 2018).
3. Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* (Nature Methods 2021).
4. Hatamizadeh et al., *UNETR: Transformers for 3D Medical Image Segmentation* (WACV 2022).
5. He et al., *VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging* (arXiv 2024).
6. Frontiers Neurology 2025 — *Research progress in predicting MCI to AD conversion via multimodal MRI and AI*.
7. Frontiers Aging Neuroscience 2022 — *Deep Learning Model for Prediction of Progressive MCI to AD Using Structural MRI*.
8. PMC review — *Survival analysis using deep learning with medical imaging* (2024).
9. npj Digital Medicine 2025 — *Multimodal deep learning for cancer prognosis with clinical information prompts*.
10. NVIDIA Tech Blog — *Visual Foundation Models for Medical Image Analysis*.
11. EU AI Act overview — high-risk medical AI obligations (Article 6 & Annex III).
12. Siemens Healthineers AI-Rad Companion product whitepaper.
13. Roche NeuroToolKit overview (NeurologyLive).
14. MONAI documentation — Bundle / Model Zoo workflow.
15. Eugene Yan, *How to Interview ML/AI Engineers* (hiring-manager perspective).

---

## Acceptance criteria for this entire plan

You consider it successful if, by Aug 4, 2026:

- ✅ Thesis is submitted on time.
- ✅ Repo has all 7 model variants benchmarked with bootstrap CIs.
- ✅ Live demo URL exists (Streamlit) and a Dockerfile builds clean.
- ✅ At least one signed offer from the target employer list, or active final-round interviews.
- ✅ Single-sentence pitch fluent and reflexive in interviews.

If 4 of 5 are true on Aug 4, this plan worked.
