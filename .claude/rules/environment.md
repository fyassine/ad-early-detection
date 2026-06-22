# Environment

Use latest APIs as of May 2026.

## Project-root `.venv`

- Python 3.10.12
- torch 2.10.0+cu128
- torch_geometric 2.7.0
- numpy 1.26.4
- pandas 2.2.0
- scikit-learn 1.7.2

This is the venv used by `CLASSIFIER/`, `PROGNOSER/`, and DASHBOARD's model-inference path.

## Subpackage requirements

- `PROGNOSER/requirements.txt` — lifelines, scikit-survival, joblib. Install on top of the root venv.
- `DASHBOARD/requirements.txt` — fastapi, uvicorn, pandas, scipy, networkx, umap-learn, openpyxl. Install on top of the root venv; do not duplicate torch/PyG entries here.
- `CLASSIFIER/requirements-explain.txt` — captum, nilearn (for the EXPLAIN notebook / `adapters/explain.py`). Install on top of the root venv; do not pin torch/PyG here.

## Do not

- Do not suggest deprecated APIs (e.g. `torch.jit.script` was deprecated in 2.10 — use `torch.compile`).
- Do not pin different torch/PyG versions in subpackage requirements.
- Do not assume CUDA is available in test code — use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
