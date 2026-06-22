import os
from pathlib import Path

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
STATIC_DIR = Path(__file__).parent / "static"

# Repo root — used to resolve sibling modules (CLASSIFIER/...) at runtime.
REPO_ROOT = Path(__file__).resolve().parents[2]

# CLASSIFIER root (env override for non-standard deployments). The
# GELSTM service imports model code from here and reads checkpoints from
# GELSTM_CHECKPOINT_DIR.
CLASSIFIER_ROOT = Path(os.environ.get(
    "CLASSIFIER_ROOT",
    str(REPO_ROOT / "CLASSIFIER"),
))
GELSTM_CHECKPOINT_DIR = Path(os.environ.get(
    "GELSTM_CHECKPOINT_DIR",
    str(CLASSIFIER_ROOT / "model" / "GELSTM" / "checkpoints"),
))

# Disk cache directory for GELSTM predictions and other expensive aggregates.
DASHBOARD_CACHE_ROOT = Path(os.environ.get(
    "DASHBOARD_CACHE_ROOT",
    str(Path(__file__).resolve().parents[1] / ".cache"),
))
DASHBOARD_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
