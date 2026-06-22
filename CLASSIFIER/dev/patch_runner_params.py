"""
Idempotent patcher: inject a papermill ``parameters`` cell into every notebook
referenced by experiments.yaml.

The injected cell declares the variables ``run_experiment.py`` overrides, each
with a safe *interactive* default so opening the notebook in Jupyter behaves
exactly as before (``None`` triggers the original ``input()`` / JSON-config
paths). Papermill replaces these values at run time.

Idempotent: a notebook that already has a ``parameters``-tagged cell is skipped.

Run with:  python CLASSIFIER/dev/patch_runner_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _CLASSIFIER_ROOT / "experiments.yaml"

PARAM_TAG = "parameters"

PARAM_CELL_SOURCE = [
    "# === Papermill parameters (injected by run_experiment.py) ===\n",
    "# Safe interactive defaults: None keeps the original Jupyter behaviour\n",
    "# (interactive checkpoint/threshold prompts, JSON-config loading).\n",
    "EXPERIMENT_ID = None\n",
    "MODE = None\n",
    "MODEL = None\n",
    "DATASET = None\n",
    "SEED = 42\n",
    "GAAE_CHECKPOINT_PATH = None   # None -> interactive checkpoint picker\n",
    "THRESHOLD_MODE = None         # None -> interactive prompt; else 'youden'|'best-f1'|'fixed'\n",
    "FIXED_THRESHOLD = None        # required when THRESHOLD_MODE == 'fixed'\n",
    "WANDB_ENABLED = True          # W&B logging is on by default\n",
    "OUTPUT_DIR = None             # defaults to outputs/<experiment-id>/ when run standalone\n",
    "RESOLVED_CONFIG = None        # merged hyperparameter dict; overrides on-disk JSON when set\n",
    "RUN_DIR = None                # set by the runner: where run_summary.json / artifacts go\n",
    "RUN_NAME = None               # set by the runner: the W&B run name\n",
]


def _has_parameters_cell(nb) -> bool:
    return any(PARAM_TAG in (c.get("metadata", {}).get("tags") or []) for c in nb["cells"])


def _make_parameters_cell():
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [PARAM_TAG]},
        "outputs": [],
        "source": PARAM_CELL_SOURCE,
    }


def _notebook_paths() -> list[Path]:
    data = yaml.safe_load(_REGISTRY.read_text())
    paths = []
    for exp in data["experiments"]:
        nb = _CLASSIFIER_ROOT / exp["notebook"]
        if nb not in paths:
            paths.append(nb)
    return paths


def patch_one(path: Path) -> str:
    if not path.is_file():
        return "missing"
    nb = json.loads(path.read_text())
    if _has_parameters_cell(nb):
        return "already-tagged"
    # Insert right after a leading markdown title, else at the very top.
    insert_idx = 1 if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown" else 0
    nb["cells"].insert(insert_idx, _make_parameters_cell())
    path.write_text(json.dumps(nb, indent=1))
    return "patched"


def main() -> int:
    any_missing = False
    for path in _notebook_paths():
        status = patch_one(path)
        print(f"  {status:<14} {path.relative_to(_CLASSIFIER_ROOT)}")
        any_missing |= status == "missing"
    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
