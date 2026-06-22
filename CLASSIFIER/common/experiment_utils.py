"""
Experiment-registry helpers for the notebook runner.

These functions turn an entry in ``CLASSIFIER/experiments.yaml`` into the flat
parameter dict that ``run_experiment.py`` injects into a notebook via papermill,
and aggregate finished runs into a single results ledger.

Layering: this module knows about config dataclasses and the registry, but does
NOT import torch or any model code — it stays cheap to import inside the CLI.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..configs import EvalConfig, GECTrainConfig, GELSTMTrainConfig

_REQUIRED_FIELDS = ("id", "mode", "model", "dataset", "seed", "notebook")
_VALID_THRESHOLD_MODES = {None, "youden", "best-f1", "fixed"}


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #
def load_registry(yaml_path: str | Path) -> List[Dict[str, Any]]:
    """Load all experiment entries, raising on a malformed registry."""
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Experiment registry not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text()) or {}
    experiments = data.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{yaml_path} has no 'experiments:' list.")
    ids = [e.get("id") for e in experiments]
    dupes = {i for i in ids if i is not None and ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate experiment id(s) in {yaml_path}: {sorted(dupes)}")
    for exp in experiments:
        _validate_experiment(exp, yaml_path)
    return experiments


def load_experiment(yaml_path: str | Path, exp_id: str) -> Dict[str, Any]:
    """Return the validated registry entry whose ``id`` equals ``exp_id``."""
    experiments = load_registry(yaml_path)
    for exp in experiments:
        if exp.get("id") == exp_id:
            return exp
    known = ", ".join(sorted(e.get("id", "?") for e in experiments))
    raise ValueError(f"No experiment with id={exp_id!r} in {yaml_path}. Known ids: {known}")


def _validate_experiment(exp: Dict[str, Any], yaml_path: Path) -> None:
    """Fail loudly on a missing/invalid field (see .claude/rules/errors.md)."""
    if not isinstance(exp, dict):
        raise ValueError(f"Each experiment in {yaml_path} must be a mapping, got {type(exp).__name__}.")
    missing = [f for f in _REQUIRED_FIELDS if exp.get(f) is None]
    if missing:
        raise ValueError(
            f"Experiment {exp.get('id', '<no-id>')!r} in {yaml_path} is missing "
            f"required field(s): {missing}."
        )
    mode = exp.get("threshold_mode")
    if mode not in _VALID_THRESHOLD_MODES:
        raise ValueError(
            f"Experiment {exp['id']!r}: threshold_mode={mode!r} invalid; "
            f"expected one of {sorted(m for m in _VALID_THRESHOLD_MODES if m)} or omitted."
        )
    if mode == "fixed" and exp.get("fixed_threshold") is None:
        raise ValueError(
            f"Experiment {exp['id']!r}: threshold_mode='fixed' requires 'fixed_threshold'."
        )


# --------------------------------------------------------------------------- #
# Build the merged hyperparameter config + papermill parameter dict
# --------------------------------------------------------------------------- #
def _dataclass_defaults(model: str) -> Dict[str, Any]:
    """Typed defaults for the model's config dataclass(es)."""
    model = (model or "").upper()
    if model == "GELSTM":
        return {**dataclasses.asdict(GELSTMTrainConfig()), **dataclasses.asdict(EvalConfig())}
    if model == "GEC":
        return dataclasses.asdict(GECTrainConfig())
    return {}


def build_config(exp: Dict[str, Any], classifier_root: str | Path) -> Dict[str, Any]:
    """Merge hyperparameters: dataclass defaults < JSON config < YAML overrides.

    ``hyperparams`` and ``eval_config`` blocks from the registry are layered on
    top. The result is the resolved config a notebook should train with; the
    runner writes it to ``resolved_config.json`` and injects it as
    ``RESOLVED_CONFIG``.
    """
    classifier_root = Path(classifier_root)
    config: Dict[str, Any] = _dataclass_defaults(exp.get("model"))

    config_path = exp.get("config_path")
    if config_path:
        json_path = classifier_root / config_path
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Experiment {exp['id']!r}: config_path {json_path} does not exist."
            )
        config.update(json.loads(json_path.read_text()))

    config.update(exp.get("hyperparams") or {})
    config.update(exp.get("eval_config") or {})
    return config


def build_parameter_dict(exp: Dict[str, Any], classifier_root: str | Path) -> Dict[str, Any]:
    """Flat papermill parameters for ``exp`` (run_dir/run_name added by runner)."""
    params: Dict[str, Any] = {
        "EXPERIMENT_ID": exp["id"],
        "MODE": exp["mode"],
        "MODEL": exp["model"],
        # Adapter registry key for the shared LONGITUDINAL_COMMON notebook; defaults
        # to MODEL when 'adapter:' is omitted (see CLASSIFIER/adapters/__init__.py).
        "ADAPTER": exp.get("adapter") or exp["model"],
        "DATASET": exp["dataset"],
        "SEED": exp["seed"],
        "GAAE_CHECKPOINT_PATH": exp.get("checkpoint_path"),
        # Source run to reload for analysis-only notebooks (e.g. the visit-count
        # confound sanity notebook): the notebook reads outputs/<id>/latest/.
        "SOURCE_EXPERIMENT": exp.get("source_experiment"),
        "THRESHOLD_MODE": exp.get("threshold_mode"),
        "FIXED_THRESHOLD": exp.get("fixed_threshold"),
        "WANDB_ENABLED": exp.get("wandb", True),
        "OUTPUT_DIR": exp.get("output_dir") or f"outputs/{exp['id']}",
        "RESOLVED_CONFIG": build_config(exp, classifier_root),
        # Filled in per-execution by the runner:
        "RUN_DIR": None,
        "RUN_NAME": None,
    }
    return params


# --------------------------------------------------------------------------- #
# Results ledger + status aggregation
# --------------------------------------------------------------------------- #
def _iter_run_summaries(outputs_root: Path):
    yield from outputs_root.glob("*/runs/*/run_summary.json")


def _cv_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise k-fold CV into ``cv.*`` columns (mean ± std across folds).

    Reads the per-fold lists the training notebooks write under ``cv_results``
    (``val_auc``, ``val_f1``, ``val_sensitivity``, ``val_specificity``, …) plus
    the top-level ``best_fold`` / ``best_val_auc``. Returns ``{}`` for runs
    without cross-validation (e.g. sanity/comparison notebooks).
    """
    import statistics

    out: Dict[str, Any] = {}
    cv = summary.get("cv_results")
    n_folds = 0
    if isinstance(cv, dict):
        for key, vals in cv.items():
            if not key.startswith("val_") or not isinstance(vals, list):
                continue
            nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not nums:
                continue
            n_folds = max(n_folds, len(nums))
            out[f"cv.{key}_mean"] = statistics.fmean(nums)
            out[f"cv.{key}_std"] = statistics.stdev(nums) if len(nums) > 1 else 0.0
    if n_folds:
        out["cv.n_folds"] = n_folds
    for src, dst in (("best_fold", "cv.best_fold"), ("best_val_auc", "cv.best_val_auc")):
        v = summary.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[dst] = v
    return out


def collect_results(outputs_root: str | Path) -> List[Dict[str, Any]]:
    """Flatten every ``run_summary.json`` into rows and write RESULTS.{csv,jsonl}."""
    outputs_root = Path(outputs_root)
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(_iter_run_summaries(outputs_root)):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        run_dir = summary_path.parent
        row: Dict[str, Any] = {
            "experiment_id": summary.get("experiment_id") or run_dir.parents[1].name,
            "run_dir": str(run_dir.relative_to(outputs_root.parent)),
            "timestamp": summary.get("timestamp"),
        }
        git = summary.get("git") or {}
        row["git_commit"] = git.get("short_commit")
        row["git_dirty"] = git.get("dirty")
        # Preferred: a uniform `metrics` block. Fallback: scalar top-level
        # `test_*` keys, which the existing notebooks already write — this keeps
        # the ledger populated even for notebooks not yet fully wired.
        metrics = dict(summary.get("metrics") or {})
        if not metrics:
            metrics = {
                k: v for k, v in summary.items()
                if k.startswith("test_") and isinstance(v, (int, float, bool))
            }
        for k, v in metrics.items():
            row[f"metric.{k}"] = v
        # Cross-validation summary (mean ± std across folds), when present.
        row.update(_cv_summary(summary))
        rows.append(row)

    if rows:
        fieldnames: List[str] = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(outputs_root / "RESULTS.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(outputs_root / "RESULTS.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return rows


def read_statuses(outputs_root: str | Path) -> List[Dict[str, Any]]:
    """Gather every run's ``status.json`` (most recent first) for ``--status``."""
    outputs_root = Path(outputs_root)
    statuses: List[Dict[str, Any]] = []
    for status_path in outputs_root.glob("*/runs/*/status.json"):
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            continue
        status["_path"] = str(status_path)
        statuses.append(status)
    statuses.sort(key=lambda s: s.get("started_at") or "", reverse=True)
    return statuses
