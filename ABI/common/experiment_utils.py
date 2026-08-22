"""
Experiment-registry helpers for the ABI notebook runner.

Mirror of ``PROGNOSER/common/experiment_utils.py``, adapted for the two ABI
notebooks (cross-sectional baseline ABI vs. longitudinal subject-level ABI).
There is no dataclass-style hyperparameter bundle here — each notebook's
"Configuration" cell is a flat dict of scalars/paths, so the merge order is
simply ``DEFAULT_CONFIG[kind] < JSON config_path < inline 'config:' override``,
mirroring CLASSIFIER's dataclass-defaults < config_path < hyperparams
layering but without the dataclass step (ABI has no training hyperparameters).

Layering: stays cheap to import (yaml/json only) so the CLI starts fast.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from SHARED.runner_io import infer_run_duration, reconcile_run_status

_REQUIRED_FIELDS = ("id", "notebook", "seed")

_NOTEBOOK_KIND = {
    "ABI_BASELINE.ipynb": "baseline",
    "ABI_LONGITUDINAL_DELCODE_WHOLE_BRAIN.ipynb": "longitudinal",
}

# Mirrors each notebook's current "## Configuration" cell. Paths are
# repo-relative strings (resolved against REPO_ROOT by the notebook).
DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "WB_ROOT": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/matrices",
        "METADATA_DIR": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata",
        "SPLITS_DIR": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata/splits_gec",
        "COHORTS_CSV": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata/cohorts.csv",
        "FILE_VARIANT": "z_transformed",
        "FILE_SUFFIX": "_whole_brain_correlation_matrix_z_transformed.npz",
        "Z_THRESHOLD": 2.0,
        "N_FOLDS": 5,
        "RANDOM_STATE": 42,
        "STD_FLOOR": 1e-6,
    },
    "longitudinal": {
        "WB_ROOT": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/matrices",
        "METADATA_DIR": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata",
        "SPLITS_GAAE_DIR": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata/splits_gaae",
        "SPLITS_GEC_DIR": "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata/splits_gec",
        "FILE_SUFFIX": "_whole_brain_correlation_matrix_z_transformed.npz",
        "Z_THRESHOLD": 3.0,
        "N_FOLDS": 5,
        "RANDOM_STATE": 42,
        "STD_FLOOR": 1e-6,
    },
}


def _kind_from_notebook(notebook: str, exp_id: str) -> str:
    """Derive the experiment kind from the notebook filename.

    There's exactly one notebook per kind, so a free-standing 'kind:' field
    in the registry could disagree with 'notebook:' for no benefit — the
    notebook path is the single source of truth.
    """
    stem = Path(notebook).name
    kind = _NOTEBOOK_KIND.get(stem)
    if kind is None:
        raise ValueError(
            f"Experiment {exp_id!r}: notebook={notebook!r} is not a wired ABI "
            f"notebook; expected one of {sorted(_NOTEBOOK_KIND)}."
        )
    return kind


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
        raise ValueError(
            f"Each experiment in {yaml_path} must be a mapping, got {type(exp).__name__}."
        )
    missing = [f for f in _REQUIRED_FIELDS if exp.get(f) is None]
    if missing:
        raise ValueError(
            f"Experiment {exp.get('id', '<no-id>')!r} in {yaml_path} is missing "
            f"required field(s): {missing}."
        )
    _kind_from_notebook(exp["notebook"], exp["id"])  # raises if not a wired notebook
    override = exp.get("config") or {}
    if not isinstance(override, dict):
        raise ValueError(f"Experiment {exp['id']!r}: 'config' must be a mapping if present.")


# --------------------------------------------------------------------------- #
# Build the merged config + papermill parameter dict
# --------------------------------------------------------------------------- #
def build_config(exp: Dict[str, Any], abi_root: str | Path) -> Dict[str, Any]:
    """Merge: ``DEFAULT_CONFIG[kind] < JSON config_path < inline 'config:' override``."""
    abi_root = Path(abi_root)
    kind = _kind_from_notebook(exp["notebook"], exp["id"])
    config: Dict[str, Any] = dict(DEFAULT_CONFIG[kind])

    config_path = exp.get("config_path")
    if config_path:
        json_path = abi_root / config_path
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Experiment {exp['id']!r}: config_path {json_path} does not exist."
            )
        config.update(json.loads(json_path.read_text()))

    config.update(exp.get("config") or {})
    config["RANDOM_STATE"] = exp["seed"]  # seed is the single source of truth
    return config


def build_parameter_dict(exp: Dict[str, Any], abi_root: str | Path) -> Dict[str, Any]:
    """Flat papermill parameters for ``exp`` (RUN_DIR/RUN_NAME added by runner)."""
    return {
        "EXPERIMENT_ID": exp["id"],
        "SEED": exp["seed"],
        "CONFIG": build_config(exp, abi_root),
        "WANDB_ENABLED": exp.get("wandb", False),  # ABI defaults W&B OFF — see README
        "OUTPUT_DIR": exp.get("output_dir") or f"outputs/{exp['id']}",
        # Filled in per-execution by the runner:
        "RUN_DIR": None,
        "RUN_NAME": None,
    }


# --------------------------------------------------------------------------- #
# Results ledger + status aggregation
# --------------------------------------------------------------------------- #
def _iter_run_summaries(outputs_root: Path):
    yield from outputs_root.glob("*/runs/*/run_summary.json")


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ABI's flat ``metrics`` block into ``metric.*`` columns.

    ABI summaries are CV-only (no train/val/test-nested survival metrics):
    ``metrics`` is a single flat dict, e.g. ``{'cv_auc_mean': .., 'best_threshold':
    .., 'test_auc': .., 'test_sensitivity': .., 'test_specificity': .., 'test_f1': ..}``.
    """
    flat: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"metric.{k}"] = v
    return flat


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
            "kind": summary.get("kind"),
        }
        duration = summary.get("duration_seconds")
        if duration is None:
            dur = infer_run_duration(run_dir)
            if dur is not None:
                duration = round(float(dur), 1)
        if duration is not None:
            row["duration_seconds"] = duration

        git = summary.get("git") or {}
        row["git_commit"] = git.get("short_commit")
        row["git_dirty"] = git.get("dirty")
        if summary.get("state") or summary.get("status"):
            row["state"] = summary.get("state") or summary.get("status")
            row["status"] = summary.get("status") or summary.get("state")
        if summary.get("error"):
            row["error"] = summary.get("error")
        row.update(_flatten_metrics(summary.get("metrics") or {}))
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


def find_run_dir(outputs_root: str | Path, target: str) -> Path | None:
    """Find a run directory given a run name, run name prefix, path, or experiment id.

    Parameters
    ----------
    outputs_root : str | Path
        Root outputs directory (e.g. ABI/outputs).
    target : str
        Run name (e.g. 'crimson-galaxy-4-5e33e2170-2026-08-22_12-54-20'),
        run name prefix, experiment id, or relative/absolute path to a run dir.
    """
    outputs_root = Path(outputs_root).resolve()
    target = target.strip()
    if not target:
        return None

    # 1. Direct path check
    direct = Path(target)
    if direct.is_dir():
        return direct.resolve()
    if (outputs_root / target).is_dir() and not (outputs_root / target / "runs").is_dir():
        return (outputs_root / target).resolve()

    # 2. Exact match for run name anywhere under outputs/*/runs/<target>
    matching_runs = list(outputs_root.glob(f"*/runs/{target}"))
    if matching_runs:
        return matching_runs[0].resolve()

    # 3. Partial / prefix match for run name
    matching_runs_prefix = list(outputs_root.glob(f"*/runs/{target}*"))
    if matching_runs_prefix:
        def _sort_key(d: Path) -> str:
            status_file = d / "status.json"
            if status_file.is_file():
                try:
                    data = json.loads(status_file.read_text())
                    return str(data.get("started_at") or "")
                except Exception:
                    pass
            return str(d.stat().st_mtime)

        matching_runs_prefix.sort(key=_sort_key, reverse=True)
        return matching_runs_prefix[0].resolve()

    # 4. Target is an experiment ID (check outputs/<target>/latest or outputs/<target>/runs/*)
    exp_dir = outputs_root / target
    if exp_dir.is_dir():
        latest_link = exp_dir / "latest"
        if latest_link.is_symlink() or latest_link.is_dir():
            try:
                resolved = latest_link.resolve()
                if resolved.is_dir():
                    return resolved
            except Exception:
                pass

        latest_txt = exp_dir / "latest.txt"
        if latest_txt.is_file():
            try:
                run_name = latest_txt.read_text().strip()
                t = exp_dir / "runs" / run_name
                if t.is_dir():
                    return t.resolve()
            except Exception:
                pass

        runs_dir = exp_dir / "runs"
        if runs_dir.is_dir():
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if run_dirs:
                def _sort_key(d: Path) -> str:
                    status_file = d / "status.json"
                    if status_file.is_file():
                        try:
                            data = json.loads(status_file.read_text())
                            return str(data.get("started_at") or "")
                        except Exception:
                            pass
                    return str(d.stat().st_mtime)

                run_dirs.sort(key=_sort_key, reverse=True)
                return run_dirs[0].resolve()

    return None


find_latest_run = find_run_dir


def read_statuses(
    outputs_root: str | Path,
    *,
    experiment_id: str | None = None,
    limit: int | None = None,
    reconcile: bool = True,
) -> List[Dict[str, Any]]:
    """Gather run ``status.json`` files (most recent first) for ``--status``."""
    outputs_root = Path(outputs_root)
    statuses: List[Dict[str, Any]] = []
    pattern = f"{experiment_id}/runs/*/status.json" if experiment_id else "*/runs/*/status.json"
    for status_path in outputs_root.glob(pattern):
        if reconcile:
            try:
                status = reconcile_run_status(status_path, write_disk=True)
            except Exception:
                try:
                    status = json.loads(status_path.read_text())
                except Exception:
                    continue
        else:
            try:
                status = json.loads(status_path.read_text())
            except Exception:
                continue
        status["_path"] = str(status_path)
        statuses.append(status)
    statuses.sort(key=lambda s: s.get("started_at") or "", reverse=True)
    if limit is not None and limit >= 0:
        return statuses[:limit]
    return statuses
