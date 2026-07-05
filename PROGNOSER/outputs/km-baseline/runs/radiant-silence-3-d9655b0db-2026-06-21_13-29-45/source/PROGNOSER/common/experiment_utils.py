"""
Experiment-registry helpers for the PROGNOSER notebook runner.

These functions turn an entry in ``PROGNOSER/experiments.yaml`` into the flat
parameter dict that ``run_experiment.py`` injects into ``PROGNOSER_RUNNER.ipynb``
via papermill, and aggregate finished runs into a single results ledger.

Mirror of ``CLASSIFIER/common/experiment_utils.py`` adapted for survival
analysis: there are no config dataclasses here — the ``EXPERIMENT`` dict *is*
the config — so the merge order is ``DEFAULT_EXPERIMENT < registry entry``.

Layering: this module stays cheap to import (yaml only, no torch / no model
code) so the CLI starts fast. ``COMBO_TABLE`` lives here as the single source of
truth for the combo -> (data_version, file_suffix) mapping; the embedding-build
CLI imports it from here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

# --------------------------------------------------------------------------- #
# Canonical network-combo table (single source of truth, no heavy imports).
# build_subject_embeddings.py imports COMBO_TABLE from here.
# --------------------------------------------------------------------------- #
COMBO_TABLE: Dict[str, tuple[str, str]] = {
    "dmn":              ("__fc_dmn_sch200_flat__",                       "_dmn_correlation_matrix_z_transformed.npz"),
    "hippo":            ("__fc_hippo_tian2_flat__",                      "_hippocampus_correlation_matrix_z_transformed.npz"),
    "limbic":           ("__fc_limbic_sch200_flat__",                    "_limbic_correlation_matrix_z_transformed.npz"),
    "dan":              ("__fc_dan_sch200_flat__",                       "_dorsal_attention_correlation_matrix_z_transformed.npz"),
    "dmn_hippo":        ("__fc_dmn-hippo_sch200-tian2_flat__",           "_dmn_hippo_correlation_matrix_z_transformed.npz"),
    "dmn_limbic":       ("__fc_dmn-limbic_sch200_flat__",                "_dmn_limbic_correlation_matrix_z_transformed.npz"),
    "dmn_limbic_hippo": ("__fc_dmn-hippo-limbic_sch200-tian2_flat__",    "_dmn_limbic_hippo_correlation_matrix_z_transformed.npz"),
    "all_combined":     ("__fc_dmn-hippo-limbic-dan_sch200-tian2_flat__", "_all_combined_correlation_matrix_z_transformed.npz"),
}

# --------------------------------------------------------------------------- #
# Canonical EXPERIMENT defaults (mirror PROGNOSER_RUNNER.ipynb cell 1).
# --------------------------------------------------------------------------- #
DEFAULT_EXPERIMENT: Dict[str, Any] = {
    "network_combo": "dmn_hippo",
    "data_version": "__fc_dmn-hippo_sch200-tian2_flat__",
    "file_suffix": "_dmn_hippo_correlation_matrix_z_transformed.npz",
    "method": "cox_clinical_longitudinal",
    "feature_set": "clinical_longitudinal",
    "embedding_strategy": "last",
    "longitudinal_features": ["mmstot", "cdrglobal"],
    "longitudinal_aggs": ["baseline", "last", "slope", "delta"],
    "eval_times": [12, 24, 36, 48, 60, 72],
    "penalizer": 0.05,
    "pca_components": 16,
    "rsf_n_estimators": 200,
    "rsf_min_samples_leaf": 5,
    "lstm_n_time_bins": 12,
    "lstm_max_horizon_months": 72,
    "lstm_hidden_dim": 64,
    "lstm_epochs": 100,
    "random_state": 42,
}

_REQUIRED_FIELDS = ("id", "method", "network_combo", "seed", "notebook")

_VALID_METHODS = {
    "km", "cox_clinical", "cox_embedding", "cox_combined",
    "cox_clinical_longitudinal", "cox_time_varying", "rsf", "deepsurv", "lstm_surv",
}
_VALID_STRATEGIES = {"baseline", "last", "mean", "slope", "all_aggs", "sequence"}
# Methods that consume GAAE embeddings and therefore require an embedding cache.
_EMBEDDING_METHODS = {"cox_embedding", "cox_combined", "rsf", "deepsurv", "lstm_surv"}


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
    method = exp["method"]
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Experiment {exp['id']!r}: method={method!r} invalid; "
            f"expected one of {sorted(_VALID_METHODS)}."
        )
    combo = exp["network_combo"]
    if combo not in COMBO_TABLE:
        raise ValueError(
            f"Experiment {exp['id']!r}: network_combo={combo!r} unknown; "
            f"expected one of {sorted(COMBO_TABLE)}."
        )
    override = exp.get("experiment") or {}
    if not isinstance(override, dict):
        raise ValueError(f"Experiment {exp['id']!r}: 'experiment' must be a mapping if present.")

    merged = build_experiment(exp)
    strategy = merged.get("embedding_strategy")
    if strategy is not None and strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Experiment {exp['id']!r}: embedding_strategy={strategy!r} invalid; "
            f"expected one of {sorted(_VALID_STRATEGIES)}."
        )
    # Embedding-based methods cannot run without a strategy that selects visits.
    if method in _EMBEDDING_METHODS:
        if method == "lstm_surv" and strategy != "sequence":
            raise ValueError(
                f"Experiment {exp['id']!r}: method='lstm_surv' requires "
                f"embedding_strategy='sequence' (got {strategy!r})."
            )
        if method != "lstm_surv" and not strategy:
            raise ValueError(
                f"Experiment {exp['id']!r}: method={method!r} consumes GAAE embeddings "
                f"and requires a non-null embedding_strategy."
            )


# --------------------------------------------------------------------------- #
# Build the merged EXPERIMENT dict + papermill parameter dict
# --------------------------------------------------------------------------- #
def build_experiment(exp: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an entry into a full ``EXPERIMENT`` dict: defaults < combo-derived < entry.

    ``network_combo`` drives ``data_version``/``file_suffix`` from
    :data:`COMBO_TABLE` (single source of truth), then the registry's optional
    ``experiment:`` override block wins for any explicitly set key.
    """
    merged: Dict[str, Any] = {**DEFAULT_EXPERIMENT}
    combo = exp["network_combo"]
    data_version, file_suffix = COMBO_TABLE[combo]
    merged.update(
        network_combo=combo,
        data_version=data_version,
        file_suffix=file_suffix,
        method=exp["method"],
        random_state=exp["seed"],
    )
    merged.update(exp.get("experiment") or {})
    return merged


def build_parameter_dict(exp: Dict[str, Any]) -> Dict[str, Any]:
    """Flat papermill parameters for ``exp`` (run_dir/run_name added by runner)."""
    return {
        "EXPERIMENT_ID": exp["id"],
        "EXPERIMENT": build_experiment(exp),
        "SEED": exp["seed"],
        "WANDB_ENABLED": exp.get("wandb", True),
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
    """Flatten the nested survival ``metrics`` block into ``metric.*`` columns.

    PROGNOSER summaries nest as ``metrics[split][c_index|ibs|auc]`` where ``auc``
    is itself a dict keyed by eval-time (string keys after JSON round-trip).
    """
    flat: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        block = metrics.get(split) or {}
        if not isinstance(block, dict):
            continue
        for key in ("c_index", "ibs"):
            if isinstance(block.get(key), (int, float)):
                flat[f"metric.{split}_{key}"] = block[key]
        auc = block.get("auc") or {}
        if isinstance(auc, dict):
            for t, v in auc.items():
                if isinstance(v, (int, float)):
                    flat[f"metric.{split}_auc_{t}"] = v
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
        exp = summary.get("experiment") or {}
        row: Dict[str, Any] = {
            "experiment_id": summary.get("experiment_id") or run_dir.parents[1].name,
            "run_dir": str(run_dir.relative_to(outputs_root.parent)),
            "timestamp": summary.get("timestamp"),
            "method": summary.get("method") or exp.get("method"),
            "network_combo": exp.get("network_combo"),
            "feature_set": summary.get("feature_set") or exp.get("feature_set"),
            "embedding_strategy": summary.get("embedding_strategy") or exp.get("embedding_strategy"),
        }
        git = summary.get("git") or {}
        row["git_commit"] = git.get("short_commit")
        row["git_dirty"] = git.get("dirty")
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
