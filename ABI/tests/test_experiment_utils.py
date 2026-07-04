"""Unit tests for the ABI experiment-registry helpers."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ABI.common.experiment_utils import (
    DEFAULT_CONFIG,
    build_config,
    build_parameter_dict,
    collect_results,
    load_experiment,
    load_registry,
    read_statuses,
)


def _write_registry(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "experiments.yaml"
    path.write_text(textwrap.dedent(body))
    return path


_BASELINE = """\
experiments:
  - id: abi-baseline-delcode
    notebook: notebooks/ABI_BASELINE.ipynb
    seed: 42
"""


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #
def test_load_experiment_found(tmp_path):
    reg = _write_registry(tmp_path, _BASELINE)
    exp = load_experiment(reg, "abi-baseline-delcode")
    assert exp["seed"] == 42


def test_load_experiment_unknown_id_lists_known(tmp_path):
    reg = _write_registry(tmp_path, _BASELINE)
    with pytest.raises(ValueError, match="abi-baseline-delcode"):
        load_experiment(reg, "does-not-exist")


def test_missing_required_field_raises(tmp_path):
    reg = _write_registry(
        tmp_path,
        """\
        experiments:
          - id: no-seed
            notebook: notebooks/ABI_BASELINE.ipynb
    """,
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_registry(reg)


def test_duplicate_ids_raise(tmp_path):
    reg = _write_registry(
        tmp_path,
        _BASELINE
        + """\
  - id: abi-baseline-delcode
    notebook: notebooks/ABI_LONGITUDINAL_DELCODE_WHOLE_BRAIN.ipynb
    seed: 1
""",
    )
    with pytest.raises(ValueError, match="Duplicate experiment id"):
        load_registry(reg)


def test_unwired_notebook_raises(tmp_path):
    reg = _write_registry(
        tmp_path,
        """\
        experiments:
          - id: bad-notebook
            notebook: notebooks/SOME_OTHER_NOTEBOOK.ipynb
            seed: 1
    """,
    )
    with pytest.raises(ValueError, match="is not a wired ABI notebook"):
        load_registry(reg)


def test_invalid_config_override_raises(tmp_path):
    reg = _write_registry(
        tmp_path,
        """\
        experiments:
          - id: bad-config
            notebook: notebooks/ABI_BASELINE.ipynb
            seed: 1
            config: not-a-mapping
    """,
    )
    with pytest.raises(ValueError, match="'config' must be a mapping"):
        load_registry(reg)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def test_build_config_layering(tmp_path):
    config_path = tmp_path / "override.json"
    config_path.write_text(json.dumps({"Z_THRESHOLD": 2.5, "N_FOLDS": 10}))

    exp = {
        "id": "abi-baseline-delcode",
        "notebook": "notebooks/ABI_BASELINE.ipynb",
        "seed": 7,
        "config_path": "override.json",
        "config": {"N_FOLDS": 3},
    }
    config = build_config(exp, tmp_path)

    # JSON config_path overrides DEFAULT_CONFIG
    assert config["Z_THRESHOLD"] == 2.5
    # inline 'config:' override wins over config_path
    assert config["N_FOLDS"] == 3
    # untouched defaults survive
    assert config["FILE_SUFFIX"] == DEFAULT_CONFIG["baseline"]["FILE_SUFFIX"]
    # seed is the single source of truth for RANDOM_STATE
    assert config["RANDOM_STATE"] == 7


def test_build_config_missing_config_path_raises(tmp_path):
    exp = {
        "id": "abi-baseline-delcode",
        "notebook": "notebooks/ABI_BASELINE.ipynb",
        "seed": 1,
        "config_path": "does_not_exist.json",
    }
    with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
        build_config(exp, tmp_path)


def test_build_parameter_dict_keys(tmp_path):
    exp = {
        "id": "abi-baseline-delcode",
        "notebook": "notebooks/ABI_BASELINE.ipynb",
        "seed": 42,
    }
    params = build_parameter_dict(exp, tmp_path)
    assert set(params) == {
        "EXPERIMENT_ID",
        "SEED",
        "CONFIG",
        "WANDB_ENABLED",
        "OUTPUT_DIR",
        "RUN_DIR",
        "RUN_NAME",
    }
    assert params["EXPERIMENT_ID"] == "abi-baseline-delcode"
    # ABI defaults W&B OFF, unlike PROGNOSER/CLASSIFIER
    assert params["WANDB_ENABLED"] is False
    assert params["RUN_DIR"] is None


def test_wandb_true_propagates(tmp_path):
    exp = {
        "id": "x",
        "notebook": "notebooks/ABI_BASELINE.ipynb",
        "seed": 1,
        "wandb": True,
    }
    assert build_parameter_dict(exp, tmp_path)["WANDB_ENABLED"] is True


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def test_collect_results_flattens_flat_metrics(tmp_path):
    run_dir = tmp_path / "abi-baseline-delcode" / "runs" / "2026-06-24_10-00-00"
    run_dir.mkdir(parents=True)
    summary = {
        "experiment_id": "abi-baseline-delcode",
        "timestamp": "2026-06-24_10-00-00",
        "kind": "baseline",
        "git": {"short_commit": "abc1234", "dirty": False},
        "metrics": {
            "cv_auc_mean": 0.86,
            "best_threshold": 0.55,
            "test_auc": 0.37,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary))

    rows = collect_results(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == "abi-baseline-delcode"
    assert row["kind"] == "baseline"
    assert row["git_commit"] == "abc1234"
    assert row["metric.cv_auc_mean"] == 0.86
    assert row["metric.test_auc"] == 0.37
    # ledger files are written
    assert (tmp_path / "RESULTS.csv").is_file()
    assert (tmp_path / "RESULTS.jsonl").is_file()


def test_read_statuses_sorted_recent_first(tmp_path):
    for ts, started in [("a", "2026-06-24_09-00-00"), ("b", "2026-06-24_11-00-00")]:
        d = tmp_path / "exp" / "runs" / ts
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"state": "done", "started_at": started}))
    statuses = read_statuses(tmp_path)
    assert [s["started_at"] for s in statuses] == ["2026-06-24_11-00-00", "2026-06-24_09-00-00"]
