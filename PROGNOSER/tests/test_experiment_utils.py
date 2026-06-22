"""Unit tests for the PROGNOSER experiment-registry helpers."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from PROGNOSER.common.experiment_utils import (
    COMBO_TABLE,
    DEFAULT_EXPERIMENT,
    build_experiment,
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


_KM = """\
experiments:
  - id: km-baseline
    method: km
    network_combo: dmn_hippo
    seed: 42
    notebook: notebooks/PROGNOSER_RUNNER.ipynb
"""


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #
def test_load_experiment_found(tmp_path):
    reg = _write_registry(tmp_path, _KM)
    exp = load_experiment(reg, "km-baseline")
    assert exp["method"] == "km"


def test_load_experiment_unknown_id_lists_known(tmp_path):
    reg = _write_registry(tmp_path, _KM)
    with pytest.raises(ValueError, match="km-baseline"):
        load_experiment(reg, "does-not-exist")


def test_missing_required_field_raises(tmp_path):
    reg = _write_registry(tmp_path, """\
        experiments:
          - id: no-seed
            method: km
            network_combo: dmn_hippo
            notebook: notebooks/PROGNOSER_RUNNER.ipynb
    """)
    with pytest.raises(ValueError, match="missing required field"):
        load_registry(reg)


def test_duplicate_ids_raise(tmp_path):
    reg = _write_registry(tmp_path, _KM + """\
  - id: km-baseline
    method: cox_clinical
    network_combo: dmn
    seed: 1
    notebook: notebooks/PROGNOSER_RUNNER.ipynb
""")
    with pytest.raises(ValueError, match="Duplicate experiment id"):
        load_registry(reg)


def test_invalid_method_raises(tmp_path):
    reg = _write_registry(tmp_path, """\
        experiments:
          - id: bad-method
            method: random_forest
            network_combo: dmn_hippo
            seed: 1
            notebook: notebooks/PROGNOSER_RUNNER.ipynb
    """)
    with pytest.raises(ValueError, match="method='random_forest' invalid"):
        load_registry(reg)


def test_unknown_combo_raises(tmp_path):
    reg = _write_registry(tmp_path, """\
        experiments:
          - id: bad-combo
            method: km
            network_combo: cerebellum
            seed: 1
            notebook: notebooks/PROGNOSER_RUNNER.ipynb
    """)
    with pytest.raises(ValueError, match="network_combo='cerebellum' unknown"):
        load_registry(reg)


def test_invalid_strategy_raises(tmp_path):
    reg = _write_registry(tmp_path, """\
        experiments:
          - id: bad-strategy
            method: cox_combined
            network_combo: dmn_hippo
            seed: 1
            notebook: notebooks/PROGNOSER_RUNNER.ipynb
            experiment:
              embedding_strategy: teleport
    """)
    with pytest.raises(ValueError, match="embedding_strategy='teleport' invalid"):
        load_registry(reg)


def test_embedding_method_requires_strategy(tmp_path):
    reg = _write_registry(tmp_path, """\
        experiments:
          - id: cox-emb-no-strategy
            method: cox_embedding
            network_combo: dmn_hippo
            seed: 1
            notebook: notebooks/PROGNOSER_RUNNER.ipynb
            experiment:
              embedding_strategy: null
    """)
    with pytest.raises(ValueError, match="requires a non-null embedding_strategy"):
        load_registry(reg)


def test_lstm_requires_sequence_strategy(tmp_path):
    reg = _write_registry(tmp_path, """\
        experiments:
          - id: lstm-wrong-strategy
            method: lstm_surv
            network_combo: dmn_hippo
            seed: 1
            notebook: notebooks/PROGNOSER_RUNNER.ipynb
            experiment:
              embedding_strategy: last
    """)
    with pytest.raises(ValueError, match="requires .*embedding_strategy='sequence'"):
        load_registry(reg)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def test_build_experiment_merge_and_combo_derivation():
    exp = {
        "id": "cox-combined", "method": "cox_combined", "network_combo": "dmn",
        "seed": 7, "notebook": "notebooks/PROGNOSER_RUNNER.ipynb",
        "experiment": {"penalizer": 0.2, "embedding_strategy": "mean"},
    }
    merged = build_experiment(exp)
    # combo-derived data_version/file_suffix come from COMBO_TABLE
    assert merged["data_version"] == COMBO_TABLE["dmn"][0]
    assert merged["file_suffix"] == COMBO_TABLE["dmn"][1]
    # method + seed propagate; override block wins
    assert merged["method"] == "cox_combined"
    assert merged["random_state"] == 7
    assert merged["penalizer"] == 0.2
    assert merged["embedding_strategy"] == "mean"
    # untouched defaults survive
    assert merged["eval_times"] == DEFAULT_EXPERIMENT["eval_times"]


def test_build_parameter_dict_keys():
    exp = {
        "id": "km-baseline", "method": "km", "network_combo": "dmn_hippo",
        "seed": 42, "notebook": "notebooks/PROGNOSER_RUNNER.ipynb",
    }
    params = build_parameter_dict(exp)
    assert set(params) == {
        "EXPERIMENT_ID", "EXPERIMENT", "SEED", "WANDB_ENABLED",
        "OUTPUT_DIR", "RUN_DIR", "RUN_NAME",
    }
    assert params["EXPERIMENT_ID"] == "km-baseline"
    assert params["WANDB_ENABLED"] is True
    assert params["RUN_DIR"] is None


def test_wandb_false_propagates():
    exp = {
        "id": "x", "method": "km", "network_combo": "dmn_hippo", "seed": 1,
        "notebook": "notebooks/PROGNOSER_RUNNER.ipynb", "wandb": False,
    }
    assert build_parameter_dict(exp)["WANDB_ENABLED"] is False


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def test_collect_results_flattens_nested_metrics(tmp_path):
    run_dir = tmp_path / "km-baseline" / "runs" / "2026-06-19_10-00-00"
    run_dir.mkdir(parents=True)
    summary = {
        "experiment_id": "km-baseline",
        "timestamp": "2026-06-19_10-00-00",
        "method": "km",
        "experiment": {"network_combo": "dmn_hippo", "feature_set": "clinical"},
        "git": {"short_commit": "abc1234", "dirty": False},
        "metrics": {
            "test": {"c_index": 0.71, "ibs": 0.18, "auc": {"24": 0.69, "36": 0.72, "60": 0.7}},
            "val": {"c_index": 0.68, "ibs": 0.2, "auc": {}},
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary))

    rows = collect_results(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == "km-baseline"
    assert row["network_combo"] == "dmn_hippo"
    assert row["git_commit"] == "abc1234"
    assert row["metric.test_c_index"] == 0.71
    assert row["metric.test_ibs"] == 0.18
    assert row["metric.test_auc_24"] == 0.69
    assert row["metric.val_c_index"] == 0.68
    # ledger files are written
    assert (tmp_path / "RESULTS.csv").is_file()
    assert (tmp_path / "RESULTS.jsonl").is_file()


def test_read_statuses_sorted_recent_first(tmp_path):
    for ts, started in [("a", "2026-06-19_09-00-00"), ("b", "2026-06-19_11-00-00")]:
        d = tmp_path / "exp" / "runs" / ts
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"state": "done", "started_at": started}))
    statuses = read_statuses(tmp_path)
    assert [s["started_at"] for s in statuses] == ["2026-06-19_11-00-00", "2026-06-19_09-00-00"]
