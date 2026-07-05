"""
Split integrity and leakage tests for DATA/src/splitting/.

Section A — Unit tests: pure logic, synthetic DataFrames, no filesystem access.
Section B — Integration tests: validate on-disk CSVs; auto-skip when files absent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Repo root → dotted imports resolve
sys.path.insert(0, str(Path(__file__).parents[4]))

from DATA.src.splitting.create_downstream_data_splits import _patient_groups as _downstream_groups
from DATA.src.splitting.create_pretrain_data_splits import _patient_groups as _pretrain_groups
from DATA.src.splitting.load_splits import (
    get_split_indices_for_dataset,
    get_split_patient_ids,
    split_csv_paths,
    splits_dir,
)
from CLASSIFIER.common.sanity import run_full_audit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohort(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal cohorts DataFrame from [(Pseudonym, diagnosis), ...]."""
    return pd.DataFrame(rows, columns=["Pseudonym", "diagnosis"])


def _fake_split_data(train=("A", "B"), val=("C",), test=("D",)) -> dict:
    return {
        "train":      {p: {} for p in train},
        "validation": {p: {} for p in val},
        "test":       {p: {} for p in test},
    }


# ---------------------------------------------------------------------------
# Section A — Unit tests
# ---------------------------------------------------------------------------

class TestDownstreamPatientGroups:
    def test_converter_wins_over_mci(self):
        df = _cohort([("P1", "mci"), ("P1", "converter")])
        result = _downstream_groups(df)
        assert result["P1"] == "converter"

    def test_mci_only(self):
        df = _cohort([("P1", "mci"), ("P1", "mci")])
        result = _downstream_groups(df)
        assert result["P1"] == "mci"

    def test_scd_only_excluded(self):
        df = _cohort([("P1", "scd")])
        result = _downstream_groups(df)
        assert "P1" not in result

    def test_ad_only_excluded(self):
        df = _cohort([("P1", "ad")])
        result = _downstream_groups(df)
        assert "P1" not in result

    def test_healthy_only_excluded(self):
        df = _cohort([("P1", "healthy")])
        result = _downstream_groups(df)
        assert "P1" not in result

    def test_mixed_scd_mci_gives_mci(self):
        df = _cohort([("P1", "scd"), ("P1", "mci")])
        result = _downstream_groups(df)
        assert result["P1"] == "mci"

    def test_multiple_patients(self):
        df = _cohort([
            ("A", "mci"), ("A", "converter"),
            ("B", "mci"),
            ("C", "healthy"),
        ])
        result = _downstream_groups(df)
        assert result["A"] == "converter"
        assert result["B"] == "mci"
        assert "C" not in result


class TestPretrainPatientGroups:
    def test_converter_wins(self):
        df = _cohort([("P1", "mci"), ("P1", "converter"), ("P1", "ad")])
        result = _pretrain_groups(df)
        assert result["P1"] == "converter"

    def test_mci_wins_over_healthy(self):
        df = _cohort([("P1", "healthy"), ("P1", "mci")])
        result = _pretrain_groups(df)
        assert result["P1"] == "mci"

    def test_healthy_only(self):
        df = _cohort([("P1", "healthy")])
        result = _pretrain_groups(df)
        assert result["P1"] == "healthy"

    def test_ad_only(self):
        df = _cohort([("P1", "ad")])
        result = _pretrain_groups(df)
        assert result["P1"] == "ad"

    def test_scd_only_excluded(self):
        df = _cohort([("P1", "scd")])
        result = _pretrain_groups(df)
        assert "P1" not in result

    def test_scd_with_healthy_gives_healthy(self):
        df = _cohort([("P1", "scd"), ("P1", "healthy")])
        result = _pretrain_groups(df)
        assert result["P1"] == "healthy"

    def test_priority_order_full(self):
        # converter always wins
        for secondary in ("mci", "healthy", "ad"):
            df = _cohort([("P1", secondary), ("P1", "converter")])
            assert _pretrain_groups(df)["P1"] == "converter", secondary
        # mci beats healthy and ad
        for secondary in ("healthy", "ad"):
            df = _cohort([("P1", secondary), ("P1", "mci")])
            assert _pretrain_groups(df)["P1"] == "mci", secondary


class TestLoadSplitsPathApi:
    def test_splits_dir_returns_path(self):
        for model in ("pretrain", "downstream"):
            result = splits_dir(model)
            assert isinstance(result, Path)

    def test_splits_dir_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            splits_dir("nonexistent_model")

    def test_split_csv_paths_keys(self):
        for model in ("pretrain", "downstream"):
            paths = split_csv_paths(model)
            assert set(paths.keys()) == {"train", "val", "test"}

    def test_split_csv_paths_filenames(self):
        for model in ("pretrain", "downstream"):
            paths = split_csv_paths(model)
            assert paths["train"].endswith("train.csv")
            assert paths["val"].endswith("val.csv")
            assert paths["test"].endswith("test.csv")

    def test_get_split_patient_ids_returns_set(self):
        data = _fake_split_data(train=("A", "B"), val=("C",), test=("D",))
        assert get_split_patient_ids(data, "train") == {"A", "B"}
        assert get_split_patient_ids(data, "validation") == {"C"}
        assert get_split_patient_ids(data, "test") == {"D"}

    def test_get_split_patient_ids_unknown_split_raises(self):
        data = _fake_split_data()
        with pytest.raises(ValueError):
            get_split_patient_ids(data, "holdout")

    def test_get_split_indices_sub_prefix_stripped(self):
        data = _fake_split_data(train=("XYZ",), val=(), test=())
        items = [SimpleNamespace(patient_id="sub-XYZ"), SimpleNamespace(patient_id="sub-ABC")]
        result = get_split_indices_for_dataset(items, data, "train")
        assert result == [0]

    def test_get_split_indices_bare_id(self):
        data = _fake_split_data(train=("XYZ",), val=(), test=())
        items = [SimpleNamespace(patient_id="XYZ"), SimpleNamespace(patient_id="ABC")]
        result = get_split_indices_for_dataset(items, data, "train")
        assert result == [0]

    def test_get_split_indices_unknown_split_raises(self):
        data = _fake_split_data()
        with pytest.raises(ValueError):
            get_split_indices_for_dataset([], data, "holdout")


# ---------------------------------------------------------------------------
# Section B — Integration tests (auto-skip when CSVs not generated)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def downstream_splits():
    paths = split_csv_paths("downstream")
    for p in paths.values():
        if not Path(p).exists():
            pytest.skip(f"Downstream splits not found: {p}")
    return {k: pd.read_csv(v) for k, v in paths.items()}


@pytest.fixture(scope="module")
def pretrain_splits():
    paths = split_csv_paths("pretrain")
    for p in paths.values():
        if not Path(p).exists():
            pytest.skip(f"Pretrain splits not found: {p}")
    return {k: pd.read_csv(v) for k, v in paths.items()}


# ── Schema ────────────────────────────────────────────────────────────────────

def test_downstream_required_columns(downstream_splits):
    required = {"Pseudonym", "diagnosis", "converter_status", "sex", "age", "n_scans"}
    for split_name, df in downstream_splits.items():
        missing = required - set(df.columns)
        assert not missing, f"Downstream {split_name} missing columns: {missing}"


def test_pretrain_required_columns(pretrain_splits):
    required = {"Pseudonym", "diagnosis", "sex", "age", "n_scans"}
    for split_name, df in pretrain_splits.items():
        missing = required - set(df.columns)
        assert not missing, f"Pretrain {split_name} missing columns: {missing}"


def test_downstream_no_nulls_key_columns(downstream_splits):
    for col in ("Pseudonym", "diagnosis", "converter_status"):
        for split_name, df in downstream_splits.items():
            nulls = df[col].isna().sum()
            assert nulls == 0, f"Downstream {split_name}.{col}: {nulls} nulls"


def test_no_zero_scan_patients_downstream(downstream_splits):
    for split_name, df in downstream_splits.items():
        bad = (df["n_scans"] <= 0).sum()
        assert bad == 0, f"Downstream {split_name}: {bad} rows with n_scans <= 0"


def test_no_zero_scan_patients_pretrain(pretrain_splits):
    for split_name, df in pretrain_splits.items():
        bad = (df["n_scans"] <= 0).sum()
        assert bad == 0, f"Pretrain {split_name}: {bad} rows with n_scans <= 0"


def test_downstream_converter_status_binary(downstream_splits):
    for split_name, df in downstream_splits.items():
        vals = set(df["converter_status"].unique())
        assert vals <= {0, 1}, f"Downstream {split_name} converter_status has unexpected values: {vals}"
        # consistency: converter_status=1 ↔ diagnosis='converter'
        mismatch = df[
            ((df["diagnosis"] == "converter") & (df["converter_status"] != 1)) |
            ((df["diagnosis"] != "converter") & (df["converter_status"] != 0))
        ]
        assert mismatch.empty, (
            f"Downstream {split_name}: {len(mismatch)} rows with mismatched diagnosis/converter_status"
        )


# ── Uniqueness within each CSV ────────────────────────────────────────────────

def test_downstream_no_intra_split_duplicates(downstream_splits):
    for split_name, df in downstream_splits.items():
        dupes = df["Pseudonym"].duplicated().sum()
        assert dupes == 0, f"Downstream {split_name}: {dupes} duplicate patient IDs"


def test_pretrain_no_intra_split_duplicates(pretrain_splits):
    for split_name, df in pretrain_splits.items():
        dupes = df["Pseudonym"].duplicated().sum()
        assert dupes == 0, f"Pretrain {split_name}: {dupes} duplicate patient IDs"


# ── Pairwise-disjoint ─────────────────────────────────────────────────────────

def _check_pairwise_disjoint(splits: dict, label: str) -> None:
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        ids_a = set(splits[a]["Pseudonym"])
        ids_b = set(splits[b]["Pseudonym"])
        shared = ids_a & ids_b
        assert not shared, (
            f"{label} {a}∩{b}: {len(shared)} shared patients — "
            f"examples: {sorted(shared)[:5]}"
        )


def test_downstream_pairwise_disjoint(downstream_splits):
    _check_pairwise_disjoint(downstream_splits, "Downstream")


def test_pretrain_pairwise_disjoint(pretrain_splits):
    _check_pairwise_disjoint(pretrain_splits, "Pretrain")


# ── Diagnosis scope ───────────────────────────────────────────────────────────

def test_downstream_diagnosis_scope(downstream_splits):
    allowed = {"mci", "converter"}
    all_diagnoses = set(pd.concat(downstream_splits.values())["diagnosis"].unique())
    unexpected = all_diagnoses - allowed
    assert not unexpected, f"Downstream splits contain unexpected diagnoses: {unexpected}"


def test_pretrain_diagnosis_scope(pretrain_splits):
    allowed = {"mci", "converter", "healthy", "ad"}
    all_diagnoses = set(pd.concat(pretrain_splits.values())["diagnosis"].unique())
    unexpected = all_diagnoses - allowed
    assert not unexpected, f"Pretrain splits contain unexpected diagnoses: {unexpected}"


def test_no_scd_in_downstream(downstream_splits):
    combined = pd.concat(downstream_splits.values())
    scd_rows = combined[combined["diagnosis"] == "scd"]
    assert scd_rows.empty, f"Downstream splits contain {len(scd_rows)} SCD patients"


def test_no_scd_in_pretrain(pretrain_splits):
    combined = pd.concat(pretrain_splits.values())
    scd_rows = combined[combined["diagnosis"] == "scd"]
    assert scd_rows.empty, f"Pretrain splits contain {len(scd_rows)} SCD patients"


# ── Downstream leakage rule — 5 independent assertions ────────────────────────

def test_pretrain_train_disjoint_from_downstream_val(pretrain_splits, downstream_splits):
    pretrain_train  = set(pretrain_splits["train"]["Pseudonym"])
    downstream_val  = set(downstream_splits["val"]["Pseudonym"])
    shared = pretrain_train & downstream_val
    assert not shared, (
        f"LEAK: {len(shared)} downstream-val patients appear in pretrain-train — "
        f"examples: {sorted(shared)[:5]}"
    )


def test_pretrain_train_disjoint_from_downstream_test(pretrain_splits, downstream_splits):
    pretrain_train  = set(pretrain_splits["train"]["Pseudonym"])
    downstream_test = set(downstream_splits["test"]["Pseudonym"])
    shared = pretrain_train & downstream_test
    assert not shared, (
        f"LEAK: {len(shared)} downstream-test patients appear in pretrain-train — "
        f"examples: {sorted(shared)[:5]}"
    )


def test_pretrain_val_disjoint_from_downstream_test(pretrain_splits, downstream_splits):
    pretrain_val    = set(pretrain_splits["val"]["Pseudonym"])
    downstream_test = set(downstream_splits["test"]["Pseudonym"])
    shared = pretrain_val & downstream_test
    assert not shared, (
        f"LEAK: {len(shared)} downstream-test patients appear in pretrain-val — "
        f"examples: {sorted(shared)[:5]}"
    )


def test_downstream_test_subset_of_pretrain_test(pretrain_splits, downstream_splits):
    pretrain_test   = set(pretrain_splits["test"]["Pseudonym"])
    downstream_test = set(downstream_splits["test"]["Pseudonym"])
    missing = downstream_test - pretrain_test
    assert not missing, (
        f"{len(missing)} downstream-test patients not found in pretrain-test — "
        f"examples: {sorted(missing)[:5]}"
    )


def test_downstream_val_subset_of_pretrain_val(pretrain_splits, downstream_splits):
    pretrain_val   = set(pretrain_splits["val"]["Pseudonym"])
    downstream_val = set(downstream_splits["val"]["Pseudonym"])
    missing = downstream_val - pretrain_val
    assert not missing, (
        f"{len(missing)} downstream-val patients not found in pretrain-val — "
        f"examples: {sorted(missing)[:5]}"
    )


# ── Stratification balance (60/20/20 ± 10 pp per cohort) ─────────────────────

def _check_balance(splits: dict, diagnoses: list[str], label: str) -> None:
    for diagnosis in diagnoses:
        counts = {s: len(df[df["diagnosis"] == diagnosis]) for s, df in splits.items()}
        total = sum(counts.values())
        if total == 0:
            continue
        for split_name, expected in [("train", 0.60), ("val", 0.20), ("test", 0.20)]:
            actual = counts[split_name] / total
            assert abs(actual - expected) < 0.10, (
                f"{label} {diagnosis}/{split_name}: {actual:.2%} (expected ~{expected:.0%} ±10pp); "
                f"counts={counts}"
            )


def test_downstream_stratification_balance(downstream_splits):
    _check_balance(downstream_splits, ["mci", "converter"], "Downstream")


def test_pretrain_stratification_balance_free_split(pretrain_splits, downstream_splits):
    # Only check patients that were freely split — downstream-reserved patients are
    # force-assigned to pretrain val/test, which intentionally distorts proportions.
    downstream_all = set().union(*(set(df["Pseudonym"]) for df in downstream_splits.values()))
    free = {s: df[~df["Pseudonym"].isin(downstream_all)] for s, df in pretrain_splits.items()}
    _check_balance(free, ["mci", "converter", "healthy", "ad"], "Pretrain-free")


# ── _all_splits_patient_info.csv coverage ────────────────────────────────────

def test_patient_info_covers_all_downstream(downstream_splits):
    info_path = splits_dir("downstream") / "_all_splits_patient_info.csv"
    if not info_path.exists():
        pytest.skip(f"Patient info file not found: {info_path}")
    info = pd.read_csv(info_path)
    assert "Pseudonym" in info.columns, "Downstream info CSV missing Pseudonym column"
    assert "diagnosis" in info.columns, "Downstream info CSV missing diagnosis column"
    info_ids = set(info["Pseudonym"].astype(str))
    all_ids = set().union(*(set(df["Pseudonym"].astype(str)) for df in downstream_splits.values()))
    missing = all_ids - info_ids
    assert not missing, f"Downstream: {len(missing)} patients not in _all_splits_patient_info.csv"


def test_patient_info_covers_all_pretrain(pretrain_splits):
    info_path = splits_dir("pretrain") / "_all_splits_patient_info.csv"
    if not info_path.exists():
        pytest.skip(f"Patient info file not found: {info_path}")
    info = pd.read_csv(info_path)
    assert "Pseudonym" in info.columns, "Pretrain info CSV missing Pseudonym column"
    assert "diagnosis" in info.columns, "Pretrain info CSV missing diagnosis column"
    info_ids = set(info["Pseudonym"].astype(str))
    all_ids = set().union(*(set(df["Pseudonym"].astype(str)) for df in pretrain_splits.values()))
    missing = all_ids - info_ids
    assert not missing, f"Pretrain: {len(missing)} patients not in _all_splits_patient_info.csv"


# ── Smoke — run_full_audit ────────────────────────────────────────────────────

def test_run_full_audit_downstream(downstream_splits):  # noqa: ARG001 — fixture triggers skip if absent
    run_full_audit(split_csv_paths("downstream"), verbose=False)


def test_run_full_audit_pretrain(pretrain_splits):  # noqa: ARG001
    run_full_audit(split_csv_paths("pretrain"), verbose=False)
