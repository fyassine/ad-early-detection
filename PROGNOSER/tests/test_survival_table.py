"""Tests for PROGNOSER/common/survival_table.py."""
import pytest
import numpy as np
import pandas as pd

from PROGNOSER.common.survival_table import build_survival_table, make_xte, filter_to_split


# ── CSV fixtures ──────────────────────────────────────────────────────────────

_COHORT_ROWS = [
    # converter: mci baseline, reaches ad at M24
    {"Pseudonym": "s01", "visit": "M0",  "diagnosis": "mci",  "age": 65, "sex": "m",
     "mmstot": 28.0, "cdrglobal": 0.5, "ApoE": "e3/e4"},
    {"Pseudonym": "s01", "visit": "M12", "diagnosis": "mci",  "age": 66, "sex": "m",
     "mmstot": 27.0, "cdrglobal": 0.5, "ApoE": "e3/e4"},
    {"Pseudonym": "s01", "visit": "M24", "diagnosis": "ad",   "age": 67, "sex": "m",
     "mmstot": 24.0, "cdrglobal": 1.0, "ApoE": "e3/e4"},
    # non-converter: mci baseline, last visit M36
    {"Pseudonym": "s02", "visit": "M0",  "diagnosis": "mci",  "age": 70, "sex": "f",
     "mmstot": 26.0, "cdrglobal": 0.5, "ApoE": "e3/e3"},
    {"Pseudonym": "s02", "visit": "M36", "diagnosis": "mci",  "age": 73, "sex": "f",
     "mmstot": 25.0, "cdrglobal": 0.5, "ApoE": "e3/e3"},
    # excluded: baseline diagnosis is 'ad' (not mci/converter)
    {"Pseudonym": "s03", "visit": "M0",  "diagnosis": "ad",   "age": 60, "sex": "m",
     "mmstot": 20.0, "cdrglobal": 2.0, "ApoE": "e4/e4"},
    # converter baseline label
    {"Pseudonym": "s04", "visit": "M0",  "diagnosis": "converter", "age": 68, "sex": "f",
     "mmstot": 27.0, "cdrglobal": 0.5, "ApoE": "e3/e4"},
    {"Pseudonym": "s04", "visit": "M12", "diagnosis": "ad",         "age": 69, "sex": "f",
     "mmstot": 24.0, "cdrglobal": 1.0, "ApoE": "e3/e4"},
]


@pytest.fixture()
def cohort_csv(tmp_path):
    p = tmp_path / "cohorts.csv"
    pd.DataFrame(_COHORT_ROWS).to_csv(p, index=False)
    return p


# ── build_survival_table ──────────────────────────────────────────────────────

def test_converter_duration_and_event(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    row = tbl[tbl["subject_id"] == "s01"].iloc[0]
    assert row["event_observed"] == 1
    assert row["duration"] == pytest.approx(24.0)


def test_non_converter_duration_and_event(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    row = tbl[tbl["subject_id"] == "s02"].iloc[0]
    assert row["event_observed"] == 0
    assert row["duration"] == pytest.approx(36.0)


def test_excluded_subject_not_in_table(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    assert "s03" not in tbl["subject_id"].values


def test_converter_baseline_label_included(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    row = tbl[tbl["subject_id"] == "s04"].iloc[0]
    assert row["event_observed"] == 1
    assert row["duration"] == pytest.approx(12.0)


def test_all_expected_subjects_present(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    assert set(tbl["subject_id"]) == {"s01", "s02", "s04"}


def test_age_extracted(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    assert "age" in tbl.columns
    row = tbl[tbl["subject_id"] == "s01"].iloc[0]
    assert row["age"] == pytest.approx(65.0)


def test_apoe4_carrier_flag(cohort_csv):
    tbl = build_survival_table(cohort_csv, include_features=("apoe4",))
    # s01 has e3/e4 → carrier (apoe4=1)
    assert tbl[tbl["subject_id"] == "s01"]["apoe4"].iloc[0] == 1
    # s02 has e3/e3 → not carrier (apoe4=0)
    assert tbl[tbl["subject_id"] == "s02"]["apoe4"].iloc[0] == 0


def test_missing_diagnosis_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"Pseudonym": "x", "visit": "M0"}]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="diagnosis"):
        build_survival_table(bad_csv)


def test_missing_visit_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"Pseudonym": "x", "diagnosis": "mci"}]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="visit"):
        build_survival_table(bad_csv)


def test_no_subject_id_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"x": 1, "visit": "M0", "diagnosis": "mci"}]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="subject ID"):
        build_survival_table(bad_csv)


# ── make_xte ──────────────────────────────────────────────────────────────────

def test_make_xte_shapes(cohort_csv):
    tbl = build_survival_table(cohort_csv, include_features=("age", "sex"))
    X, T, E, used = make_xte(tbl, feature_cols=["age", "sex"])
    n = len(used)
    assert X.shape == (n, 2)
    assert T.shape == (n,)
    assert E.shape == (n,)
    assert set(np.unique(E)).issubset({0, 1})


def test_make_xte_missing_column_raises(cohort_csv):
    tbl = build_survival_table(cohort_csv)
    with pytest.raises(KeyError, match="nonexistent_col"):
        make_xte(tbl, feature_cols=["nonexistent_col"])


# ── filter_to_split ───────────────────────────────────────────────────────────

def test_filter_to_split(cohort_csv, tmp_path):
    tbl = build_survival_table(cohort_csv)
    split_csv = tmp_path / "train.csv"
    pd.DataFrame({"Pseudonym": ["s01", "s02"]}).to_csv(split_csv, index=False)

    filtered = filter_to_split(tbl, splits_dir=tmp_path, split="train")
    assert set(filtered["subject_id"]) == {"s01", "s02"}
    assert "s04" not in filtered["subject_id"].values
