"""Unit tests for DATA/manifest/build_pooled_assets.py.

Section A — pure logic on synthetic frames, no filesystem access.
Section B — integration smoke tests against on-disk ADNI/DELCODE splits,
auto-skipped when either cohort's downstream/pretrain CSVs are missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DATA.manifest.build_pooled_assets import (
    ADNI_DOWNSTREAM_DIR,
    ADNI_MANIFEST,
    ADNI_MATRICES_DIR,
    ADNI_PRETRAIN_DIR,
    DELCODE_DOWNSTREAM_DIR,
    DELCODE_MATRICES_DIR,
    DELCODE_PRETRAIN_DIR,
    POOLED_DOWNSTREAM_COLUMNS,
    POOLED_PRETRAIN_COLUMNS,
    _adni_pretrain_to_pooled,
    _dayscoded_downstream_to_pooled,
    _delcode_downstream_to_pooled,
    _delcode_pretrain_to_pooled,
    build_adni_pretrain_splits,
    build_pooled_downstream_splits,
    build_pooled_pretrain_splits,
    build_symlink_farm,
)

# ---------------------------------------------------------------------------
# Section A — pure logic
# ---------------------------------------------------------------------------


def _delcode_downstream_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pseudonym": "abc123",
                "diagnosis": "converter",
                "converter_status": 1,
                "sex": "f",
                "age": 70,
                "n_scans": 3,
                "allowed_months": "0;12;24",
            },
            {
                "Pseudonym": "def456",
                "diagnosis": "mci",
                "converter_status": 0,
                "sex": "m",
                "age": 65,
                "n_scans": 1,
                "allowed_months": "",
            },
        ]
    )


def _dayscoded_downstream_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "ADNI002S1261",
                "label": "converter",
                "converter_status": 1,
                "sex": "m",
                "age": 72.5,
                "n_scans": 2,
                "allowed_days": "778;1486",
            }
        ]
    )


class TestHarmonizeDownstream:
    def test_delcode_row_has_pooled_schema(self):
        out = _delcode_downstream_to_pooled(_delcode_downstream_df())
        assert list(out.columns) == POOLED_DOWNSTREAM_COLUMNS
        assert out.loc[0, "subject_id"] == "abc123"
        assert out.loc[0, "cohort"] == "delcode"
        assert out.loc[0, "allowed_months"] == "0;12;24"
        assert out.loc[0, "allowed_days"] == ""

    def test_dayscoded_row_has_pooled_schema(self):
        out = _dayscoded_downstream_to_pooled(_dayscoded_downstream_df(), cohort="adni")
        assert list(out.columns) == POOLED_DOWNSTREAM_COLUMNS
        assert out.loc[0, "cohort"] == "adni"
        assert out.loc[0, "allowed_days"] == "778;1486"
        assert out.loc[0, "allowed_months"] == ""

    def test_min_visits_filter_drops_single_visit_subjects(self):
        delcode = _delcode_downstream_to_pooled(_delcode_downstream_df())
        merged = delcode[delcode["n_scans"] >= 2]
        assert list(merged["subject_id"]) == ["abc123"]


class TestPretrainHarmonize:
    def test_adni_pretrain_row_has_pseudonym_schema(self):
        adni_df = pd.DataFrame(
            [{"subject_id": "ADNI002S1261", "label": "stable", "sex": "f", "age": 71.0, "n_scans": 1}]
        )
        out = _adni_pretrain_to_pooled(adni_df)
        assert list(out.columns) == POOLED_PRETRAIN_COLUMNS
        assert out.loc[0, "Pseudonym"] == "ADNI002S1261"
        assert out.loc[0, "diagnosis"] == "stable"

    def test_delcode_pretrain_row_requires_pseudonym_column(self):
        delcode_df = pd.DataFrame(
            [{"Pseudonym": "abc123", "diagnosis": "mci", "sex": "f", "age": 65, "n_scans": 2}]
        )
        out = _delcode_pretrain_to_pooled(delcode_df)
        assert list(out.columns) == POOLED_PRETRAIN_COLUMNS

    def test_delcode_pretrain_missing_column_raises(self):
        bad_df = pd.DataFrame([{"Pseudonym": "abc123", "sex": "f", "age": 65, "n_scans": 2}])
        with pytest.raises(ValueError, match="diagnosis"):
            _delcode_pretrain_to_pooled(bad_df)


# ---------------------------------------------------------------------------
# Section B — integration smoke tests (auto-skip if inputs absent)
# ---------------------------------------------------------------------------

_ADNI_READY = ADNI_MANIFEST.exists() and (ADNI_DOWNSTREAM_DIR / "val.csv").exists()
_DELCODE_READY = (DELCODE_DOWNSTREAM_DIR / "train.csv").exists() and (
    DELCODE_PRETRAIN_DIR / "train.csv"
).exists()


@pytest.mark.skipif(not _ADNI_READY, reason="ADNI manifest/downstream splits not found")
def test_adni_pretrain_splits_no_leakage_and_covers_downstream_holdout():
    splits = build_adni_pretrain_splits(seed=42)
    downstream_val = set(pd.read_csv(ADNI_DOWNSTREAM_DIR / "val.csv")["subject_id"].astype(str))
    downstream_test = set(pd.read_csv(ADNI_DOWNSTREAM_DIR / "test.csv")["subject_id"].astype(str))

    train_ids = set(splits["train"]["subject_id"])
    val_ids = set(splits["val"]["subject_id"])
    test_ids = set(splits["test"]["subject_id"])

    assert not (train_ids & downstream_val)
    assert not (train_ids & downstream_test)
    assert not (val_ids & downstream_test)
    assert downstream_val.issubset(val_ids)
    assert downstream_test.issubset(test_ids)
    # ADNI has no separate "healthy" pool: pretrain (min_sessions=1) must be a
    # superset of downstream (min_sessions=2).
    downstream_train = set(pd.read_csv(ADNI_DOWNSTREAM_DIR / "train.csv")["subject_id"].astype(str))
    assert downstream_train.issubset(train_ids | val_ids | test_ids)


@pytest.mark.skipif(not (_ADNI_READY and _DELCODE_READY), reason="ADNI or DELCODE splits not found")
def test_pooled_pretrain_splits_disjoint_and_pseudonym_schema():
    pooled = build_pooled_pretrain_splits(seed=42)
    for split_name, df in pooled.items():
        assert list(df.columns) == POOLED_PRETRAIN_COLUMNS, split_name
    ids = [set(df["Pseudonym"]) for df in pooled.values()]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])


@pytest.mark.skipif(not (_ADNI_READY and _DELCODE_READY), reason="ADNI or DELCODE splits not found")
def test_pooled_downstream_splits_match_preregistered_counts():
    pooled = build_pooled_downstream_splits(min_visits=2)
    for split_name, df in pooled.items():
        assert list(df.columns) == POOLED_DOWNSTREAM_COLUMNS, split_name
        assert (df["n_scans"] >= 2).all()
        both = df["allowed_days"].astype(str).str.len().gt(0) & df["allowed_months"].astype(
            str
        ).str.len().gt(0)
        assert not both.any()

    ids = [set(df["subject_id"]) for df in pooled.values()]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])

    # Pre-registered in DOCS/flipped/PLAN.md and DOCS/temporal-first-ablation.md.
    assert len(pooled["train"]) + len(pooled["val"]) == 248
    assert len(pooled["test"]) == 64


@pytest.mark.skipif(
    not (DELCODE_MATRICES_DIR.is_dir() and ADNI_MATRICES_DIR.is_dir()),
    reason="DELCODE/ADNI FC matrices not found on disk",
)
def test_symlink_farm_is_idempotent():
    first_created, _ = build_symlink_farm()
    second_created, second_present = build_symlink_farm()
    assert second_created == 0
    if first_created:
        assert second_present >= first_created
