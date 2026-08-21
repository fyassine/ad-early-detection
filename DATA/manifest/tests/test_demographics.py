"""Integration tests for DATA/manifest/demographics.py, run against the real
on-disk ADNI/OASIS-3 metadata. Auto-skips per-cohort when its source data
isn't present (mirrors test_builders.py's pattern).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DATA.manifest.build_adni_manifest import DEFAULT_OUTPUT_CSV as ADNI_MANIFEST_CSV
from DATA.manifest.build_oasis3_manifest import DEFAULT_OUTPUT_CSV as OASIS3_MANIFEST_CSV
from DATA.manifest.demographics import (
    _ADNI_ARTIFACTS_DIR,
    DEMOGRAPHICS_COLUMNS,
    build_adni_demographics,
    build_oasis3_demographics,
)


def _skip_unless_dir(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Source not found: {path}")


def test_adni_demographics_covers_every_manifest_subject():
    _skip_unless_dir(_ADNI_ARTIFACTS_DIR)
    _skip_unless_dir(ADNI_MANIFEST_CSV)
    import pandas as pd

    df = build_adni_demographics()
    assert list(df.columns) == DEMOGRAPHICS_COLUMNS
    manifest_ids = set(pd.read_csv(ADNI_MANIFEST_CSV)["subject_id"].astype(str).unique())
    assert set(df["subject_id"]) == manifest_ids
    assert df["sex"].isin({"m", "f"}).all()
    assert df["age_at_baseline"].notna().all()


def test_oasis3_demographics_covers_every_manifest_subject():
    _skip_unless_dir(OASIS3_MANIFEST_CSV)
    import pandas as pd

    df = build_oasis3_demographics()
    assert list(df.columns) == DEMOGRAPHICS_COLUMNS
    manifest_ids = set(pd.read_csv(OASIS3_MANIFEST_CSV)["subject_id"].astype(str).unique())
    assert set(df["subject_id"]) == manifest_ids
    assert df["sex"].isin({"m", "f"}).all()
