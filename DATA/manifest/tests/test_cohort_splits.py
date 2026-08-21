"""Unit tests for DATA/manifest/build_cohort_splits.py.

Section A — pure logic on synthetic manifests, no filesystem access.
Section B — integration smoke test against on-disk ADNI/OASIS-3 manifests +
demographics, auto-skipped when either is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DATA.manifest.build_cohort_splits import (
    _COHORT_PATHS,
    SPLIT_COLUMNS,
    build_cohort_splits,
    build_subject_table,
    split_subjects,
)
from DATA.manifest.schema import MANIFEST_COLUMNS


def _manifest(rows: list[dict]) -> pd.DataFrame:
    filled = [{col: row.get(col) for col in MANIFEST_COLUMNS} for row in rows]
    return pd.DataFrame(filled, columns=MANIFEST_COLUMNS)


def _demographics(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["subject_id", "sex", "age_at_baseline", "education", "source"])


def _two_session_subject(subject_id: str, label: str, days=(0, 100), fc="/x.npz") -> list[dict]:
    return [
        {
            "subject_id": subject_id,
            "label": label,
            "days_from_baseline": d,
            "fc_path": fc,
        }
        for d in days
    ]


# ---------------------------------------------------------------------------
# Section A — build_subject_table
# ---------------------------------------------------------------------------


class TestBuildSubjectTable:
    def test_eligible_subject_included(self):
        manifest = _manifest(_two_session_subject("A", "converter"))
        demog = _demographics([{"subject_id": "A", "sex": "f", "age_at_baseline": 70.0, "education": None, "source": "x"}])
        subjects = build_subject_table(manifest, demog, min_sessions=2)
        assert list(subjects.columns) == SPLIT_COLUMNS
        assert len(subjects) == 1
        row = subjects.iloc[0]
        assert row["subject_id"] == "A"
        assert row["label"] == "converter"
        assert row["converter_status"] == 1
        assert row["n_scans"] == 2
        assert row["allowed_days"] == "0;100"
        assert row["sex"] == "f"

    def test_below_min_sessions_excluded(self):
        manifest = _manifest(_two_session_subject("A", "stable", days=(0,)))
        demog = _demographics([{"subject_id": "A", "sex": "m", "age_at_baseline": 60.0, "education": None, "source": "x"}])
        subjects = build_subject_table(manifest, demog, min_sessions=2)
        assert subjects.empty

    def test_null_label_dropped(self):
        rows = _two_session_subject("A", "converter") + _two_session_subject("B", None)
        manifest = _manifest(rows)
        demog = _demographics(
            [
                {"subject_id": "A", "sex": "f", "age_at_baseline": 70.0, "education": None, "source": "x"},
                {"subject_id": "B", "sex": "m", "age_at_baseline": 65.0, "education": None, "source": "x"},
            ]
        )
        subjects = build_subject_table(manifest, demog, min_sessions=2)
        assert set(subjects["subject_id"]) == {"A"}

    def test_null_fc_path_dropped(self):
        manifest = _manifest(_two_session_subject("A", "converter", fc=None))
        demog = _demographics([{"subject_id": "A", "sex": "f", "age_at_baseline": 70.0, "education": None, "source": "x"}])
        subjects = build_subject_table(manifest, demog, min_sessions=2)
        assert subjects.empty

    def test_unrecognized_label_raises(self):
        manifest = _manifest(_two_session_subject("A", "ad"))
        demog = _demographics([{"subject_id": "A", "sex": "f", "age_at_baseline": 70.0, "education": None, "source": "x"}])
        with pytest.raises(ValueError, match="Unrecognized label"):
            build_subject_table(manifest, demog, min_sessions=2)

    def test_conflicting_label_across_sessions_raises(self):
        rows = [
            {"subject_id": "A", "label": "converter", "days_from_baseline": 0, "fc_path": "/x.npz"},
            {"subject_id": "A", "label": "stable", "days_from_baseline": 100, "fc_path": "/y.npz"},
        ]
        manifest = _manifest(rows)
        demog = _demographics([{"subject_id": "A", "sex": "f", "age_at_baseline": 70.0, "education": None, "source": "x"}])
        with pytest.raises(ValueError, match="conflicting labels"):
            build_subject_table(manifest, demog, min_sessions=2)

    def test_missing_demographics_raises(self):
        manifest = _manifest(_two_session_subject("A", "converter"))
        demog = _demographics([])
        with pytest.raises(ValueError, match="missing demographics"):
            build_subject_table(manifest, demog, min_sessions=2)

    def test_allowed_days_sorted(self):
        manifest = _manifest(_two_session_subject("A", "stable", days=(200, 0, 100)))
        demog = _demographics([{"subject_id": "A", "sex": "m", "age_at_baseline": 60.0, "education": None, "source": "x"}])
        subjects = build_subject_table(manifest, demog, min_sessions=2)
        assert subjects.iloc[0]["allowed_days"] == "0;100;200"


# ---------------------------------------------------------------------------
# Section A — split_subjects
# ---------------------------------------------------------------------------


class TestSplitSubjects:
    def _subjects(self, n_converter=10, n_stable=10) -> pd.DataFrame:
        rows = []
        for i in range(n_converter):
            rows.append({"subject_id": f"C{i}", "label": "converter", "converter_status": 1, "sex": "f", "age": 70.0, "n_scans": 2, "allowed_days": "0;100"})
        for i in range(n_stable):
            rows.append({"subject_id": f"S{i}", "label": "stable", "converter_status": 0, "sex": "m", "age": 65.0, "n_scans": 2, "allowed_days": "0;100"})
        return pd.DataFrame(rows, columns=SPLIT_COLUMNS)

    def test_no_subject_in_two_splits(self):
        splits = split_subjects(self._subjects(), seed=42)
        ids = [set(df["subject_id"]) for df in splits.values()]
        assert not (ids[0] & ids[1])
        assert not (ids[0] & ids[2])
        assert not (ids[1] & ids[2])

    def test_all_subjects_placed_exactly_once(self):
        subjects = self._subjects()
        splits = split_subjects(subjects, seed=42)
        placed = pd.concat(splits.values())["subject_id"]
        assert sorted(placed) == sorted(subjects["subject_id"])
        assert placed.duplicated().sum() == 0

    def test_deterministic_across_runs(self):
        subjects = self._subjects()
        a = split_subjects(subjects, seed=42)
        b = split_subjects(subjects, seed=42)
        for name in ("train", "val", "test"):
            assert sorted(a[name]["subject_id"]) == sorted(b[name]["subject_id"])

    def test_roughly_60_20_20_per_label(self):
        splits = split_subjects(self._subjects(n_converter=20, n_stable=20), seed=42)
        for label in ("converter", "stable"):
            counts = {name: (df["label"] == label).sum() for name, df in splits.items()}
            total = sum(counts.values())
            assert abs(counts["train"] / total - 0.6) < 0.15
            assert abs(counts["val"] / total - 0.2) < 0.15
            assert abs(counts["test"] / total - 0.2) < 0.15


# ---------------------------------------------------------------------------
# Section B — integration smoke test (auto-skip if manifests/demographics absent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cohort", ["adni", "oasis3"])
def test_build_cohort_splits_smoke(cohort):
    paths = _COHORT_PATHS[cohort]
    if not paths["manifest"].exists() or not paths["demographics"].exists():
        pytest.skip(f"{cohort}: manifest or demographics CSV not found")

    splits = build_cohort_splits(cohort, seed=42, min_sessions=2)
    for df in splits.values():
        assert list(df.columns) == SPLIT_COLUMNS if len(df) else True
    ids = [set(df["subject_id"]) for df in splits.values()]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])
