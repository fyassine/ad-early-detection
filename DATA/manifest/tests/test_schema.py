"""
Unit tests for DATA/manifest/schema.py's build-time assertions.

Section A — pure logic on synthetic DataFrames, no filesystem access (except
the two path-existence tests, which use tmp_path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DATA.manifest.schema import (
    MANIFEST_COLUMNS,
    assert_counts_match,
    assert_delta_t_monotonic,
    assert_every_subject_dir_contributes_sessions,
    assert_fc_paths_present,
    assert_no_cross_label_duplicates,
    assert_paths_exist_and_nonempty,
    assert_schema,
    assert_visit_index_contiguous,
    validate_manifest,
)


def _manifest(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal manifest DataFrame, filling any missing schema columns with None."""
    filled = [{col: row.get(col) for col in MANIFEST_COLUMNS} for row in rows]
    return pd.DataFrame(filled, columns=MANIFEST_COLUMNS)


class TestAssertSchema:
    def test_exact_columns_pass(self):
        df = _manifest([{"subject_id": "A"}])
        assert_schema(df)  # no raise

    def test_missing_column_raises(self):
        df = pd.DataFrame({"subject_id": ["A"]})
        with pytest.raises(ValueError):
            assert_schema(df)

    def test_extra_column_raises(self):
        df = _manifest([{"subject_id": "A"}])
        df["surprise"] = 1
        with pytest.raises(ValueError):
            assert_schema(df)


class TestAssertPathsExistAndNonempty:
    def test_missing_path_raises(self, tmp_path):
        df = _manifest([{"bold_path": str(tmp_path / "does_not_exist.nii.gz")}])
        with pytest.raises(ValueError, match="missing"):
            assert_paths_exist_and_nonempty(df)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.nii.gz"
        f.touch()
        df = _manifest([{"bold_path": str(f)}])
        with pytest.raises(ValueError, match="empty"):
            assert_paths_exist_and_nonempty(df)

    def test_nonempty_existing_path_passes(self, tmp_path):
        f = tmp_path / "real.nii.gz"
        f.write_bytes(b"data")
        df = _manifest([{"bold_path": str(f)}])
        assert_paths_exist_and_nonempty(df)  # no raise

    def test_null_path_is_skipped(self):
        df = _manifest([{"bold_path": None}])
        assert_paths_exist_and_nonempty(df)  # no raise


class TestAssertEverySubjectDirContributesSessions:
    def test_all_dirs_covered_passes(self):
        df = _manifest([{"subject_id": "A"}, {"subject_id": "B"}])
        assert_every_subject_dir_contributes_sessions(df, {"A", "B"}, cohort="test")

    def test_uncovered_dir_raises(self):
        df = _manifest([{"subject_id": "A"}])
        with pytest.raises(ValueError, match="zero sessions"):
            assert_every_subject_dir_contributes_sessions(df, {"A", "B"}, cohort="test")

    def test_acknowledged_empty_subject_does_not_raise(self):
        df = _manifest([{"subject_id": "A"}])
        acknowledged = assert_every_subject_dir_contributes_sessions(
            df, {"A", "B"}, cohort="test", known_empty_subjects=frozenset({"B"})
        )
        assert acknowledged == ["B"]

    def test_unacknowledged_dir_among_acknowledged_still_raises(self):
        df = _manifest([{"subject_id": "A"}])
        with pytest.raises(ValueError):
            assert_every_subject_dir_contributes_sessions(
                df, {"A", "B", "C"}, cohort="test", known_empty_subjects=frozenset({"B"})
            )


class TestAssertCountsMatch:
    def test_matching_counts_pass(self):
        df = _manifest([{"subject_id": "A"}, {"subject_id": "A"}, {"subject_id": "B"}])
        assert_counts_match(df, cohort="test", expected_subjects=2, expected_sessions=3)

    def test_subject_mismatch_raises(self):
        df = _manifest([{"subject_id": "A"}])
        with pytest.raises(ValueError):
            assert_counts_match(df, cohort="test", expected_subjects=2, expected_sessions=1)

    def test_session_mismatch_raises(self):
        df = _manifest([{"subject_id": "A"}])
        with pytest.raises(ValueError):
            assert_counts_match(df, cohort="test", expected_subjects=1, expected_sessions=2)


class TestAssertVisitIndexContiguous:
    def test_contiguous_passes(self):
        df = _manifest(
            [
                {"subject_id": "A", "visit_index": 0},
                {"subject_id": "A", "visit_index": 1},
                {"subject_id": "B", "visit_index": 0},
            ]
        )
        assert_visit_index_contiguous(df)

    def test_gap_raises(self):
        df = _manifest([{"subject_id": "A", "visit_index": 0}, {"subject_id": "A", "visit_index": 2}])
        with pytest.raises(ValueError):
            assert_visit_index_contiguous(df)

    def test_nonzero_start_raises(self):
        df = _manifest([{"subject_id": "A", "visit_index": 1}])
        with pytest.raises(ValueError):
            assert_visit_index_contiguous(df)


class TestAssertDeltaTMonotonic:
    def test_strictly_increasing_passes(self):
        df = _manifest(
            [
                {"subject_id": "A", "visit_index": 0, "delta_t_months": 0.0},
                {"subject_id": "A", "visit_index": 1, "delta_t_months": 12.0},
            ]
        )
        assert_delta_t_monotonic(df)

    def test_equal_values_raise(self):
        df = _manifest(
            [
                {"subject_id": "A", "visit_index": 0, "delta_t_months": 5.0},
                {"subject_id": "A", "visit_index": 1, "delta_t_months": 5.0},
            ]
        )
        with pytest.raises(ValueError):
            assert_delta_t_monotonic(df)

    def test_decreasing_raises(self):
        df = _manifest(
            [
                {"subject_id": "A", "visit_index": 0, "delta_t_months": 12.0},
                {"subject_id": "A", "visit_index": 1, "delta_t_months": 5.0},
            ]
        )
        with pytest.raises(ValueError):
            assert_delta_t_monotonic(df)


class TestAssertFcPathsPresent:
    def test_all_present_passes(self):
        df = _manifest([{"subject_id": "A", "fc_path": "/x.npz"}, {"subject_id": "B", "fc_path": "/y.npz"}])
        assert_fc_paths_present(df, cohort="test")  # no raise

    def test_missing_raises(self):
        df = _manifest([{"subject_id": "A", "fc_path": "/x.npz"}, {"subject_id": "B", "fc_path": None}])
        with pytest.raises(ValueError, match="fc_path"):
            assert_fc_paths_present(df, cohort="test")


class TestAssertNoCrossLabelDuplicates:
    def test_disjoint_passes(self):
        assert_no_cross_label_duplicates({"A"}, {"B"}, cohort="test")

    def test_overlap_raises(self):
        with pytest.raises(ValueError):
            assert_no_cross_label_duplicates({"A", "B"}, {"B"}, cohort="test")


class TestValidateManifest:
    def test_valid_manifest_passes_and_summarizes(self, tmp_path):
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"data")
        df = _manifest(
            [
                {
                    "subject_id": "A",
                    "visit_index": 0,
                    "delta_t_months": 0.0,
                    "bold_path": str(f),
                }
            ]
        )
        summary = validate_manifest(df, cohort="test", subject_dirs_on_disk={"A"})
        assert summary == {
            "cohort": "test",
            "subjects": 1,
            "sessions": 1,
            "acknowledged_empty_subjects": [],
            "acknowledged_duplicate_day_subjects": [],
        }

    def test_bad_manifest_raises_before_writing(self, tmp_path):
        df = _manifest([{"subject_id": "A", "visit_index": 1, "delta_t_months": 0.0}])
        with pytest.raises(ValueError):
            validate_manifest(df, cohort="test", subject_dirs_on_disk={"A"})
