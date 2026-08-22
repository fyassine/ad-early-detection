import numpy as np
import pandas as pd
import pytest

from CLASSIFIER.common.dataset import ClassificationDataset
from CLASSIFIER.model.GAAE.dataset import GraphDatasetInMemoryFiltered
from CLASSIFIER.model.GELSTM.dataset import LongitudinalSubjectDataset

_SUFFIX = "_whole_brain_correlation_matrix_z_transformed.npz"


def _write_npz(directory, pid, month):
    arr = np.eye(4, dtype=np.float32)
    np.savez(directory / f"sub-{pid}_ses-01_M{month}_task-rest{_SUFFIX}", array=arr)


@pytest.fixture
def matrices_dir(tmp_path):
    d = tmp_path / "matrices"
    d.mkdir()
    for month in (0, 12, 24):
        _write_npz(d, "X1", month)
    return d


class TestGraphDatasetInMemoryFilteredMonthFilter:
    def test_drops_files_outside_allowed_months(self, tmp_path, matrices_dir):
        filter_csv = tmp_path / "filter.csv"
        pd.DataFrame({"Pseudonym": ["X1"], "allowed_months": ["0;12"]}).to_csv(
            filter_csv, index=False
        )
        ds = GraphDatasetInMemoryFiltered(
            root=str(matrices_dir),
            adjacency_args={"k": 1},
            filter_csv_path=str(filter_csv),
        )
        assert len(ds) == 2

    def test_no_allowed_months_column_keeps_all_files(self, tmp_path, matrices_dir):
        filter_csv = tmp_path / "filter.csv"
        pd.DataFrame({"Pseudonym": ["X1"]}).to_csv(filter_csv, index=False)
        ds = GraphDatasetInMemoryFiltered(
            root=str(matrices_dir),
            adjacency_args={"k": 1},
            filter_csv_path=str(filter_csv),
        )
        assert len(ds) == 3


class TestClassificationDatasetMonthFilter:
    def test_drops_files_outside_allowed_months(self, tmp_path, matrices_dir):
        filter_csv = tmp_path / "filter.csv"
        pd.DataFrame({"Pseudonym": ["X1"], "allowed_months": ["0;12"]}).to_csv(
            filter_csv, index=False
        )
        ds = ClassificationDataset(
            root=str(matrices_dir),
            adjacency_function=lambda m, k: np.ones_like(m),
            adjacency_args={"k": 1},
            filter_csv_path=str(filter_csv),
        )
        assert len(ds) == 2


class TestLongitudinalSubjectDatasetMonthFilter:
    def test_drops_visits_outside_allowed_months(self, matrices_dir):
        subject_df = pd.DataFrame(
            {
                "Pseudonym": ["X1"],
                "diagnosis": ["converter"],
                "sex": ["m"],
                "age": [70],
                "allowed_months": ["0;12"],
            }
        )
        cohorts = pd.DataFrame(
            {
                "Pseudonym": ["X1", "X1", "X1"],
                "visit": ["M0", "M12", "M24"],
            }
        )
        cohorts_csv = matrices_dir.parent / "cohorts.csv"
        cohorts.to_csv(cohorts_csv, index=False)

        ds = LongitudinalSubjectDataset(
            matrices_dir=str(matrices_dir),
            subject_df=subject_df,
            cohorts_csv=str(cohorts_csv),
        )
        assert len(ds) == 1
        item = ds[0]
        assert item["visit_months"] == [0, 12]
        assert item["n_scans"] == 2

    def test_no_allowed_months_column_keeps_all_visits(self, matrices_dir):
        subject_df = pd.DataFrame(
            {
                "Pseudonym": ["X1"],
                "diagnosis": ["converter"],
                "sex": ["m"],
                "age": [70],
            }
        )
        cohorts = pd.DataFrame(
            {
                "Pseudonym": ["X1", "X1", "X1"],
                "visit": ["M0", "M12", "M24"],
            }
        )
        cohorts_csv = matrices_dir.parent / "cohorts.csv"
        cohorts.to_csv(cohorts_csv, index=False)

        ds = LongitudinalSubjectDataset(
            matrices_dir=str(matrices_dir),
            subject_df=subject_df,
            cohorts_csv=str(cohorts_csv),
        )
        assert len(ds) == 1
        assert ds[0]["n_scans"] == 3


class TestLongitudinalSubjectDatasetVisitWindow:
    """min_visits / max_visits — mirrors BrainTokenGTAdapter.prepare_data's
    `it["n_scans"] >= min_visits` keep-rule followed by `window_item` truncation
    (BRAINTOKENGT/adapter.py:167-169), so the two pipelines see the same cohort.
    """

    @pytest.fixture
    def two_subject_df(self):
        return pd.DataFrame(
            {
                "Pseudonym": ["X1", "X2"],
                "diagnosis": ["converter", "mci"],
                "sex": ["m", "f"],
                "age": [70, 65],
            }
        )

    @pytest.fixture
    def two_subject_matrices_and_cohorts(self, tmp_path):
        d = tmp_path / "matrices"
        d.mkdir()
        for month in (0, 12, 24):
            _write_npz(d, "X1", month)  # X1: 3 visits
        for month in (0, 12):
            _write_npz(d, "X2", month)  # X2: 2 visits
        cohorts = pd.DataFrame(
            {
                "Pseudonym": ["X1", "X1", "X1", "X2", "X2"],
                "visit": ["M0", "M12", "M24", "M0", "M12"],
            }
        )
        cohorts_csv = tmp_path / "cohorts.csv"
        cohorts.to_csv(cohorts_csv, index=False)
        return d, cohorts_csv

    def test_min_visits_drops_subjects_below_floor(self, two_subject_df, two_subject_matrices_and_cohorts):
        matrices_dir, cohorts_csv = two_subject_matrices_and_cohorts
        ds = LongitudinalSubjectDataset(
            matrices_dir=str(matrices_dir),
            subject_df=two_subject_df,
            cohorts_csv=str(cohorts_csv),
            min_visits=3,
        )
        assert ds.get_subject_ids() == ["X1"]

    def test_min_visits_evaluated_before_max_visits_truncation(
        self, two_subject_df, two_subject_matrices_and_cohorts
    ):
        """min_visits=2, max_visits=3: X2 (2 visits) is kept unpadded, not
        dropped — the floor checks the FULL visit count, not the post-truncation
        one, matching BrainTokenGT's order (filter, then window_item)."""
        matrices_dir, cohorts_csv = two_subject_matrices_and_cohorts
        ds = LongitudinalSubjectDataset(
            matrices_dir=str(matrices_dir),
            subject_df=two_subject_df,
            cohorts_csv=str(cohorts_csv),
            min_visits=2,
            max_visits=3,
        )
        assert set(ds.get_subject_ids()) == {"X1", "X2"}
        by_id = {s["subject_id"]: s for s in ds.subjects}
        assert by_id["X1"]["n_scans"] == 3
        assert by_id["X2"]["n_scans"] == 2

    def test_min_visits_none_applies_no_floor(self, two_subject_df, two_subject_matrices_and_cohorts):
        matrices_dir, cohorts_csv = two_subject_matrices_and_cohorts
        ds = LongitudinalSubjectDataset(
            matrices_dir=str(matrices_dir),
            subject_df=two_subject_df,
            cohorts_csv=str(cohorts_csv),
        )
        assert set(ds.get_subject_ids()) == {"X1", "X2"}
