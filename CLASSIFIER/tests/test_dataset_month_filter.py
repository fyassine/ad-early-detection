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
