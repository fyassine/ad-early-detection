from pathlib import Path

import pandas as pd
import pytest

from CLASSIFIER.common.pooled_data import COHORT_ROOTS, build_multicohort_bundle

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POOLED_TEST_CSV = _REPO_ROOT / "DATA" / "POOLED_ADNI_DELCODE" / "SPLITS" / "downstream" / "test.csv"


def _synthetic_pooled_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "delcode-subj-1",
                "cohort": "delcode",
                "converter_status": 1,
                "sex": "f",
                "age": 70,
                "n_scans": 2,
                "allowed_days": "",
                "allowed_months": "0;12",
            },
            {
                "subject_id": "ADNI-subj-1",
                "cohort": "adni",
                "converter_status": 0,
                "sex": "m",
                "age": 68,
                "n_scans": 2,
                "allowed_days": "0;365",
                "allowed_months": "",
            },
        ]
    )


class TestGuards:
    def test_missing_cohort_column_raises(self):
        df = _synthetic_pooled_df().drop(columns=["cohort"])
        with pytest.raises(ValueError, match="cohort"):
            build_multicohort_bundle(df)

    def test_all_null_native_allow_column_raises(self):
        df = _synthetic_pooled_df()
        df.loc[df["cohort"] == "delcode", "allowed_months"] = ""
        with pytest.raises(ValueError, match="allowed_months"):
            build_multicohort_bundle(df, min_visits=None)

    def test_unknown_cohort_raises(self):
        df = _synthetic_pooled_df()
        df.loc[0, "cohort"] = "unknown_cohort"
        with pytest.raises(ValueError, match="Unknown cohort"):
            build_multicohort_bundle(df, min_visits=None)

    @pytest.mark.parametrize("forbidden_kwarg", ["cohort", "cohorts_csv"])
    def test_forbidden_dataset_kwargs_raise(self, forbidden_kwarg):
        df = _synthetic_pooled_df()
        with pytest.raises(ValueError, match=forbidden_kwarg):
            build_multicohort_bundle(df, **{forbidden_kwarg: "x"})

    def test_cohort_roots_cover_all_three_cohorts(self):
        assert set(COHORT_ROOTS) == {"delcode", "adni", "oasis3"}


@pytest.mark.skipif(not _POOLED_TEST_CSV.exists(), reason="pooled downstream splits not built")
class TestRealPooledBundle:
    def test_bundle_spans_both_cohorts_with_pooled_test_split(self):
        df = pd.read_csv(_POOLED_TEST_CSV)
        bundle = build_multicohort_bundle(df, min_visits=2)

        cohorts_seen = {item["cohort"] for item in bundle.items}
        assert cohorts_seen == {"adni", "delcode"}
        assert len(bundle.items) == len(bundle.labels) == len(bundle.groups)
        assert len(bundle.items) == len(df)
        for item in bundle.items:
            assert item["n_scans"] >= 2
            assert len(item["graphs"]) == item["n_scans"]
