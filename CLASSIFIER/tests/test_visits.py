import math

import pandas as pd
import pytest

from CLASSIFIER.common.visits import (
    allowed_months_map,
    month_allowed,
    parse_adni_protocol_month,
    parse_allowed_months,
    parse_day,
    parse_month,
    visit_identity,
)


class TestParseMonth:
    def test_extracts_month(self):
        assert parse_month("sub-X_ses-01_M12_task-rest_..._z_transformed.npz") == 12

    def test_extracts_zero_month(self):
        assert parse_month("sub-X_ses-01_M0_task-rest.npz") == 0

    def test_returns_none_when_absent(self):
        assert parse_month("sub-X_ses-01_task-rest.npz") is None


class TestParseAllowedMonths:
    def test_valid_semicolon_list(self):
        assert parse_allowed_months("0;12;24") == {0, 12, 24}

    def test_single_month(self):
        assert parse_allowed_months("36") == {36}

    def test_none_input_returns_none(self):
        assert parse_allowed_months(None) is None

    def test_nan_returns_none(self):
        assert parse_allowed_months(float("nan")) is None
        assert math.isnan(float("nan"))  # sanity on the fixture itself

    def test_empty_string_returns_none(self):
        assert parse_allowed_months("") is None

    def test_malformed_value_raises(self):
        with pytest.raises(ValueError):
            parse_allowed_months("twelve;24")


class TestAllowedMonthsMap:
    def test_builds_per_patient_map(self):
        df = pd.DataFrame(
            {
                "Pseudonym": ["A", "B"],
                "allowed_months": ["0;12", ""],
            }
        )
        result = allowed_months_map(df)
        assert result == {"A": {0, 12}, "B": None}

    def test_returns_none_when_column_absent(self):
        df = pd.DataFrame({"Pseudonym": ["A"]})
        assert allowed_months_map(df) is None


class TestMonthAllowed:
    def test_none_allowed_set_permits_any_file(self):
        assert month_allowed("sub-X_ses-01_M36_....npz", None) is True

    def test_month_in_set_is_kept(self):
        assert month_allowed("sub-X_ses-01_M12_....npz", {0, 12}) is True

    def test_month_not_in_set_is_dropped(self):
        assert month_allowed("sub-X_ses-01_M36_....npz", {0, 12}) is False

    def test_unparseable_month_is_dropped_when_restricted(self):
        assert month_allowed("sub-X_ses-01_....npz", {0, 12}) is False


class TestParseDay:
    def test_extracts_day(self):
        assert (
            parse_day("sub-ADNI002S2043_ses-d0381_task-rest_..._bold_reoriented.nii.gz") == 381
        )

    def test_extracts_zero_day(self):
        assert (
            parse_day("sub-ADNI002S2043_ses-d0000_task-rest_..._bold_reoriented.nii.gz") == 0
        )

    def test_returns_none_when_absent(self):
        assert parse_day("sub-X_ses-01_M12_task-rest_..._bold_reoriented.nii.gz") is None


class TestParseAdniProtocolMonth:
    def test_baseline_viscode(self):
        assert parse_adni_protocol_month("bl") == 0

    def test_month_viscode(self):
        assert parse_adni_protocol_month("m12") == 12
        assert parse_adni_protocol_month("M48") == 48

    def test_unscheduled_visit_code_returns_none(self):
        assert parse_adni_protocol_month("v01") is None

    def test_screening_code_returns_none(self):
        assert parse_adni_protocol_month("scmri") is None


class TestVisitIdentity:
    def test_delcode_delta_t_equals_protocol_month(self):
        visit_index, delta_t = visit_identity("delcode", [0, 12, 24])
        assert visit_index == [0, 1, 2]
        assert delta_t == [0.0, 12.0, 24.0]

    def test_adni_delta_t_from_elapsed_days(self):
        visit_index, delta_t = visit_identity("adni", [0, 381])
        assert visit_index == [0, 1]
        assert delta_t[0] == 0.0
        assert delta_t[1] == pytest.approx(381 / 30.44)

    def test_oasis3_uses_same_day_based_formula_as_adni(self):
        _, delta_t_oasis3 = visit_identity("oasis3", [0, 653])
        _, delta_t_adni = visit_identity("adni", [0, 653])
        assert delta_t_oasis3 == delta_t_adni

    def test_unsorted_raw_values_raises(self):
        with pytest.raises(ValueError):
            visit_identity("delcode", [12, 0])

    def test_unknown_cohort_raises(self):
        with pytest.raises(ValueError):
            visit_identity("unknown", [0])


class TestNonZeroVisitsPerCohort:
    """Regression test for the silent-drop bug in comparison-plan-v2.md §1.4:
    ADNI/OASIS-3 filenames encode elapsed days as ``ses-d<n>``, not the
    DELCODE-style ``_M<n>_`` token, so ``parse_month`` alone returns None for
    every ADNI/OASIS-3 scan and a loader built only on it silently discards
    the whole cohort (zero visits, no error). ``parse_day`` is the fix."""

    DELCODE_FILE = "sub-011d501d1_ses-01_M0_task-rest_..._bold_reoriented.nii.gz"
    ADNI_FILE = "sub-ADNI002S2043_ses-d0381_task-rest_..._bold_reoriented.nii.gz"
    OASIS3_FILE = "sub-OAS30002_ses-d0653_task-rest_..._bold_reoriented.nii.gz"

    def test_delcode_visit_recovered_via_parse_month(self):
        assert parse_month(self.DELCODE_FILE) == 0

    def test_adni_visit_not_recovered_via_parse_month(self):
        assert parse_month(self.ADNI_FILE) is None

    def test_adni_visit_recovered_via_parse_day(self):
        assert parse_day(self.ADNI_FILE) == 381

    def test_oasis3_visit_not_recovered_via_parse_month(self):
        assert parse_month(self.OASIS3_FILE) is None

    def test_oasis3_visit_recovered_via_parse_day(self):
        assert parse_day(self.OASIS3_FILE) == 653
