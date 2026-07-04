import math

import pandas as pd
import pytest

from CLASSIFIER.common.visits import (
    allowed_months_map,
    month_allowed,
    parse_allowed_months,
    parse_month,
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
