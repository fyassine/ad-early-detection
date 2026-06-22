"""Tests for PROGNOSER/common/longitudinal.py."""
import math

import numpy as np
import pandas as pd
import pytest

from PROGNOSER.common.longitudinal import (
    visit_months,
    compute_at_risk_window,
    LongitudinalAggregator,
)


# ── visit_months ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("v,expected", [
    ("M0",   0),
    ("M12",  12),
    ("M36",  36),
    ("m0",   0),    # case-insensitive
    ("M024", 24),
])
def test_visit_months_valid(v, expected):
    assert visit_months(v) == expected


@pytest.mark.parametrize("v", ["baseline", "V1", "12", "", None, float("nan")])
def test_visit_months_invalid_returns_none(v):
    assert visit_months(v) is None


# ── compute_at_risk_window ────────────────────────────────────────────────────

def _make_grp(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["_months"] = df["visit"].apply(visit_months)
    df["_diagnosis_norm"] = df["diagnosis"].str.lower().str.strip()
    return df.sort_values("_months", na_position="last").reset_index(drop=True)


def test_at_risk_window_converter():
    grp = _make_grp([
        {"visit": "M0",  "diagnosis": "mci"},
        {"visit": "M12", "diagnosis": "mci"},
        {"visit": "M24", "diagnosis": "ad"},
    ])
    start, end, event = compute_at_risk_window(grp)
    assert start == 0
    assert end   == 24
    assert event == 1


def test_at_risk_window_non_converter():
    grp = _make_grp([
        {"visit": "M0",  "diagnosis": "mci"},
        {"visit": "M12", "diagnosis": "mci"},
        {"visit": "M36", "diagnosis": "mci"},
    ])
    start, end, event = compute_at_risk_window(grp)
    assert start == 0
    assert end   == 36
    assert event == 0


def test_at_risk_window_converter_picks_earliest_ad():
    """When multiple AD visits exist, window_end = earliest AD month."""
    grp = _make_grp([
        {"visit": "M0",  "diagnosis": "mci"},
        {"visit": "M12", "diagnosis": "ad"},
        {"visit": "M24", "diagnosis": "ad"},
    ])
    _, end, event = compute_at_risk_window(grp)
    assert end   == 12
    assert event == 1


# ── LongitudinalAggregator ────────────────────────────────────────────────────

@pytest.fixture()
def agg_df():
    rows = [
        {"Pseudonym": "s1", "visit": "M0",  "diagnosis": "mci",  "mmstot": 28.0},
        {"Pseudonym": "s1", "visit": "M12", "diagnosis": "mci",  "mmstot": 26.0},
        {"Pseudonym": "s1", "visit": "M24", "diagnosis": "mci",  "mmstot": 24.0},
        {"Pseudonym": "s1", "visit": "M36", "diagnosis": "ad",   "mmstot": 20.0},
        {"Pseudonym": "s2", "visit": "M0",  "diagnosis": "mci",  "mmstot": 25.0},
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def agg(agg_df):
    return LongitudinalAggregator(agg_df, id_col="Pseudonym", visit_col="visit", diagnosis_col="diagnosis")


def test_baseline_returns_m0_value(agg):
    assert agg.baseline("s1", "mmstot") == pytest.approx(28.0)


def test_last_within_window(agg):
    # window_end=24 → visits M0=28, M12=26 included; M24 is excluded (< 24 → no, equal)
    # window is *strictly* < window_end, so M24 is excluded
    val = agg.last("s1", "mmstot", window_end=24)
    assert val == pytest.approx(26.0)


def test_last_includes_m24_when_window_end_36(agg):
    val = agg.last("s1", "mmstot", window_end=36)
    assert val == pytest.approx(24.0)


def test_mean_within_window(agg):
    # window_end=24 → M0=28, M12=26 → mean=27
    val = agg.mean("s1", "mmstot", window_end=24)
    assert val == pytest.approx(27.0)


def test_delta_within_window(agg):
    # window_end=36 → M0=28, M12=26, M24=24 → last - baseline = 24 - 28 = -4
    val = agg.delta("s1", "mmstot", window_end=36)
    assert val == pytest.approx(-4.0)


def test_slope_sign_and_units(agg):
    # mmstot goes 28→26→24 over M0→M12→M24 (window_end=36)
    # slope should be negative (declining score)
    slope = agg.slope("s1", "mmstot", window_end=36)
    assert slope is not None
    assert slope < 0


def test_slope_units_are_per_year(agg):
    # 28→26→24 over 0→12→24 months = -2/12 per month = -2 per year
    slope = agg.slope("s1", "mmstot", window_end=36)
    assert slope == pytest.approx(-2.0, rel=1e-4)


def test_n_visits_counts_correctly(agg):
    # window_end=36 → M0(0), M12(12), M24(24) included (all < 36)
    assert agg.n_visits("s1", window_end=36) == 3


def test_single_visit_delta_returns_none(agg):
    # s2 has only M0; delta requires ≥2 points
    assert agg.delta("s2", "mmstot", window_end=24) is None


def test_single_visit_slope_returns_none(agg):
    assert agg.slope("s2", "mmstot", window_end=24) is None


def test_unknown_subject_returns_none(agg):
    assert agg.baseline("unknown_sid", "mmstot") is None


def test_windowing_excludes_event_visit(agg):
    # window_end=36 strictly excludes month 36 (the ad visit)
    w = agg.windowed("s1", window_end=36)
    assert 36 not in w["_months"].values
