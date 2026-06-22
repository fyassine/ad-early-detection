"""Tests for PROGNOSER/common/metrics.py.

Focus: the time-window guard in time_dependent_auc / integrated_brier_score uses
inclusive endpoints (>=/<=) so a horizon equal to T_test.min() or T_test.max() is
evaluated rather than silently dropped to NaN.
"""
import numpy as np

from PROGNOSER.common.metrics import integrated_brier_score, time_dependent_auc


def _toy_survival():
    # Two groups separated in time so AUC is well-defined; events spread across
    # the window. Endpoints (min, max) of T_test land exactly on eval horizons.
    rng = np.random.default_rng(0)
    T_train = np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 18.0, 30.0, 42.0, 54.0])
    E_train = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    T_test = np.array([24.0, 36.0, 48.0, 60.0])  # min=24, max=60
    E_test = np.array([1, 1, 0, 1])
    risk = rng.normal(size=len(T_test))
    return T_train, E_train, T_test, E_test, risk


def test_time_dependent_auc_includes_min_endpoint():
    """A horizon equal to T_test.min() must be evaluated (inclusive guard),
    not left at its NaN initializer (regression guard for the strict-`>` bug)."""
    T_train, E_train, T_test, E_test, risk = _toy_survival()
    out = time_dependent_auc(T_train, E_train, T_test, E_test, risk, times=(24, 36, 60))
    # 24 == T_test.min(): with the old strict `>` guard this stayed NaN.
    assert not np.isnan(out[24]), "horizon at T_test.min() should be evaluated"
    assert not np.isnan(out[36])


def test_time_dependent_auc_excludes_out_of_range():
    """Horizons strictly outside [min, max] stay NaN."""
    T_train, E_train, T_test, E_test, risk = _toy_survival()
    out = time_dependent_auc(T_train, E_train, T_test, E_test, risk, times=(12, 84))
    assert np.isnan(out[12]) and np.isnan(out[84])


def test_integrated_brier_score_includes_endpoint():
    """IBS guard is inclusive: an endpoint horizon yields a finite score."""
    T_train, E_train, T_test, E_test, _ = _toy_survival()
    eval_times = np.array([24.0, 36.0, 48.0])  # 24 == T_test.min()
    # Constant survival curve (KM-style): same row tiled across test subjects.
    surv = np.tile(np.array([0.9, 0.7, 0.5]), (len(T_test), 1))
    ibs = integrated_brier_score(T_train, E_train, T_test, E_test, surv, eval_times)
    assert np.isfinite(ibs)
