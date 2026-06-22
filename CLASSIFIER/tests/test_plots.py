"""Smoke tests for CLASSIFIER.common.plots (headless Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from CLASSIFIER.common.plots import plot_conversion_trajectories, plot_oof_test_roc


def test_plot_oof_test_roc_returns_figure():
    oof_t = np.array([0, 0, 1, 1])
    oof_p = np.array([0.2, 0.4, 0.6, 0.8])
    te_t = np.array([0, 1, 0, 1])
    te_p = np.array([0.3, 0.7, 0.4, 0.9])
    fig = plot_oof_test_roc(oof_t, oof_p, te_t, te_p, title="unit test")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_conversion_trajectories_returns_figure():
    df = pd.DataFrame(
        {
            "pid": ["a", "a", "b", "b"],
            "label": [1, 1, 0, 0],
            "month": [0, 6, 0, 12],
            "prob": [0.4, 0.7, 0.3, 0.2],
        }
    )
    fig = plot_conversion_trajectories(df, threshold=0.5, title="unit test")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
