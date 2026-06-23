"""
longitudinal.py — At-risk window utilities and longitudinal feature aggregation.

Core concept: every subject has an "at-risk window" [0, window_end_months) during
which they are at risk of MCI→AD conversion. The window endpoint is the same
computation for both groups — the critical difference is only that converters
have event=1 at window_end whereas non-converters are censored (event=0).

All longitudinal feature extraction is restricted to visits within this window,
ensuring symmetric handling of both classes.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def visit_months(visit) -> int | None:
    """Parse 'M12' → 12. Returns None if unrecognised."""
    if visit is None or (isinstance(visit, float) and math.isnan(visit)):
        return None
    m = re.match(r"M(\d+)", str(visit).strip().upper())
    return int(m.group(1)) if m else None


def compute_at_risk_window(
    grp_sorted: pd.DataFrame,
    diagnosis_col: str = "_diagnosis_norm",
    months_col: str = "_months",
) -> tuple[int, int, int]:
    """
    Compute (window_start, window_end, event) for one subject's sorted visit rows.

    window_start : always 0 (baseline month)
    window_end   : months to first AD diagnosis (converter) OR months of last visit (non-converter)
    event        : 1 if AD diagnosis seen, 0 otherwise (right-censored)

    The same computation applies regardless of whether the subject is a converter
    or non-converter — only the event flag and window_end value differ.
    """
    diags = grp_sorted[diagnosis_col].astype(str).str.lower().str.strip()
    months = grp_sorted[months_col]

    ad_mask = diags == "ad"
    if ad_mask.any():
        ad_months = months[ad_mask].dropna()
        window_end = int(ad_months.min()) if not ad_months.empty else 0
        event = 1
    else:
        valid_months = months.dropna()
        window_end = int(valid_months.max()) if not valid_months.empty else 0
        event = 0

    return 0, window_end, event


class LongitudinalAggregator:
    """
    Per-subject longitudinal feature aggregation within the at-risk window.

    Usage:
        agg = LongitudinalAggregator(cohorts_df, id_col='Pseudonym')
        slope = agg.slope('subject_123', 'mmstot', window_end=48)
        last_val = agg.last('subject_123', 'mmstot', window_end=48)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str = "Pseudonym",
        visit_col: str = "visit",
        diagnosis_col: str = "diagnosis",
    ):
        self.id_col = id_col
        self.visit_col = visit_col
        self.diagnosis_col = diagnosis_col

        df = df.copy()
        df["_months"] = df[visit_col].apply(visit_months)
        df["_diagnosis_norm"] = df[diagnosis_col].astype(str).str.lower().str.strip()
        self._df = df
        self._by_subject: dict[str, pd.DataFrame] = {}

    def _get_subject(self, subject_id: str) -> pd.DataFrame:
        if subject_id not in self._by_subject:
            sub = self._df[self._df[self.id_col].astype(str) == str(subject_id)].copy()
            sub = sub.sort_values("_months", na_position="last").reset_index(drop=True)
            self._by_subject[subject_id] = sub
        return self._by_subject[subject_id]

    def windowed(self, subject_id: str, window_end: int) -> pd.DataFrame:
        """Rows for subject with _months < window_end (exclusive of event visit)."""
        grp = self._get_subject(subject_id)
        mask = grp["_months"].notna() & (grp["_months"] < window_end)
        return grp[mask].reset_index(drop=True)

    def _coerce(self, v) -> float | None:
        try:
            f = float(v)
            return None if (math.isnan(f) or not math.isfinite(f)) else f
        except (TypeError, ValueError):
            return None

    def _valid_series(self, subject_id: str, col: str, window_end: int) -> pd.Series:
        """Float-coerced values within the window, dropping NaN."""
        win = self.windowed(subject_id, window_end)
        if col not in win.columns:
            return pd.Series([], dtype=float)
        vals = win[col].apply(self._coerce)
        months = win["_months"].apply(lambda x: float(x) if x is not None else float("nan"))
        valid = vals.notna() & months.notna()
        return pd.Series(vals[valid].values, index=months[valid].values, dtype=float)

    def baseline(self, subject_id: str, col: str, window_end: int | None = None) -> float | None:
        """Value at M0 (earliest valid visit)."""
        series = self._valid_series(subject_id, col, window_end or 9999)
        return float(series.iloc[0]) if not series.empty else None

    def last(self, subject_id: str, col: str, window_end: int) -> float | None:
        """Value at the latest visit within the window."""
        series = self._valid_series(subject_id, col, window_end)
        return float(series.iloc[-1]) if not series.empty else None

    def mean(self, subject_id: str, col: str, window_end: int) -> float | None:
        series = self._valid_series(subject_id, col, window_end)
        return float(series.mean()) if not series.empty else None

    def delta(self, subject_id: str, col: str, window_end: int) -> float | None:
        """last - baseline."""
        series = self._valid_series(subject_id, col, window_end)
        if len(series) < 2:
            return None
        return float(series.iloc[-1] - series.iloc[0])

    def slope(self, subject_id: str, col: str, window_end: int) -> float | None:
        """Linear slope in units per YEAR (OLS, requires ≥2 data points)."""
        series = self._valid_series(subject_id, col, window_end)
        if len(series) < 2:
            return None
        x = series.index.to_numpy(dtype=float)  # months
        y = series.to_numpy(dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx < 1e-9:
            return None
        b = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
        return b * 12.0  # months → years

    def n_visits(self, subject_id: str, window_end: int) -> int:
        """Number of valid visits within the window."""
        return len(self.windowed(subject_id, window_end))


def to_long_format(
    survival_table: pd.DataFrame,
    cohorts_df: pd.DataFrame,
    feature_cols: list[str],
    id_col: str = "Pseudonym",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """
    Build a long-format (start_months, stop_months, event, *features) DataFrame
    for time-varying Cox / LSTM models.

    Each row represents one inter-visit interval for a subject. Features carry
    the value from the START of that interval.

    Returns DataFrame with columns:
        subject_id, start_months, stop_months, event, n_visits_in_window,
        <feature_cols>
    """
    cohorts_df = cohorts_df.copy()
    id_col_actual = next((c for c in (id_col, "Pseudonym", "subject_id")
                          if c in cohorts_df.columns), None)
    if id_col_actual is None:
        raise ValueError(f"No subject ID column found in cohorts_df: {list(cohorts_df.columns)}")

    cohorts_df[id_col_actual] = cohorts_df[id_col_actual].astype(str)
    cohorts_df["_months"] = cohorts_df[visit_col].apply(visit_months)
    cohorts_df["_diagnosis_norm"] = cohorts_df["diagnosis"].astype(str).str.lower().str.strip()

    LongitudinalAggregator(
        cohorts_df, id_col=id_col_actual, visit_col=visit_col, diagnosis_col="diagnosis"
    )

    rows = []
    for _, srow in survival_table.iterrows():
        sid = str(srow["subject_id"])
        window_end = int(srow["duration"])
        event = int(srow["event_observed"])

        grp = cohorts_df[cohorts_df[id_col_actual] == sid].copy()
        grp = grp.sort_values("_months", na_position="last").reset_index(drop=True)
        win_rows = grp[grp["_months"].notna() & (grp["_months"] < window_end)].reset_index(drop=True)

        if len(win_rows) == 0:
            # Fall back to baseline row as a single interval
            win_rows = grp.iloc[[0]].copy() if len(grp) > 0 else pd.DataFrame()

        if len(win_rows) == 0:
            continue

        visit_months_list = win_rows["_months"].tolist()

        for i, (_, row_visit) in enumerate(win_rows.iterrows()):
            start = int(row_visit["_months"])
            stop = int(visit_months_list[i + 1]) if i + 1 < len(visit_months_list) else window_end
            if stop <= start:
                stop = start + 1  # ensure positive duration

            # This interval is the last one and had an event
            interval_event = 1 if (i == len(win_rows) - 1 and event == 1) else 0

            rec: dict = {
                "subject_id": sid,
                "start_months": start,
                "stop_months": stop,
                "event": interval_event,
            }
            for col in feature_cols:
                if col in win_rows.columns:
                    val = row_visit.get(col, None)
                    try:
                        val = float(val)
                        if not math.isfinite(val):
                            val = None
                    except (TypeError, ValueError):
                        val = None
                    rec[col] = val
            rows.append(rec)

    return pd.DataFrame(rows)


def to_sequence_tensors(
    survival_table: pd.DataFrame,
    cohorts_df: pd.DataFrame,
    feature_cols: list[str],
    max_len: int = 10,
    id_col: str = "Pseudonym",
    visit_col: str = "visit",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build padded sequence tensors for LSTM training.

    Returns:
        sequences : (n_subjects, max_len, n_features) float32
        lengths   : (n_subjects,) int — actual sequence length per subject
        T         : (n_subjects,) float — event/censoring time in months
        E         : (n_subjects,) int — event indicator (0/1)
        subject_ids : list of subject IDs in the same order
    """
    cohorts_df = cohorts_df.copy()
    id_col_actual = next((c for c in (id_col, "Pseudonym", "subject_id")
                          if c in cohorts_df.columns), None)
    if id_col_actual is None:
        raise ValueError(f"No subject ID column found in cohorts_df: {list(cohorts_df.columns)}")

    cohorts_df[id_col_actual] = cohorts_df[id_col_actual].astype(str)
    cohorts_df["_months"] = cohorts_df[visit_col].apply(visit_months)
    n_feat = len(feature_cols)

    all_seqs, all_lengths, all_T, all_E, all_ids = [], [], [], [], []

    for _, srow in survival_table.iterrows():
        sid = str(srow["subject_id"])
        window_end = float(srow["duration"])
        event = int(srow["event_observed"])

        grp = cohorts_df[cohorts_df[id_col_actual] == sid].copy()
        grp = grp.sort_values("_months", na_position="last").reset_index(drop=True)
        win = grp[grp["_months"].notna() & (grp["_months"] < window_end)].reset_index(drop=True)

        if len(win) == 0:
            win = grp.iloc[[0]] if len(grp) > 0 else None
        if win is None or len(win) == 0:
            continue

        seq = np.zeros((max_len, n_feat), dtype=np.float32)
        length = min(len(win), max_len)
        for t, row in enumerate(win.iloc[:length].itertuples()):
            for fi, col in enumerate(feature_cols):
                val = getattr(row, col, None)
                try:
                    v = float(val)
                    seq[t, fi] = v if math.isfinite(v) else 0.0
                except (TypeError, ValueError):
                    seq[t, fi] = 0.0

        all_seqs.append(seq)
        all_lengths.append(length)
        all_T.append(window_end)
        all_E.append(event)
        all_ids.append(sid)

    sequences = np.stack(all_seqs, axis=0) if all_seqs else np.zeros((0, max_len, n_feat), dtype=np.float32)
    lengths = np.array(all_lengths, dtype=np.int64)
    T = np.array(all_T, dtype=np.float64)
    E = np.array(all_E, dtype=np.int64)
    return sequences, lengths, T, E, all_ids
