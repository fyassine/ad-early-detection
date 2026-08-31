"""
CLASSIFIER GELSTM/dataset.py — LongitudinalSubjectDataset.

v2 additions (vs CLASSIFIER/model/GELSTM/dataset.py):
    * max_visits           — truncate each subject to its first N visits.
    * require_full_window  — drop subjects with fewer than max_visits scans
                              (enforces equal sequence length across subjects).

Each item is one subject's longitudinal sequence:
    {
        'subject_id':    str,
        'label':         int,          # 1=converter, 0=stable_mci
        'visit_months':  list[int],    # sorted, ascending
        'delta_t':       list[float],  # normalised inter-visit intervals; 0.0 for first visit
        'graphs':        list[Data],   # PyG Data per visit, sorted by month
        'sex':           int,          # 0=female, 1=male
        'age':           float,        # normalised age [0,1]
    }
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.GAAE.utils import knn_binary_adjacency_matrix_no_diag  # noqa: E402

from CLASSIFIER.common.visits import (  # noqa: E402
    Cohort,
    parse_allowed_months,
    parse_day,
    parse_month,
    visit_identity,
)

# Maximum visit interval for Δt normalisation (months); covers up to M108.
MAX_INTERVAL_MONTHS: float = 108.0


class LongitudinalSubjectDataset(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    matrices_dir : str
        Directory containing per-visit .npz FC matrix files.
        Filename pattern: sub-{subject_id}_..._z_transformed.npz
    subject_df : pd.DataFrame
        Must contain columns: Pseudonym/subject_id, diagnosis/label/converter_status, sex, age.
        Each row is one subject (not one visit).
    cohorts_csv : str | None
        Unused. Accepted only so existing call sites (adapters, checkpointed
        notebooks) don't need to change their signature; visit allow-lists are
        read from `allowed_days`/`allowed_months` columns already present in
        `subject_df`, never from this file.
    adjacency_k : int
        k for kNN adjacency construction.
    file_variant : str
        'z_transformed' | 'raw'
    min_visits : int | None
        If set, drop subjects whose total (untruncated) visit count is below
        this floor, evaluated BEFORE any `max_visits` truncation — mirrors
        BrainTokenGT's cohort window (`BrainTokenGTAdapter.prepare_data`:
        `it["n_scans"] >= min_visits`, then `window_item(..., max_visits=...)`).
        Default None → no floor (legacy behaviour).
    max_visits : int | None
        If set, keep only the first `max_visits` (earliest) visits per subject.
        Δt is re-normalised over the kept window so the model never sees future
        scans. Default None → use all available visits (legacy behaviour).
    require_full_window : bool
        Only meaningful with max_visits != None. If True, subjects with fewer
        than `max_visits` scans are dropped entirely so every retained subject
        has exactly `max_visits` visits — this neutralises "longer sequence =
        more likely converter" leakage. Default False. Independent of
        `min_visits`: a subject with exactly `min_visits` visits and
        `min_visits < max_visits` is kept unpadded unless this is also set.
    cohort : str
        'delcode' | 'adni' | 'oasis3'. Governs whether filenames encode
        nominal protocol months (_M<n>_) or elapsed days from baseline (ses-d<n>).
    """

    _VARIANT_SUFFIX: Dict[str, str] = {
        "z_transformed": "_whole_brain_correlation_matrix_z_transformed.npz",
        "raw": "_whole_brain_correlation_matrix.npz",
    }

    def __init__(
        self,
        matrices_dir: str,
        subject_df: pd.DataFrame,
        cohorts_csv: Optional[str] = None,
        adjacency_k: int = 8,
        file_variant: str = "z_transformed",
        min_visits: Optional[int] = None,
        max_visits: Optional[int] = None,
        require_full_window: bool = False,
        cohort: Cohort = "delcode",
    ):
        self.matrices_dir = matrices_dir
        self.adjacency_k = adjacency_k
        self.file_variant = file_variant
        self.min_visits = min_visits
        self.max_visits = max_visits
        self.require_full_window = require_full_window
        self.cohort: Cohort = str(cohort).lower()  # type: ignore[assignment]
        self.suffix = self._VARIANT_SUFFIX.get(file_variant, self._VARIANT_SUFFIX["z_transformed"])

        if require_full_window and max_visits is None:
            raise ValueError("require_full_window=True requires max_visits to be set")

        id_col = "subject_id" if "subject_id" in subject_df.columns else "Pseudonym"
        if id_col not in subject_df.columns:
            raise ValueError(
                f"subject_df must contain 'subject_id' or 'Pseudonym'; got {list(subject_df.columns)}"
            )

        sub_df = subject_df.copy()
        if "diagnosis" in sub_df.columns:
            allowed = {"mci", "converter"}
            sub_df = sub_df[sub_df["diagnosis"].isin(allowed)].copy()
        elif "label" in sub_df.columns:
            allowed = {"mci", "stable", "converter"}
            sub_df = sub_df[sub_df["label"].isin(allowed)].copy()
        sub_df[id_col] = sub_df[id_col].astype(str)

        self.subjects: List[Dict] = []
        n_dropped_min_visits = 0
        n_dropped_full_window = 0

        allow_col = None
        for cand in ("allowed_days", "allowed_months"):
            if cand in sub_df.columns:
                allow_col = cand
                break

        for _, row in sub_df.iterrows():
            pid = str(row[id_col])
            if "converter_status" in row and pd.notna(row["converter_status"]):
                label = int(row["converter_status"])
            elif "label" in row and pd.notna(row["label"]):
                label = 1 if str(row["label"]).lower() == "converter" else 0
            elif "diagnosis" in row and pd.notna(row["diagnosis"]):
                label = 1 if str(row["diagnosis"]).lower() == "converter" else 0
            else:
                raise ValueError(
                    f"Subject {pid!r} has no usable label: 'converter_status', "
                    "'label' and 'diagnosis' are all absent or NaN "
                    f"(columns present: {list(row.index)}). Silently defaulting to "
                    "non-converter would corrupt the CV split without a trace."
                )

            sex = 1 if str(row.get("sex", "f")).lower() in ("m", "1", "true") else 0
            age_raw = row.get("age", 50.0)
            try:
                age_f = float(age_raw)
            except (ValueError, TypeError):
                age_f = 50.0
            age = float(min(max(age_f / 100.0, 0.0), 1.0))

            allowed_visits = (
                parse_allowed_months(row[allow_col]) if allow_col is not None else None
            )
            visit_files = self._find_visit_files(pid, allowed_visits)
            if not visit_files:
                continue

            # min_visits is a floor on the FULL (untruncated) visit count —
            # evaluated before any truncation, matching BrainTokenGT's
            # `it["n_scans"] >= min_visits` keep-rule.
            if min_visits is not None and len(visit_files) < min_visits:
                n_dropped_min_visits += 1
                continue

            # Truncate to the first N (earliest) visits BEFORE computing Δt.
            if max_visits is not None:
                if require_full_window and len(visit_files) < max_visits:
                    n_dropped_full_window += 1
                    continue
                visit_files = visit_files[:max_visits]

            raw_vals = [v for v, _ in visit_files]
            fpaths = [f for _, f in visit_files]

            _, cum_months = visit_identity(self.cohort, raw_vals)
            deltas = [0.0]
            for i in range(1, len(cum_months)):
                deltas.append((cum_months[i] - cum_months[i - 1]) / MAX_INTERVAL_MONTHS)

            self.subjects.append(
                {
                    "subject_id": pid,
                    "label": label,
                    "visit_months": raw_vals,
                    "delta_t": deltas,
                    "file_paths": fpaths,
                    "sex": sex,
                    "age": age,
                    "n_scans": len(raw_vals),
                }
            )

        n_pos = sum(s["label"] for s in self.subjects)
        n_neg = len(self.subjects) - n_pos
        print(
            f"LongitudinalSubjectDataset[v2][{self.cohort}]: {len(self.subjects)} subjects "
            f"({n_pos} converter, {n_neg} stable/MCI)"
        )
        if min_visits is not None:
            print(f"  min_visits={min_visits}; dropped (too few visits)={n_dropped_min_visits}")
        if max_visits is not None:
            print(
                f"  Window: first {max_visits} visit(s); "
                f"require_full_window={require_full_window}; "
                f"dropped (insufficient visits)={n_dropped_full_window}"
            )
        if self.subjects:
            ns = [s["n_scans"] for s in self.subjects]
            print(f"  Scans per subject: min={min(ns)}  max={max(ns)}  mean={np.mean(ns):.1f}")

    def _find_visit_files(self, pid: str, allowed_visits: Optional[set] = None) -> List[tuple]:
        pattern = os.path.join(self.matrices_dir, f"sub-{pid}_*{self.suffix}")
        files = glob.glob(pattern)
        result = []
        for f in files:
            fname = os.path.basename(f)
            val = parse_month(fname) if self.cohort == "delcode" else parse_day(fname)
            if val is None:
                continue
            if allowed_visits is not None and val not in allowed_visits:
                continue
            result.append((val, f))
        return sorted(result, key=lambda x: x[0])

    def _load_graph(self, filepath: str) -> Data:
        arr = np.load(filepath)["array"]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat = torch.tensor(arr, dtype=torch.float)
        adj = knn_binary_adjacency_matrix_no_diag(torch.abs(feat), k=self.adjacency_k)
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adj, dtype=torch.float32)
        ei, ew = dense_to_sparse(adj)
        return Data(x=feat, edge_index=ei, edge_attr=ew)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict:
        sub = self.subjects[idx]
        graphs = [self._load_graph(fp) for fp in sub["file_paths"]]
        return {
            "subject_id": sub["subject_id"],
            "label": sub["label"],
            "visit_months": sub["visit_months"],
            "delta_t": sub["delta_t"],
            "graphs": graphs,
            "sex": sub["sex"],
            "age": sub["age"],
            "n_scans": sub["n_scans"],
        }

    def get_labels(self) -> List[int]:
        return [s["label"] for s in self.subjects]

    def get_subject_ids(self) -> List[str]:
        return [s["subject_id"] for s in self.subjects]

    def get_n_scans(self) -> List[int]:
        return [s["n_scans"] for s in self.subjects]
