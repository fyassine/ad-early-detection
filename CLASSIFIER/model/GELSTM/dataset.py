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
import re
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

from model.GAAE.utils import knn_binary_adjacency_matrix_no_diag

# Maximum visit interval for Δt normalisation (months); covers up to M108.
MAX_INTERVAL_MONTHS: float = 108.0


class LongitudinalSubjectDataset(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    matrices_dir : str
        Directory containing per-visit .npz FC matrix files.
        Filename pattern: sub-{Pseudonym}_ses-XX_{visit}_..._z_transformed.npz
    subject_df : pd.DataFrame
        Must contain columns: Pseudonym, diagnosis, sex, age.
        Each row is one subject (not one visit).
        Only rows with diagnosis in {'mci', 'converter'} are used.
    cohorts_csv : str
        Path to cohorts.csv; used to obtain per-subject visit months from the
        'visit' column (e.g. 'M0', 'M12', 'M24').
    adjacency_k : int
        k for kNN adjacency construction.
    file_variant : str
        'z_transformed' | 'raw'
    max_visits : int | None
        If set, keep only the first `max_visits` (earliest) visits per subject.
        Δt is re-normalised over the kept window so the model never sees future
        scans. Default None → use all available visits (legacy behaviour).
    require_full_window : bool
        Only meaningful with max_visits != None. If True, subjects with fewer
        than `max_visits` scans are dropped entirely so every retained subject
        has exactly `max_visits` visits — this neutralises "longer sequence =
        more likely converter" leakage. Default False.
    """

    _VARIANT_SUFFIX: Dict[str, str] = {
        "z_transformed": "_whole_brain_correlation_matrix_z_transformed.npz",
        "raw":           "_whole_brain_correlation_matrix.npz",
    }

    def __init__(
        self,
        matrices_dir: str,
        subject_df: pd.DataFrame,
        cohorts_csv: str,
        adjacency_k: int = 8,
        file_variant: str = "z_transformed",
        max_visits: Optional[int] = None,
        require_full_window: bool = False,
    ):
        self.matrices_dir       = matrices_dir
        self.adjacency_k        = adjacency_k
        self.file_variant       = file_variant
        self.max_visits         = max_visits
        self.require_full_window = require_full_window
        self.suffix             = self._VARIANT_SUFFIX.get(
            file_variant, self._VARIANT_SUFFIX["z_transformed"]
        )

        if require_full_window and max_visits is None:
            raise ValueError("require_full_window=True requires max_visits to be set")

        allowed = {"mci", "converter"}
        sub_df  = subject_df[subject_df["diagnosis"].isin(allowed)].copy()
        sub_df["Pseudonym"] = sub_df["Pseudonym"].astype(str)

        cohorts = pd.read_csv(cohorts_csv)
        id_col  = "Pseudonym"
        cohorts[id_col]    = cohorts[id_col].astype(str)
        cohorts["visit_m"] = cohorts["visit"].str.replace("M", "", regex=False).astype(float)

        self.subjects: List[Dict] = []
        n_dropped_full_window = 0
        for _, row in sub_df.iterrows():
            pid   = str(row["Pseudonym"])
            label = 1 if row["diagnosis"] == "converter" else 0
            sex   = 1 if str(row.get("sex", "f")).lower() == "m" else 0
            age_raw = row.get("age", 50.0)
            age   = float(min(max(float(age_raw) / 100.0, 0.0), 1.0))

            visit_files = self._find_visit_files(pid)
            if not visit_files:
                continue

            # Truncate to the first N (earliest) visits BEFORE computing Δt.
            if max_visits is not None:
                if require_full_window and len(visit_files) < max_visits:
                    n_dropped_full_window += 1
                    continue
                visit_files = visit_files[:max_visits]

            months  = [m for m, _ in visit_files]
            fpaths  = [f for _, f in visit_files]

            deltas  = [0.0]
            for i in range(1, len(months)):
                deltas.append((months[i] - months[i - 1]) / MAX_INTERVAL_MONTHS)

            self.subjects.append({
                "subject_id":   pid,
                "label":        label,
                "visit_months": months,
                "delta_t":      deltas,
                "file_paths":   fpaths,
                "sex":          sex,
                "age":          age,
                "n_scans":      len(months),
            })

        n_pos = sum(s["label"] for s in self.subjects)
        n_neg = len(self.subjects) - n_pos
        print(
            f"LongitudinalSubjectDataset[v2]: {len(self.subjects)} subjects "
            f"({n_pos} converter, {n_neg} stable MCI)"
        )
        if max_visits is not None:
            print(
                f"  Window: first {max_visits} visit(s); "
                f"require_full_window={require_full_window}; "
                f"dropped (insufficient visits)={n_dropped_full_window}"
            )
        if self.subjects:
            ns = [s["n_scans"] for s in self.subjects]
            print(f"  Scans per subject: min={min(ns)}  max={max(ns)}  mean={np.mean(ns):.1f}")

    def _find_visit_files(self, pid: str) -> List[tuple]:
        pattern = os.path.join(self.matrices_dir, f"sub-{pid}_*{self.suffix}")
        files   = glob.glob(pattern)
        result  = []
        for f in files:
            m = re.search(r"_(M\d+)_", os.path.basename(f))
            if m:
                month = int(m.group(1).replace("M", ""))
                result.append((month, f))
        return sorted(result, key=lambda x: x[0])

    def _load_graph(self, filepath: str) -> Data:
        arr = np.load(filepath)["array"]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat = torch.tensor(arr, dtype=torch.float)
        adj  = knn_binary_adjacency_matrix_no_diag(torch.abs(feat), k=self.adjacency_k)
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adj, dtype=torch.float32)
        ei, ew = dense_to_sparse(adj)
        return Data(x=feat, edge_index=ei, edge_attr=ew)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict:
        sub  = self.subjects[idx]
        graphs = [self._load_graph(fp) for fp in sub["file_paths"]]
        return {
            "subject_id":   sub["subject_id"],
            "label":        sub["label"],
            "visit_months": sub["visit_months"],
            "delta_t":      sub["delta_t"],
            "graphs":       graphs,
            "sex":          sub["sex"],
            "age":          sub["age"],
            "n_scans":      sub["n_scans"],
        }

    def get_labels(self) -> List[int]:
        return [s["label"] for s in self.subjects]

    def get_subject_ids(self) -> List[str]:
        return [s["subject_id"] for s in self.subjects]

    def get_n_scans(self) -> List[int]:
        return [s["n_scans"] for s in self.subjects]
