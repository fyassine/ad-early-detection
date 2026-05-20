"""
GELSTM/dataset.py — LongitudinalSubjectDataset

Each item is one subject's complete longitudinal sequence:
    {
        'subject_id':    str,
        'label':         int,          # 1=converter, 0=stable_mci
        'visit_months':  list[int],    # sorted, e.g. [0, 12, 24, 36]
        'delta_t':       list[float],  # normalised inter-visit intervals; 0.0 for first visit
        'graphs':        list[Data],   # PyG Data per visit, sorted by month
        'sex':           int,          # 0=female, 1=male
        'age':           float,        # normalised age [0,1]
    }

All available scans are used (no limit on max visits).
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

from __CLASSIFIER__.CLASSIFIER.model.GAAE.utils import knn_binary_adjacency_matrix_no_diag

# Maximum visit interval for Δt normalisation (months).
# Covers up to M108 which is the max observed visit.
MAX_INTERVAL_MONTHS: float = 108.0


class LongitudinalSubjectDataset(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    matrices_dir : str
        Directory containing per-visit .npz FC matrix files.
        Filename pattern: sub-{Repseudonym}_ses-XX_{visit}_..._z_transformed.npz
    subject_df : pd.DataFrame
        Must contain columns: Repseudonym, diagnosis, sex, age.
        Each row is one subject (not one visit).
        Only rows with diagnosis in {'mci', 'converter'} are used.
    cohorts_csv : str
        Path to cohorts.csv; used to obtain per-subject visit months from the
        'visit' column (e.g. 'M0', 'M12', 'M24').
    adjacency_k : int
        k for kNN adjacency construction.
    file_variant : str
        'z_transformed' | 'raw'
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
    ):
        self.matrices_dir  = matrices_dir
        self.adjacency_k   = adjacency_k
        self.file_variant  = file_variant
        self.suffix        = self._VARIANT_SUFFIX.get(file_variant, self._VARIANT_SUFFIX["z_transformed"])

        # Subject-level metadata
        allowed = {"mci", "converter"}
        sub_df  = subject_df[subject_df["diagnosis"].isin(allowed)].copy()
        sub_df["Repseudonym"] = sub_df["Repseudonym"].astype(str)

        # Build visit-month map from cohorts.csv
        cohorts = pd.read_csv(cohorts_csv)
        # cohorts.csv uses 'Pseudonym' as the ID column
        id_col  = "Pseudonym" if "Pseudonym" in cohorts.columns else "Repseudonym"
        cohorts[id_col]    = cohorts[id_col].astype(str)
        cohorts["visit_m"] = cohorts["visit"].str.replace("M", "", regex=False).astype(float)
        visit_map: Dict[str, List[int]] = (
            cohorts.groupby(id_col)["visit_m"]
            .apply(lambda s: sorted(int(v) for v in s.dropna()))
            .to_dict()
        )

        self.subjects: List[Dict] = []
        for _, row in sub_df.iterrows():
            pid   = str(row["Repseudonym"])
            label = 1 if row["diagnosis"] == "converter" else 0
            sex   = 1 if str(row.get("sex", "f")).lower() == "m" else 0
            age_raw = row.get("age", 50.0)
            age   = float(min(max(float(age_raw) / 100.0, 0.0), 1.0))

            # Find all matching .npz files from filesystem
            visit_files = self._find_visit_files(pid)
            if not visit_files:
                continue   # subject has no matrix files

            months  = [m for m, _ in visit_files]
            fpaths  = [f for _, f in visit_files]

            # Δt: inter-visit intervals normalised by MAX_INTERVAL_MONTHS
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

        print(
            f"LongitudinalSubjectDataset: {len(self.subjects)} subjects "
            f"({sum(s['label'] for s in self.subjects)} converter, "
            f"{sum(1-s['label'] for s in self.subjects)} stable MCI)"
        )
        ns = [s["n_scans"] for s in self.subjects]
        print(f"  Scans per subject: min={min(ns)}  max={max(ns)}  mean={np.mean(ns):.1f}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_visit_files(self, pid: str) -> List[tuple]:
        """Return sorted (visit_months, filepath) tuples from the filesystem."""
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
        """Load one .npz file → PyG Data with kNN adjacency."""
        arr = np.load(filepath)["array"]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat = torch.tensor(arr, dtype=torch.float)
        adj  = knn_binary_adjacency_matrix_no_diag(torch.abs(feat), k=self.adjacency_k)
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adj, dtype=torch.float32)
        ei, ew = dense_to_sparse(adj)
        return Data(x=feat, edge_index=ei, edge_attr=ew)

    # ── Dataset interface ─────────────────────────────────────────────────────

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
