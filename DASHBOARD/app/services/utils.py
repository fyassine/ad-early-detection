import os
import numpy as np
from ..config import DATA_ROOT


def _safe_round_matrix(m: np.ndarray, decimals: int = 4) -> list:
    """JSON-safe nested list with NaN/inf clipped to 0."""
    arr = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.round(arr.astype(np.float64), decimals)
    return arr.tolist()


def _safe_under_root(abs_path: str) -> bool:
    """Guard against directory traversal — abs_path must live under DATA_ROOT."""
    try:
        root = os.path.realpath(DATA_ROOT)
        target = os.path.realpath(abs_path)
        return target == root or target.startswith(root + os.sep)
    except Exception:
        return False
