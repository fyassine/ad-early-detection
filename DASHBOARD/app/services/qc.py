import os
import hashlib
import numpy as np
from pathlib import Path
from ..config import DATA_ROOT

_QC_MEAN_DIR = Path(os.environ.get(
    "DASHBOARD_CACHE_DIR",
    os.path.join(DATA_ROOT, "_dashboard_cache"),
)) / "qc_mean"


def _qc_mean_path(src_abs: str) -> Path:
    h = hashlib.sha1(src_abs.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(src_abs)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return _QC_MEAN_DIR / f"{base}_{h}_mean.nii.gz"


def _ensure_qc_mean(src_abs: str) -> str:
    """
    Return the path to a cached 3D mean image for src_abs. If the source
    is already 3D, returns it unchanged. If it's 4D, computes
    data.mean(axis=3), writes to disk, and returns the cached path.
    Subsequent calls reuse the cache.
    """
    cached = _qc_mean_path(src_abs)
    if cached.exists():
        return str(cached)

    try:
        import nibabel as nib
    except ImportError:
        return src_abs

    img = nib.load(src_abs)
    if img.ndim < 4 or img.shape[-1] <= 1:
        return src_abs

    data = np.asarray(img.dataobj).astype(np.float32, copy=False)
    mean = data.mean(axis=-1).astype(np.float32, copy=False)
    out = nib.Nifti1Image(mean, img.affine, img.header)
    out.header.set_data_dtype(np.float32)
    cached.parent.mkdir(parents=True, exist_ok=True)
    # gzip level 9 — written exactly once, saves ~30% over default
    import gzip as _gzip
    raw_bytes = out.to_bytes()
    with _gzip.open(str(cached), "wb", compresslevel=9) as f:
        f.write(raw_bytes)
    return str(cached)
