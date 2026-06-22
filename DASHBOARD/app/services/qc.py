import hashlib
import os
from pathlib import Path

import numpy as np

from ..config import DATA_ROOT

_QC_CACHE_DIR = Path(os.environ.get(
    "DASHBOARD_CACHE_DIR",
    os.path.join(DATA_ROOT, "_dashboard_cache"),
)) / "qc_std"


def _qc_std_path(src_abs: str) -> Path:
    h = hashlib.sha1(src_abs.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(src_abs)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return _QC_CACHE_DIR / f"{base}_{h}_std.nii.gz"


def _ensure_qc_reduce(src_abs: str) -> str:
    """
    Return a cached 3D temporal-std image for QC display.

    Uses temporal standard deviation instead of temporal mean because
    connectivity-analysis pipelines typically residualise the fMRI timeseries
    (removing confounds and the mean signal), leaving a temporal mean ≈ 0 in
    every voxel. The temporal std is always positive where BOLD signal exists,
    giving clear brain-background contrast regardless of preprocessing.

    For already-3D files the source is returned unchanged.
    Cache is keyed on the absolute source path; stale _mean caches are ignored.
    """
    cached = _qc_std_path(src_abs)
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
    result = data.std(axis=-1).astype(np.float32, copy=False)

    out = nib.Nifti1Image(result, img.affine, img.header)
    out.header.set_data_dtype(np.float32)
    # Embed valid cal_min/cal_max so NiiVue gets correct windowing from the header
    finite_pos = result[np.isfinite(result) & (result > 0)]
    if finite_pos.size > 0:
        out.header['cal_min'] = float(np.percentile(finite_pos, 2))
        out.header['cal_max'] = float(np.percentile(finite_pos, 98))

    cached.parent.mkdir(parents=True, exist_ok=True)
    import gzip as _gzip
    raw_bytes = out.to_bytes()
    with _gzip.open(str(cached), "wb", compresslevel=9) as f:
        f.write(raw_bytes)
    return str(cached)


# Keep old name as alias so any third-party code importing it doesn't break
_ensure_qc_mean = _ensure_qc_reduce
