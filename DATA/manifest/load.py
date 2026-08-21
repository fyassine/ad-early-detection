"""Read a built ``cohort_manifest.csv`` — the one place manifest consumers go
instead of re-globbing the flat BOLD/FC directories (see A.0's rationale in
``schema.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from DATA.manifest.schema import MANIFEST_COLUMNS

# FC-matrix filename suffix shared by every cohort's Schaefer-200 z-transformed
# output (``process_using_schaeffer_atlas.py``'s ``OUTPUT_Z_SUFFIX``).
FC_Z_SUFFIX = "_whole_brain_correlation_matrix_z_transformed.npz"


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: manifest missing column(s) {missing}; expected {MANIFEST_COLUMNS}.")
    return df


def bold_paths(df: pd.DataFrame) -> list[Path]:
    """Every ``bold_path`` in the manifest, in row order.

    Raises rather than skipping a null/missing path — a manifest that passed
    A.0's ``assert_paths_exist_and_nonempty`` should never contain one.
    """
    missing_rows = df[df["bold_path"].isna()]
    if not missing_rows.empty:
        raise ValueError(
            f"{len(missing_rows)} manifest row(s) have a null bold_path "
            f"(subjects: {sorted(missing_rows['subject_id'].unique())[:20]}). "
            "Re-run the manifest builder — this should have failed A.0 validation."
        )
    paths = [Path(str(p)) for p in df["bold_path"]]
    bad = [p for p in paths if not p.exists() or p.stat().st_size == 0]
    if bad:
        raise ValueError(f"{len(bad)} manifest bold_path(s) missing or empty on disk: {bad[:20]}")
    return paths


def fc_path_for(bold_path: Path, fc_root: Path) -> Path:
    """The FC-matrix path a given BOLD file's extraction is expected to produce."""
    stem = bold_path.name.removesuffix(".nii.gz")
    return fc_root / f"{stem}{FC_Z_SUFFIX}"
