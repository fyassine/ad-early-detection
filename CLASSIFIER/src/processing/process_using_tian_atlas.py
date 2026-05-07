"""
process_using_tian_atlas.py — Extract hippocampal FC matrices using the Tian
subcortical atlas (Scale II, bilateral hippocampus = 4 parcels).

The Tian atlas NIfTI must be downloaded first:
    templateflow: tpl-MNI152NLin6Asym_atlas-Tian_res-1_dseg.nii.gz
    or from: https://github.com/yetianmed/subcortex (Scale II, MNI152NLin6Asym)

Usage (from repo root):
    python -m CLASSIFIER.src.processing.process_using_tian_atlas \\
        --atlas-path /path/to/Tian_Subcortex_S2_3T.nii.gz \\
        --labels-path /path/to/Tian_Subcortex_S2_3T_label.txt

Output saved to DATA/DELCODE/__v5__/matrices/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FMRI_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "__v1__" / "fmri"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "__v5__"
SUBJECT_GLOB = "sub-*"
OUTPUT_RAW_SUFFIX = "_hippocampus_correlation_matrix.npz"
OUTPUT_Z_SUFFIX = "_hippocampus_correlation_matrix_z_transformed.npz"


def load_hippocampus_labels(labels_path: Path) -> list[str]:
    """
    Read a Tian label text file (one label per line) and return only labels
    that contain 'Hippocampus' (case-insensitive). Returns all labels if the
    file is absent — caller must verify the atlas has the right structure.
    """
    if not labels_path.exists():
        return []
    lines = labels_path.read_text().splitlines()
    return [ln.strip() for ln in lines if "hippocampus" in ln.lower()]


def build_masker(atlas_path: Path, labels_path: Path | None) -> tuple[NiftiLabelsMasker, list[int]]:
    """
    Build a NiftiLabelsMasker over the Tian atlas restricted to hippocampal
    parcels. Returns the masker and the 1-based label indices of hippocampal
    parcels within the atlas.
    """
    if labels_path is not None and labels_path.exists():
        all_labels = labels_path.read_text().splitlines()
        hippo_1based = [
            i + 1 for i, ln in enumerate(all_labels)
            if "hippocampus" in ln.lower()
        ]
    else:
        # Assume all parcels are hippocampal (single-structure atlas)
        hippo_1based = []

    masker = NiftiLabelsMasker(
        labels_img=str(atlas_path),
        labels=None,
        standardize=cast(Any, "zscore_sample"),
        resampling_target="data",
    )
    return masker, hippo_1based


def is_rest_bold_nifti(path: Path) -> bool:
    if not path.is_file():
        return False
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return False
    name = path.name.lower()
    return "task-rest" in name and "_bold" in name


def iter_bold_files(baseline_root: Path):
    for subject_dir in sorted(baseline_root.glob(SUBJECT_GLOB)):
        if not subject_dir.is_dir():
            continue
        for nii_path in sorted(subject_dir.iterdir()):
            if is_rest_bold_nifti(nii_path):
                yield nii_path


def strip_nifti_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def compute_connectivity_matrices(
    bold_path: Path,
    masker: NiftiLabelsMasker,
    hippo_indices: list[int],
    correlation_measure: ConnectivityMeasure,
) -> tuple[np.ndarray, np.ndarray]:
    time_series = masker.fit_transform(str(bold_path))  # (T, N_tian_parcels)

    if hippo_indices:
        # Subset to hippocampal parcels only (1-based → 0-based)
        idx = [i - 1 for i in hippo_indices]
        time_series = time_series[:, idx]

    corr_matrix = correlation_measure.fit_transform([time_series])[0]
    clipped = np.clip(corr_matrix, -0.999999, 0.999999)
    z_matrix = np.arctanh(clipped)
    np.fill_diagonal(z_matrix, 0.0)
    z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return corr_matrix, z_matrix


def process_file(
    bold_path: Path,
    masker: NiftiLabelsMasker,
    hippo_indices: list[int],
    correlation_measure: ConnectivityMeasure,
    matrices_out: Path,
) -> str:
    prefix = strip_nifti_suffix(bold_path.name)
    raw_out = matrices_out / f"{prefix}{OUTPUT_RAW_SUFFIX}"
    z_out = matrices_out / f"{prefix}{OUTPUT_Z_SUFFIX}"

    if raw_out.exists() and z_out.exists():
        return f"SKIP {bold_path.name}"

    corr, z = compute_connectivity_matrices(bold_path, masker, hippo_indices, correlation_measure)
    np.savez_compressed(raw_out, array=corr)
    np.savez_compressed(z_out, array=z)
    return f"DONE {bold_path.name} -> shape={corr.shape}"


def main(
    atlas_path: Path,
    labels_path: Path | None,
    fmri_root: Path | None = None,
    output_root: Path | None = None,
) -> None:
    fmri_root = fmri_root or DEFAULT_FMRI_ROOT
    output_root = output_root or DEFAULT_OUTPUT_ROOT

    if not fmri_root.exists():
        raise FileNotFoundError(f"fMRI root directory not found: {fmri_root}")
    if not atlas_path.exists():
        raise FileNotFoundError(f"Tian atlas not found: {atlas_path}")

    matrices_out = output_root / "matrices"
    matrices_out.mkdir(parents=True, exist_ok=True)

    metadata_link = output_root / "metadata"
    if not metadata_link.exists():
        v3_meta = REPO_ROOT / "DATA" / "DELCODE" / "__v3__" / "metadata"
        if v3_meta.exists():
            metadata_link.symlink_to(v3_meta.resolve())

    masker, hippo_indices = build_masker(atlas_path, labels_path)
    print(f"Hippocampal parcel indices (1-based): {hippo_indices or 'all'}")
    print(f"Source: {fmri_root}  |  Output: {matrices_out}")

    correlation_measure = ConnectivityMeasure(kind="correlation")
    bold_files = list(iter_bold_files(fmri_root))
    if not bold_files:
        print(f"No rest BOLD files found under {fmri_root}")
        return

    processed, skipped, failed = 0, 0, 0
    iterator: Any = tqdm(bold_files, unit="file", dynamic_ncols=True) if tqdm else bold_files

    for bold_path in iterator:
        try:
            msg = process_file(bold_path, masker, hippo_indices, correlation_measure, matrices_out)
            if msg.startswith("SKIP"):
                skipped += 1
            else:
                processed += 1
            if tqdm is None:
                print(msg)
        except Exception as exc:
            failed += 1
            err = f"ERROR {bold_path.name}: {exc}"
            if tqdm is not None:
                iterator.write(err)
            else:
                print(err)

    if tqdm is not None:
        iterator.close()

    print(f"\nDone — processed={processed}, skipped={skipped}, failed={failed}\nOutput: {matrices_out}")

    if labels_path and labels_path.exists():
        all_labels = labels_path.read_text().splitlines()
        selected = [all_labels[i - 1] for i in hippo_indices if i <= len(all_labels)] if hippo_indices else all_labels
        (output_root / "parcel_labels.txt").write_text("\n".join(selected) + "\n")
        print(f"Parcel labels: {selected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas-path", required=True, type=Path, help="Path to Tian atlas NIfTI file")
    parser.add_argument("--labels-path", type=Path, default=None, help="Path to Tian label text file (optional)")
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT, help="Root fMRI directory (all visits)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output version root (e.g. __v5__)")
    args = parser.parse_args()
    main(atlas_path=args.atlas_path, labels_path=args.labels_path,
         fmri_root=args.fmri_root, output_root=args.output_root)
