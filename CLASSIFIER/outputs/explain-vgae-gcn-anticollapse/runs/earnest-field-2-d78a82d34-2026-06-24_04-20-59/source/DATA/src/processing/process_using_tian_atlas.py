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

Output saved to DATA/DELCODE/__fc_hippo_tian2_flat__/matrices/
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from joblib import Parallel, delayed
from nilearn import image as nli_image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FMRI_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "__fmri_wholebrain_sch200_flat__" / "fmri"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "__fc_hippo_tian2_flat__"
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
    return [ln.strip() for ln in lines if "hip" in ln.lower()]


def build_masker(atlas_path: Path, labels_path: Path | None) -> tuple[NiftiLabelsMasker, list[int]]:
    """
    Build a NiftiLabelsMasker restricted to hippocampal parcels only.

    The atlas NIfTI is subsetted *in memory* so that Nilearn never touches
    the other ~28 subcortical regions during fit_transform.  Returns the
    masker and the original 1-based label indices (kept for label saving);
    the masker itself already handles the subsetting so callers should pass
    hippo_indices=[] to compute_connectivity_matrices.
    """
    if labels_path is not None and labels_path.exists():
        all_labels = labels_path.read_text().splitlines()
        hippo_1based = [
            i + 1 for i, ln in enumerate(all_labels)
            if "hip" in ln.lower()
        ]
    else:
        hippo_1based = []

    # Subset the atlas to hippocampal voxels only so fit_transform is fast.
    atlas_img = nli_image.load_img(str(atlas_path))
    if hippo_1based:
        atlas_data = atlas_img.get_fdata()
        subset_data = np.zeros_like(atlas_data)
        for label in hippo_1based:
            subset_data[atlas_data == label] = label
        atlas_img = nli_image.new_img_like(atlas_img, subset_data)

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        labels=None,
        standardize=cast(Any, "zscore_sample"),
        resampling_target="data",
    )
    # Return the original indices for label-saving; pass [] to compute fn.
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
    bold_img,   # pre-loaded NIfTI image (or path for fallback)
    masker: NiftiLabelsMasker,
    hippo_indices: list[int],
    correlation_measure: ConnectivityMeasure,
) -> tuple[np.ndarray, np.ndarray]:
    # masker is pre-fitted; use transform() to skip resampling overhead
    time_series = masker.transform(bold_img)  # (T, N_hippo_parcels)

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

    # Load image once; masker.transform() reuses fitted affine
    bold_img = nli_image.load_img(str(bold_path))
    corr, z = compute_connectivity_matrices(bold_img, masker, hippo_indices, correlation_measure)
    np.savez_compressed(raw_out, array=corr)
    np.savez_compressed(z_out, array=z)
    return f"DONE {bold_path.name} -> shape={corr.shape}"


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report batch completions into a tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback


def _parallel_worker(
    bold_path: Path,
    masker: NiftiLabelsMasker,
    matrices_out: Path,
) -> tuple[str | None, str | None]:
    """Top-level worker for joblib. Each process creates its own ConnectivityMeasure."""
    cm = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    try:
        msg = process_file(bold_path, masker, [], cm, matrices_out)
        return msg, None
    except Exception as exc:
        return None, f"ERROR {bold_path.name}: {exc}"


def main(
    atlas_path: Path,
    labels_path: Path | None,
    fmri_root: Path | None = None,
    output_root: Path | None = None,
    n_jobs: int = 16,
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
        v3_meta = REPO_ROOT / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__" / "metadata"
        if v3_meta.exists():
            metadata_link.symlink_to(v3_meta.resolve())

    masker, hippo_indices = build_masker(atlas_path, labels_path)
    print(f"Hippocampal parcel indices (1-based): {hippo_indices or 'all'}")
    print(f"Source: {fmri_root}  |  Output: {matrices_out}")

    bold_files = list(iter_bold_files(fmri_root))
    if not bold_files:
        print(f"No rest BOLD files found under {fmri_root}")
        return

    # ── Pre-fit masker once on a single 3D volume ──────────────────────────────
    # index_img(ref, 0) extracts only the first timepoint (3D), which avoids
    # NiftiLabelsMasker.fit()'s compute_middle_image() seeking the midpoint of
    # a compressed .nii.gz file (sequential gzip decompression — very slow).
    print("Pre-fitting hippocampal masker on reference image (3D single volume)...")
    ref_img_3d = nli_image.index_img(str(bold_files[0]), 0)
    masker.fit(ref_img_3d)
    print("  Masker fitted ✓")


    print(f"Processing {len(bold_files)} files in parallel (n_jobs={n_jobs})…")

    _tqdm = tqdm or (lambda **kw: contextlib.nullcontext())
    with tqdm_joblib(_tqdm(desc="Processing fMRI", total=len(bold_files), dynamic_ncols=True)) as _:
        raw_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_parallel_worker)(p, masker, matrices_out) for p in bold_files
        )

    processed, skipped, failed = 0, 0, 0
    for msg, err in raw_results:
        if err:
            failed += 1
            print(err)
        elif msg and msg.startswith("SKIP"):
            skipped += 1
        else:
            processed += 1

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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output version root (e.g. __fc_hippo_tian2_flat__)")
    parser.add_argument("--n-jobs", type=int, default=16,
                        help="Parallel workers (default: 16). Lower if you hit OOM.")
    args = parser.parse_args()
    main(atlas_path=args.atlas_path, labels_path=args.labels_path,
         fmri_root=args.fmri_root, output_root=args.output_root, n_jobs=args.n_jobs)
