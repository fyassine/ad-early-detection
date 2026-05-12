"""
process_combined_schaefer_tian.py — Joint FC matrix from Schaefer cortical
network subsets and Tian hippocampal parcels.

Concatenates time series from both maskers before computing FC so that
cross-region edges (e.g. DMN ↔ Hippocampus) are preserved.

Usage (from repo root):
    # DMN + Hippocampus → __v8__
    python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \\
        --networks Default \\
        --output-version __v8__ \\
        --output-suffix dmn_hippo \\
        --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \\
        --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt

    # DMN + Limbic + Hippocampus → __v10__
    python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \\
        --networks Default Limbic \\
        --output-version __v10__ \\
        --output-suffix dmn_limbic_hippo \\
        --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \\
        --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt

    # DMN + Limbic + DorsAttn + Hippocampus → __v11__
    python -m CLASSIFIER.src.processing.process_combined_schaefer_tian \\
        --networks Default Limbic DorsAttn \\
        --output-version __v11__ \\
        --output-suffix all_combined \\
        --tian-atlas /path/to/Tian_Subcortex_S2_3T.nii.gz \\
        --tian-labels /path/to/Tian_Subcortex_S2_3T_label.txt
"""

from __future__ import annotations

import argparse
import contextlib
import joblib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from nilearn import datasets, image as nli_image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FMRI_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "__v1__" / "fmri"
DELCODE_ROOT = REPO_ROOT / "DATA" / "DELCODE"
ATLAS_JSON = REPO_ROOT / "DASHBOARD" / "app" / "static" / "data" / "schaefer_200_coords.json"
SUBJECT_GLOB = "sub-*"


def load_schaefer_network_indices(networks: list[str]) -> tuple[list[int], list[str]]:
    """Return sorted 0-based ROI indices and labels for the given Schaefer networks."""
    with ATLAS_JSON.open("r") as f:
        data = json.load(f)
    rois = data["rois"]
    indices, labels = [], []
    for roi in rois:
        if roi.get("network") in networks:
            indices.append(roi["index"])
            labels.append(roi["label"])
    return sorted(range(len(indices)), key=lambda i: indices[i]), sorted(indices), labels


def load_hippocampus_label_indices(labels_path: Path | None) -> list[int]:
    """Return 1-based Tian atlas indices for hippocampal parcels."""
    if labels_path is None or not labels_path.exists():
        return []
    lines = labels_path.read_text().splitlines()
    return [i + 1 for i, ln in enumerate(lines) if "hip" in ln.lower()]


def build_schaefer_masker(network_indices: list[int]) -> NiftiLabelsMasker:
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    return NiftiLabelsMasker(
        labels_img=schaefer.maps,
        standardize=cast(Any, "zscore_sample"),
    )


def build_tian_masker(atlas_path: Path, labels_path: Path | None = None) -> NiftiLabelsMasker:
    """
    Build a NiftiLabelsMasker for the Tian atlas, subsetted to hippocampal
    parcels in memory so Nilearn only processes those voxels.
    """
    if labels_path is not None and labels_path.exists():
        all_labels = labels_path.read_text().splitlines()
        hippo_1based = [i + 1 for i, ln in enumerate(all_labels) if "hip" in ln.lower()]
    else:
        hippo_1based = []

    atlas_img = nli_image.load_img(str(atlas_path))
    if hippo_1based:
        atlas_data = atlas_img.get_fdata()
        subset_data = np.zeros_like(atlas_data)
        for label in hippo_1based:
            subset_data[atlas_data == label] = label
        atlas_img = nli_image.new_img_like(atlas_img, subset_data)

    return NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=cast(Any, "zscore_sample"),
        resampling_target="data",
    )


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


def compute_joint_connectivity(
    bold_path: Path,
    schaefer_masker: NiftiLabelsMasker,
    schaefer_col_indices: list[int],
    tian_masker: NiftiLabelsMasker,
    tian_hippo_indices: list[int],
    correlation_measure: ConnectivityMeasure,
) -> tuple[np.ndarray, np.ndarray]:
    # Load the fMRI image once and pass it to both maskers
    bold_img = nli_image.load_img(str(bold_path))

    # Schaefer: use transform() (masker already fitted) — avoids resampling
    ts_schaefer = schaefer_masker.transform(bold_img)  # (T, 200)
    ts_net = ts_schaefer[:, schaefer_col_indices]       # (T, N_schaefer_subset)

    # Tian: use transform() — masker is pre-fitted and pre-subsetted to hippo
    ts_tian = tian_masker.transform(bold_img)           # (T, N_hippo)
    # tian_masker is already subsetted; pass all columns
    ts_hippo = ts_tian if not tian_hippo_indices else ts_tian

    # Joint time series: [Schaefer subset | Tian hippocampus]
    ts_joint = np.concatenate([ts_net, ts_hippo], axis=1)  # (T, N_net + N_hippo)

    corr_matrix = correlation_measure.fit_transform([ts_joint])[0]
    clipped = np.clip(corr_matrix, -0.999999, 0.999999)
    z_matrix = np.arctanh(clipped)
    np.fill_diagonal(z_matrix, 0.0)
    z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return corr_matrix, z_matrix


def process_file(
    bold_path: Path,
    schaefer_masker: NiftiLabelsMasker,
    schaefer_col_indices: list[int],
    tian_masker: NiftiLabelsMasker,
    tian_hippo_indices: list[int],
    correlation_measure: ConnectivityMeasure,
    matrices_out: Path,
    raw_suffix: str,
    z_suffix: str,
) -> str:
    prefix = strip_nifti_suffix(bold_path.name)
    raw_out = matrices_out / f"{prefix}{raw_suffix}"
    z_out = matrices_out / f"{prefix}{z_suffix}"

    if raw_out.exists() and z_out.exists():
        return f"SKIP {bold_path.name}"

    corr, z = compute_joint_connectivity(
        bold_path, schaefer_masker, schaefer_col_indices,
        tian_masker, tian_hippo_indices, correlation_measure,
    )
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
    schaefer_masker: NiftiLabelsMasker,
    schaefer_col_indices: list[int],
    tian_masker: NiftiLabelsMasker,
    matrices_out: Path,
    raw_suffix: str,
    z_suffix: str,
) -> tuple[str | None, str | None]:
    """Top-level worker for joblib. Each process creates its own ConnectivityMeasure.

    Both maskers are pre-fitted before the parallel loop; workers call
    transform() instead of fit_transform(), eliminating per-file resampling.
    """
    cm = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    try:
        msg = process_file(
            bold_path, schaefer_masker, schaefer_col_indices,
            tian_masker, [], cm, matrices_out, raw_suffix, z_suffix,
        )
        return msg, None
    except Exception as exc:
        return None, f"ERROR {bold_path.name}: {exc}"


def main(
    networks: list[str],
    output_version: str,
    output_suffix: str,
    tian_atlas: Path,
    tian_labels: Path | None,
    fmri_root: Path | None = None,
    n_jobs: int = 16,
) -> None:
    fmri_root = fmri_root or DEFAULT_FMRI_ROOT
    if not fmri_root.exists():
        raise FileNotFoundError(f"fMRI root directory not found: {fmri_root}")
    if not tian_atlas.exists():
        raise FileNotFoundError(f"Tian atlas not found: {tian_atlas}")

    # Load Schaefer network ROI indices
    with ATLAS_JSON.open("r") as f:
        all_rois = json.load(f)["rois"]
    schaefer_col_indices = sorted(
        [r["index"] for r in all_rois if r.get("network") in networks]
    )
    schaefer_labels = [r["label"] for r in all_rois if r.get("network") in networks]
    tian_hippo_indices = load_hippocampus_label_indices(tian_labels)

    n_schaefer = len(schaefer_col_indices)
    n_hippo = len(tian_hippo_indices) if tian_hippo_indices else "all"
    print(f"Schaefer networks: {networks}  |  cortical ROIs: {n_schaefer}")
    print(f"Tian hippocampal parcels: {n_hippo}")

    out_root = DELCODE_ROOT / output_version
    matrices_out = out_root / "matrices"
    matrices_out.mkdir(parents=True, exist_ok=True)

    metadata_link = out_root / "metadata"
    if not metadata_link.exists():
        v3_meta = DELCODE_ROOT / "__v3__" / "metadata"
        if v3_meta.exists():
            metadata_link.symlink_to(v3_meta.resolve())

    raw_suffix = f"_{output_suffix}_correlation_matrix.npz"
    z_suffix = f"_{output_suffix}_correlation_matrix_z_transformed.npz"

    schaefer_masker = build_schaefer_masker(schaefer_col_indices)
    tian_masker = build_tian_masker(tian_atlas, tian_labels)  # pre-subsetted to hippo

    print(f"Source: {fmri_root}  |  Output: {matrices_out}")
    bold_files = list(iter_bold_files(fmri_root))
    if not bold_files:
        print(f"No rest BOLD files found under {fmri_root}")
        return

    # ── Pre-fit both maskers once on a single 3D volume ────────────────────────
    # Using index_img(ref, 0) extracts just the FIRST volume (3D) as the fitting
    # reference.  This avoids NiftiLabelsMasker.fit()'s compute_middle_image()
    # call, which seeks to the midpoint of the 4D file — prohibitively slow for
    # compressed .nii.gz files (requires sequential gzip decompression).
    # A 3D single-volume image is sufficient: the masker only needs the affine
    # and voxel grid to pre-compute the resampling transform.
    print("Pre-fitting maskers on reference image (3D single volume)...")
    ref_img_3d = nli_image.index_img(str(bold_files[0]), 0)
    schaefer_masker.fit(ref_img_3d)
    tian_masker.fit(ref_img_3d)
    print("  Maskers fitted ✓")

    # ── Cap n_jobs to prevent OOM ─────────────────────────────────────────────
    # Combined processing runs TWO maskers per file (Schaefer 200-ROI + Tian
    # hippocampus). Each worker holds two NIfTI images in memory simultaneously.
    # With 24 workers this easily exceeds available RAM → SIGKILL(-9).
    # Safe ceiling: ~8-12 workers for combined experiments.
    safe_jobs = min(n_jobs, 12)
    if safe_jobs < n_jobs:
        print(
            f"  ⚠  Capping n_jobs {n_jobs} → {safe_jobs} for combined processing\n"
            f"     (two NIfTI maskers per worker → higher per-worker RAM).\n"
            f"     Override with --n-jobs {safe_jobs} to silence this warning."
        )

    print(f"Processing {len(bold_files)} files in parallel (n_jobs={safe_jobs})…")

    _tqdm = tqdm or (lambda **kw: contextlib.nullcontext())
    with tqdm_joblib(_tqdm(desc="Processing fMRI", total=len(bold_files), dynamic_ncols=True)) as _:
        raw_results = Parallel(n_jobs=safe_jobs, verbose=0)(
            delayed(_parallel_worker)(
                p, schaefer_masker, schaefer_col_indices,
                tian_masker, matrices_out, raw_suffix, z_suffix,
            )
            for p in bold_files
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

    print(
        f"\nDone — processed={processed}, skipped={skipped}, failed={failed}"
        f"\nOutput: {matrices_out}"
    )

    # Save parcel labels for traceability
    all_labels = schaefer_labels[:]
    if tian_labels and tian_labels.exists():
        tian_all = tian_labels.read_text().splitlines()
        if tian_hippo_indices:
            all_labels += [tian_all[i - 1] for i in tian_hippo_indices if i <= len(tian_all)]
        else:
            all_labels += tian_all
    (out_root / "parcel_labels.txt").write_text("\n".join(all_labels) + "\n")
    print(f"Parcel labels saved to {out_root / 'parcel_labels.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks", nargs="+", required=True, help="Schaefer networks to include")
    parser.add_argument("--output-version", required=True, help="DELCODE version dir (e.g. __v8__)")
    parser.add_argument("--output-suffix", required=True, help="File suffix (e.g. dmn_hippo)")
    parser.add_argument("--tian-atlas", required=True, type=Path, help="Tian atlas NIfTI")
    parser.add_argument("--tian-labels", type=Path, default=None, help="Tian label text file")
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT,
                        help="Root fMRI directory (all visits, default: __v1__/fmri)")
    parser.add_argument("--n-jobs", type=int, default=16,
                        help="Parallel workers (default: 16). Lower if you hit OOM.")
    args = parser.parse_args()
    main(
        networks=args.networks,
        output_version=args.output_version,
        output_suffix=args.output_suffix,
        tian_atlas=args.tian_atlas,
        tian_labels=args.tian_labels,
        fmri_root=args.fmri_root,
        n_jobs=args.n_jobs,
    )
