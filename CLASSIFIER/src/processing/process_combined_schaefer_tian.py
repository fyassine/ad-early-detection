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
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "fmri" / "baseline"
DELCODE_ROOT = REPO_ROOT / "DATA" / "DELCODE"
ATLAS_JSON = REPO_ROOT / "DASHBOARD" / "app" / "static" / "data" / "schaefer_200_coords.json"
SUBJECT_GLOB = "sub-*"
OVERWRITE_EXISTING = False


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
    return [i + 1 for i, ln in enumerate(lines) if "hippocampus" in ln.lower()]


def build_schaefer_masker(network_indices: list[int]) -> NiftiLabelsMasker:
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    return NiftiLabelsMasker(
        labels_img=schaefer.maps,
        standardize=cast(Any, "zscore_sample"),
    )


def build_tian_masker(atlas_path: Path) -> NiftiLabelsMasker:
    return NiftiLabelsMasker(
        labels_img=str(atlas_path),
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
    # Schaefer: extract full 200-ROI time series, then select network columns
    ts_schaefer = schaefer_masker.fit_transform(str(bold_path))  # (T, 200)
    ts_net = ts_schaefer[:, schaefer_col_indices]  # (T, N_schaefer_subset)

    # Tian: extract all parcels, then keep hippocampus (1-based → 0-based)
    ts_tian = tian_masker.fit_transform(str(bold_path))  # (T, N_tian)
    if tian_hippo_indices:
        ts_hippo = ts_tian[:, [i - 1 for i in tian_hippo_indices]]
    else:
        ts_hippo = ts_tian

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

    if not OVERWRITE_EXISTING and raw_out.exists() and z_out.exists():
        return f"SKIP {bold_path.name}"

    corr, z = compute_joint_connectivity(
        bold_path, schaefer_masker, schaefer_col_indices,
        tian_masker, tian_hippo_indices, correlation_measure,
    )
    np.savez_compressed(raw_out, array=corr)
    np.savez_compressed(z_out, array=z)
    return f"DONE {bold_path.name} -> shape={corr.shape}"


def main(
    networks: list[str],
    output_version: str,
    output_suffix: str,
    tian_atlas: Path,
    tian_labels: Path | None,
) -> None:
    if not BASELINE_ROOT.exists():
        raise FileNotFoundError(f"Baseline fMRI directory not found: {BASELINE_ROOT}")
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
    tian_masker = build_tian_masker(tian_atlas)
    correlation_measure = ConnectivityMeasure(kind="correlation")

    bold_files = list(iter_bold_files(BASELINE_ROOT))
    if not bold_files:
        print(f"No rest BOLD files found under {BASELINE_ROOT}")
        return

    processed, skipped, failed = 0, 0, 0
    iterator: Any = tqdm(bold_files, unit="file", dynamic_ncols=True) if tqdm else bold_files

    for bold_path in iterator:
        try:
            msg = process_file(
                bold_path, schaefer_masker, schaefer_col_indices,
                tian_masker, tian_hippo_indices, correlation_measure,
                matrices_out, raw_suffix, z_suffix,
            )
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
    args = parser.parse_args()
    main(
        networks=args.networks,
        output_version=args.output_version,
        output_suffix=args.output_suffix,
        tian_atlas=args.tian_atlas,
        tian_labels=args.tian_labels,
    )
