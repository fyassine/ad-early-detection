"""
subset_schaefer_networks.py — Extract network-specific correlation submatrices
from existing whole-brain __v3__ Schaefer 200 correlation matrices.

Usage (from repo root):
    python -m CLASSIFIER.src.processing.subset_schaefer_networks \\
        --networks Default Limbic \\
        --output-version __v9__ \\
        --output-suffix dmn_limbic

Available networks (Schaefer 200, 7 Yeo):
    Default (46), DorsAttn (26), Limbic (12), Cont (30),
    SomMot (35), Vis (29), SalVentAttn (22)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS_JSON = REPO_ROOT / "DASHBOARD" / "app" / "static" / "data" / "schaefer_200_coords.json"
V3_ROOT = REPO_ROOT / "DATA" / "DELCODE" / "__v3__"
DELCODE_ROOT = REPO_ROOT / "DATA" / "DELCODE"

INPUT_RAW_SUFFIX = "_whole_brain_correlation_matrix.npz"
INPUT_Z_SUFFIX = "_whole_brain_correlation_matrix_z_transformed.npz"
OVERWRITE_EXISTING = False


def load_network_indices(networks: list[str]) -> list[int]:
    """Return sorted list of 0-based ROI indices belonging to the given networks."""
    with ATLAS_JSON.open("r") as f:
        data = json.load(f)
    rois = data["rois"]
    indices: list[int] = []
    for roi in rois:
        if roi.get("network") in networks:
            indices.append(roi["index"])
    return sorted(indices)


def subset_matrix(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    idx = np.array(indices)
    return matrix[np.ix_(idx, idx)]


def process_file(
    npz_path: Path,
    indices: list[int],
    out_path: Path,
) -> str:
    if not OVERWRITE_EXISTING and out_path.exists():
        return f"SKIP {npz_path.name}"
    matrix = np.load(npz_path)["array"]
    sub = subset_matrix(matrix, indices)
    np.savez_compressed(out_path, array=sub)
    return f"DONE {npz_path.name} -> {out_path.name}"


def main(networks: list[str], output_version: str, output_suffix: str) -> None:
    matrices_in = V3_ROOT / "matrices"
    if not matrices_in.exists():
        raise FileNotFoundError(f"__v3__ matrices dir not found: {matrices_in}")

    indices = load_network_indices(networks)
    if not indices:
        raise ValueError(f"No ROIs found for networks: {networks}")
    print(f"Networks: {networks}  |  ROIs selected: {len(indices)}  |  indices: {indices[:5]}...")

    out_root = DELCODE_ROOT / output_version
    matrices_out = out_root / "matrices"
    matrices_out.mkdir(parents=True, exist_ok=True)

    # Symlink metadata from __v3__ (same subjects)
    metadata_link = out_root / "metadata"
    if not metadata_link.exists():
        metadata_link.symlink_to((V3_ROOT / "metadata").resolve())
        print(f"Linked metadata: {metadata_link} -> {V3_ROOT / 'metadata'}")

    raw_suffix_out = f"_{output_suffix}_correlation_matrix.npz"
    z_suffix_out = f"_{output_suffix}_correlation_matrix_z_transformed.npz"

    # Gather all raw input files (exclude z-transformed, we'll handle them separately)
    raw_files = sorted(matrices_in.glob(f"*{INPUT_RAW_SUFFIX}"))
    if not raw_files:
        print(f"No files matching *{INPUT_RAW_SUFFIX} in {matrices_in}")
        return

    processed, skipped, failed = 0, 0, 0
    iterator = tqdm(raw_files, unit="file", dynamic_ncols=True) if tqdm else raw_files

    for npz_path in iterator:
        stem = npz_path.name[: -len(INPUT_RAW_SUFFIX)]
        try:
            msg_raw = process_file(
                npz_path,
                indices,
                matrices_out / f"{stem}{raw_suffix_out}",
            )
            z_npz = matrices_in / f"{stem}{INPUT_Z_SUFFIX}"
            msg_z = "NO_Z_SOURCE" if not z_npz.exists() else process_file(
                z_npz,
                indices,
                matrices_out / f"{stem}{z_suffix_out}",
            )
            if msg_raw.startswith("SKIP"):
                skipped += 1
            else:
                processed += 1
            if tqdm is None:
                print(f"{msg_raw} | {msg_z}")
        except Exception as exc:
            failed += 1
            msg = f"ERROR {npz_path.name}: {exc}"
            if tqdm is not None:
                iterator.write(msg)
            else:
                print(msg)

    if tqdm is not None:
        iterator.close()

    print(
        f"\nDone — processed={processed}, skipped={skipped}, failed={failed}"
        f"\nOutput: {matrices_out}"
    )

    # Save parcel label list for traceability
    with ATLAS_JSON.open("r") as f:
        rois = json.load(f)["rois"]
    selected_labels = [r["label"] for r in rois if r.get("network") in networks]
    labels_path = out_root / "parcel_labels.txt"
    labels_path.write_text("\n".join(selected_labels) + "\n")
    print(f"Parcel labels saved: {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--networks", nargs="+", required=True,
        help="Schaefer 7-network names to include (e.g. Default Limbic)"
    )
    parser.add_argument(
        "--output-version", required=True,
        help="DELCODE data version directory name (e.g. __v6__)"
    )
    parser.add_argument(
        "--output-suffix", required=True,
        help="File name suffix for output matrices (e.g. limbic)"
    )
    args = parser.parse_args()
    main(networks=args.networks, output_version=args.output_version, output_suffix=args.output_suffix)
