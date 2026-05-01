"""
generate_schaefer_coords.py — One-time helper to produce the JSON file that
``/api/atlas/schaefer/coords`` serves to the Brain View glass-brain.

Computes the MNI centroid of every Schaefer parcel from the parcellation NIfTI
and pairs it with the network/hemisphere parsed from the standard label file.

Usage
-----
    python -m app.generate_schaefer_coords \\
        --parcellation /path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \\
        --labels       /path/to/Schaefer2018_200Parcels_7Networks_order.txt \\
        --n-parcels    200

Both reference files ship with the official Schaefer atlas release (CBIG GitHub).
The output lands at ``app/static/data/schaefer_{n_parcels}_coords.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _parse_labels(labels_path: Path) -> dict[int, dict]:
    """
    Parse the Schaefer label file (whitespace-delimited):
        <index> <label> <R> <G> <B> <A>
    where <label> looks like ``7Networks_LH_Default_PFC_1`` or
    ``17Networks_RH_VisCent_ExStr_3``.
    """
    out: dict[int, dict] = {}
    pattern = re.compile(
        r"^\s*(\d+)\s+(\S+)"
    )
    with labels_path.open("r") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            idx = int(m.group(1))  # 1-based in the file
            label = m.group(2)
            parts = label.split("_")
            # parts[0] = '7Networks' / '17Networks'
            hemisphere = parts[1] if len(parts) > 1 else ""
            network = parts[2] if len(parts) > 2 else ""
            out[idx - 1] = {  # store 0-based for consistency with array indices
                "label": label,
                "network": network,
                "hemisphere": hemisphere,
            }
    return out


def _compute_centroids(parc_path: Path) -> tuple[dict[int, tuple[float, float, float]], int]:
    """
    Compute MNI-mm centroid for each non-zero label in the parcellation NIfTI.
    Returns ``(centroids[label_idx_0based] = (x, y, z), max_label)``.
    """
    import nibabel as nib  # imported here so the runtime endpoints don't need nibabel

    img = nib.load(str(parc_path))
    data = np.asarray(img.dataobj)
    affine = img.affine

    centroids: dict[int, tuple[float, float, float]] = {}
    labels = np.unique(data)
    labels = labels[labels > 0]
    for lab in labels:
        coords = np.argwhere(data == lab)
        if coords.size == 0:
            continue
        # Voxel centroid → MNI mm via affine
        vox = coords.mean(axis=0)
        mni = affine @ np.array([vox[0], vox[1], vox[2], 1.0])
        centroids[int(lab) - 1] = (float(mni[0]), float(mni[1]), float(mni[2]))

    return centroids, int(labels.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parcellation", required=True, type=Path,
                        help="Path to the Schaefer parcellation NIfTI (MNI space).")
    parser.add_argument("--labels", required=True, type=Path,
                        help="Path to the Schaefer label .txt (e.g. Schaefer2018_200Parcels_7Networks_order.txt).")
    parser.add_argument("--n-parcels", type=int, default=200,
                        help="Number of parcels (used for the output filename).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent / "static" / "data",
                        help="Output directory.")
    args = parser.parse_args()

    print(f"Parsing labels: {args.labels}")
    labels_meta = _parse_labels(args.labels)

    print(f"Computing centroids: {args.parcellation}")
    centroids, max_label = _compute_centroids(args.parcellation)

    rois: list[dict] = []
    for idx in sorted(centroids.keys()):
        meta = labels_meta.get(idx, {})
        x, y, z = centroids[idx]
        rois.append({
            "index": idx,                  # 0-based
            "label": meta.get("label", ""),
            "network": meta.get("network", ""),
            "hemisphere": meta.get("hemisphere", ""),
            "x_mni": round(x, 2),
            "y_mni": round(y, 2),
            "z_mni": round(z, 2),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"schaefer_{args.n_parcels}_coords.json"
    payload = {"n_parcels": args.n_parcels, "rois": rois}
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(rois)} ROIs → {out_path}")


if __name__ == "__main__":
    main()
