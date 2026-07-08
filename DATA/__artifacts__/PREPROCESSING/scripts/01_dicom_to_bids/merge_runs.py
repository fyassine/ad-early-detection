#!/usr/bin/env python3
"""Stage 1.2: merge a split resting-state session into one image.

nibabel/numpy replacement for the original fslmerge3.py — no FSL dependency. Unlike the
original (which hardcodes run-folder numbers 005/006/007 for every subject), this discovers
however many BOLD run files are passed in and merges them in the given order. If only one run
is given, it is copied through unchanged rather than merged (true for the SAMPLE subject,
which has a single uninterrupted RestingState series).

Adds a shape/affine consistency check before concatenating — the original had none, which is a
silent-failure risk if two runs don't actually share geometry.

Usage:
    python merge_runs.py <out_merged.nii.gz> <run1.nii.gz> [<run2.nii.gz> ...]

Writes <out_merged>.frames.json recording how many volumes came from each input run, so
downstream FD/QC steps can still recover per-run boundaries from the merged series.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+", help="run NIfTI files, in acquisition order")
    args = parser.parse_args()

    imgs = [nib.load(str(p)) for p in args.runs]

    if len(imgs) == 1:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(imgs[0], str(args.output))
        frame_counts = [imgs[0].shape[3] if imgs[0].ndim == 4 else 1]
    else:
        ref_shape, ref_affine = imgs[0].shape[:3], imgs[0].affine
        for path, img in zip(args.runs[1:], imgs[1:]):
            if img.shape[:3] != ref_shape:
                sys.exit(f"Spatial shape mismatch: {args.runs[0]} {ref_shape} vs {path} {img.shape[:3]}")
            if not np.allclose(img.affine, ref_affine, atol=1e-3):
                sys.exit(f"Affine mismatch: {args.runs[0]} vs {path} — refusing to merge silently")

        data = np.concatenate([img.get_fdata(dtype=np.float32) for img in imgs], axis=3)
        merged = nib.Nifti1Image(data, ref_affine, imgs[0].header)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(merged, str(args.output))
        frame_counts = [img.shape[3] for img in imgs]

    sidecar = args.output.with_suffix("").with_suffix(".frames.json")
    sidecar.write_text(json.dumps({
        "source_runs": [str(p) for p in args.runs],
        "frames_per_run": frame_counts,
    }, indent=2))

    print(f"Wrote {args.output} ({sum(frame_counts)} volumes from {len(imgs)} run(s))")


if __name__ == "__main__":
    main()
