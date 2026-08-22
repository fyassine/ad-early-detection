#!/usr/bin/env python3
"""Stage 5: late-stage reorientation to radiological convention.

nibabel/numpy replacement for the original final_reorient.py — no FSL dependency. Runs AFTER
fMRIPrep + postprocessing, on the final ICAAROMA2Phys1GS images (decision: reorientation is
late-only, not early after dcm2niix), exactly mirroring where the original ran.

The original did two operationally distinct things via two separate FSL calls, and this
replacement keeps them as two distinct steps rather than collapsing them into one (collapsing
them is the easy-to-get-wrong bug: it can look "radiological" in the affine while the voxel
array was never actually reordered):

  1. `fslswapdim` physically permutes/flips the voxel data array to a canonical axis order.
     Replicated here with nibabel.as_closest_canonical(), which reorders data+affine to the
     closest RAS+ approximation — nibabel's own documented equivalent of this operation (the
     original docs themselves point at the NiBabel Orientation Guide for this).
  2. `fslorient -forceradiological` only flips the sign convention in the affine/header
     (negates the x-axis mapping) to mark radiological convention, WITHOUT touching voxel data.
     Replicated here as a standalone affine-only edit after step 1.

CAVEAT: this has not been validated byte-for-byte against real FSL output (FSL itself isn't
installed on this system — see docs/OPEN_QUESTIONS.md). Before trusting this on real data,
sanity-check with --print-orientation on a known case, and ideally compare against an
FSL-processed reference if one becomes available.

Usage:
    python final_reorient_nibabel.py <input.nii.gz> <output.nii.gz>
    python final_reorient_nibabel.py --glob "<fmriprep_out>/sub-*/ses-*/func/*_desc-ICAAROMA2Phys1GS_bold.nii.gz"
"""
import argparse
import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def reorient_to_radiological(img: nib.Nifti1Image) -> nib.Nifti1Image:
    # Step 1: physically reorder/flip the data array to the closest canonical RAS+ orientation.
    canonical = nib.as_closest_canonical(img)

    # Step 2: header-only sign flip of the x (left-right) axis to mark radiological convention,
    # without touching the voxel array (mirrors fslorient -forceradiological).
    # Negating the direction cosine alone anchors the flip at world x=0 instead of the
    # image's own bounding box, translating the whole volume by (nx-1)*voxel_size_x — the
    # translation term must be compensated to keep the image in the same physical location.
    affine = canonical.affine.copy()
    x_dir = affine[:3, 0].copy()
    nx = canonical.shape[0]
    affine[:3, 0] = -x_dir
    affine[:3, 3] += x_dir * (nx - 1)
    return nib.Nifti1Image(canonical.dataobj, affine, canonical.header)


def process_one(input_path: Path, output_path: Path) -> None:
    img = nib.load(str(input_path))
    before = nib.aff2axcodes(img.affine)
    reoriented = reorient_to_radiological(img)
    after = nib.aff2axcodes(reoriented.affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reoriented.to_filename(str(output_path))
    print(f"{input_path.name}: {before} -> {after}  ->  {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--glob", help="glob pattern to discover multiple subjects' final BOLD files")
    args = parser.parse_args()

    if args.glob:
        matches = sorted(glob.glob(args.glob))
        if not matches:
            sys.exit(f"No files matched: {args.glob}")
        for match in matches:
            in_path = Path(match)
            out_path = in_path.with_name(in_path.name.replace(".nii.gz", "_reoriented.nii.gz"))
            process_one(in_path, out_path)
    elif args.input and args.output:
        process_one(args.input, args.output)
    else:
        parser.error("either give <input> <output>, or --glob <pattern>")


if __name__ == "__main__":
    main()
