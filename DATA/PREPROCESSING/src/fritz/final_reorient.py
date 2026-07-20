#!/usr/bin/env python3
"""Late-stage reorientation to radiological convention (Fritz-side, self-contained).

Runs AFTER fMRIPrep + the postprocessing container, on the final
``*_desc-<STRATEGY>_bold.nii.gz`` images, turning them into the
``*_desc-<STRATEGY>_bold_reoriented.nii.gz`` files that the flat product
(``__fmri_wholebrain_sch200_flat__/fmri/sub-*/``) expects. Pure nibabel/numpy —
no FSL dependency.

This is a self-contained copy of the reference implementation
``DATA/__artifacts__/PREPROCESSING/scripts/05_reorientation/final_reorient_nibabel.py``
so the active Fritz→CORE continuous pipeline owns its own dependency and does not
reach into the legacy ``__artifacts__`` tree at runtime. Keep the two in sync if
the reference ever changes.

The original institutional ``final_reorient.py`` did two operationally distinct
things via two separate FSL calls, and this replacement keeps them as two
distinct steps rather than collapsing them (collapsing is the easy-to-get-wrong
bug: the affine can look "radiological" while the voxel array was never actually
reordered):

  1. ``fslswapdim`` physically permutes/flips the voxel data array to a canonical
     axis order. Replicated with ``nibabel.as_closest_canonical()``, which
     reorders data+affine to the closest RAS+ approximation.
  2. ``fslorient -forceradiological`` only flips the sign convention in the
     affine/header (negates the x-axis mapping) to mark radiological convention,
     WITHOUT touching voxel data. Replicated as a standalone affine-only edit.

CAVEAT: not validated byte-for-byte against real FSL output. Sanity-check with
``--print-orientation`` on a known case before trusting on real data.

Usage:
    python final_reorient.py <input.nii.gz> <output.nii.gz>
    python final_reorient.py --glob "<postproc_out>/sub-*/ses-*/*_desc-ICAAROMA2Phys1GS_bold.nii.gz"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import nibabel as nib


def reorient_to_radiological(img: nib.Nifti1Image) -> nib.Nifti1Image:
    # Step 1: physically reorder/flip the data array to the closest canonical RAS+ orientation.
    canonical = nib.as_closest_canonical(img)

    # Step 2: header-only sign flip of the x (left-right) axis to mark radiological
    # convention, without touching the voxel array (mirrors fslorient -forceradiological).
    affine = canonical.affine.copy()
    affine[:, 0] *= -1
    return nib.Nifti1Image(canonical.dataobj, affine, canonical.header)


def reoriented_output_path(input_path: Path) -> Path:
    """The canonical ``*_bold.nii.gz`` -> ``*_bold_reoriented.nii.gz`` name."""
    return input_path.with_name(input_path.name.replace(".nii.gz", "_reoriented.nii.gz"))


def process_one(input_path: Path, output_path: Path, *, overwrite: bool = False) -> str:
    if output_path.exists() and not overwrite:
        return f"SKIP {input_path.name} (exists)"
    img = nib.load(str(input_path))
    before = nib.aff2axcodes(img.affine)
    reoriented = reorient_to_radiological(img)
    after = nib.aff2axcodes(reoriented.affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reoriented.to_filename(str(output_path))
    return f"DONE {input_path.name}: {before} -> {after}  ->  {output_path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--glob", help="glob pattern to discover multiple final BOLD files")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-reorient even if the output exists"
    )
    parser.add_argument(
        "--print-orientation",
        action="store_true",
        help="only print before/after axis codes, do not write output",
    )
    args = parser.parse_args()

    if args.glob:
        matches = sorted(glob.glob(args.glob))
        if not matches:
            sys.exit(f"No files matched: {args.glob}")
        for match in matches:
            in_path = Path(match)
            if args.print_orientation:
                img = nib.load(str(in_path))
                print(f"{in_path.name}: {nib.aff2axcodes(img.affine)}")
                continue
            print(process_one(in_path, reoriented_output_path(in_path), overwrite=args.overwrite))
    elif args.input and args.print_orientation:
        img = nib.load(str(args.input))
        print(f"{args.input.name}: {nib.aff2axcodes(img.affine)}")
    elif args.input and args.output:
        print(process_one(args.input, args.output, overwrite=args.overwrite))
    else:
        parser.error("either give <input> <output>, or --glob <pattern>")


if __name__ == "__main__":
    main()
