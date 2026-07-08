#!/usr/bin/env python3
"""Stage 1.3: assemble dcm2niix output (per-subject staging dir) into a real BIDS tree.

Replacement for the original BIDS_og.py. Differences that fix real bugs in the original:
  - Sidecar content (RepetitionTime, EchoTime, SliceTiming, ...) comes from dcm2niix's own
    `-b y` JSON output, never hand-rolled/hardcoded.
  - Writes magnitude1/magnitude2/phasediff fieldmap triples with EchoTime1/EchoTime2 and
    IntendedFor populated (geometry-matched against the func run), which the original never
    produced at all.
  - Series that don't map to anything this pipeline needs (FLASH, IR-EPI, T2, a fieldmap with
    no geometry match) are left out of the BIDS tree and logged, rather than silently mis-copied.

Usage:
    python build_bids.py <staging_dir> <bids_root> <subject_id> [--session 1]

Example:
    python build_bids.py staging/03a0a6663-M0_T1_01 BIDS 03a0a6663 --session 1
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from series_classification import classify


def sanitize_label(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", raw)


def load_sidecars(staging_dir: Path) -> list[tuple[Path, dict]]:
    pairs = []
    for json_path in sorted(staging_dir.glob("*.json")):
        nii_path = next(
            (p for p in [json_path.with_suffix(".nii.gz"), Path(str(json_path)[:-5] + ".nii.gz")]
             if p.exists()),
            None,
        )
        if nii_path is None:
            # dcm2niix names are <stem>.json / <stem>.nii.gz where <stem> already has no .json
            candidate = staging_dir / (json_path.stem + ".nii.gz")
            nii_path = candidate if candidate.exists() else None
        if nii_path is None:
            print(f"  [WARN] no NIfTI found for sidecar {json_path}", file=sys.stderr)
            continue
        pairs.append((nii_path, json.loads(json_path.read_text())))
    return pairs


def voxel_size(nii_path: Path) -> tuple[float, ...]:
    # dcm2niix's BIDS JSON sidecar doesn't carry in-plane PixelSpacing (that's a raw DICOM tag
    # name it doesn't re-expose) — read voxel dimensions from the NIfTI header instead, which
    # is reliable regardless of what the sidecar does or doesn't include.
    import nibabel as nib
    zooms = nib.load(str(nii_path)).header.get_zooms()[:3]
    return tuple(round(float(z), 1) for z in zooms)


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=4))


def group_by_voxel(items: list[tuple[Path, dict]]) -> dict[tuple[float, ...], list[tuple[Path, dict]]]:
    """Group (nii_path, sidecar) pairs by their 3D voxel geometry, preserving input order."""
    groups: dict[tuple[float, ...], list[tuple[Path, dict]]] = {}
    for nii_path, sidecar in items:
        groups.setdefault(voxel_size(nii_path), []).append((nii_path, sidecar))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging_dir", type=Path)
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("subject_id", help="raw subject identifier, will be sanitized to a BIDS label")
    parser.add_argument("--session", default="1")
    args = parser.parse_args()

    sub_label = sanitize_label(args.subject_id)
    ses_label = sanitize_label(args.session)
    sub_dir = args.bids_root / f"sub-{sub_label}" / f"ses-{ses_label}"
    anat_dir, func_dir, fmap_dir, dwi_dir = (sub_dir / d for d in ("anat", "func", "fmap", "dwi"))

    pairs = load_sidecars(args.staging_dir)
    if not pairs:
        sys.exit(f"No NIfTI+JSON pairs found in {args.staging_dir} — run run_dcm2niix.py first")

    func_runs, fmap_magnitudes, fmap_phasediffs, unmapped = [], [], [], []

    for nii_path, sidecar in pairs:
        desc = sidecar.get("SeriesDescription", "")
        image_type = sidecar.get("ImageType", [])
        result = classify(desc, image_type)

        if result.category == "skip":
            continue
        if result.category == "unmapped":
            unmapped.append((nii_path, desc))
            continue

        if result.category == "anat_T1w":
            anat_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_rec-{result.rec_label}" if result.rec_label else ""
            dest = anat_dir / f"sub-{sub_label}_ses-{ses_label}{suffix}_T1w.nii.gz"
            shutil.copy(nii_path, dest)
            write_json(dest.with_suffix("").with_suffix(".json"), sidecar)

        elif result.category == "anat_FLAIR":
            anat_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_rec-{result.rec_label}" if result.rec_label else ""
            dest = anat_dir / f"sub-{sub_label}_ses-{ses_label}{suffix}_FLAIR.nii.gz"
            shutil.copy(nii_path, dest)
            write_json(dest.with_suffix("").with_suffix(".json"), sidecar)

        elif result.category == "dwi":
            dwi_dir.mkdir(parents=True, exist_ok=True)
            dest = dwi_dir / f"sub-{sub_label}_ses-{ses_label}_dwi.nii.gz"
            shutil.copy(nii_path, dest)
            write_json(dest.with_suffix("").with_suffix(".json"), sidecar)
            bval = nii_path.with_suffix("").with_suffix(".bval")
            bvec = nii_path.with_suffix("").with_suffix(".bvec")
            if bval.exists():
                shutil.copy(bval, dest.with_suffix("").with_suffix(".bval"))
            if bvec.exists():
                shutil.copy(bvec, dest.with_suffix("").with_suffix(".bvec"))

        elif result.category == "func_bold":
            func_runs.append((nii_path, sidecar))

        elif result.category == "fmap_magnitude":
            fmap_magnitudes.append((nii_path, sidecar))

        elif result.category == "fmap_phasediff":
            fmap_phasediffs.append((nii_path, sidecar))

    # --- functional runs (merge handled by merge_runs.py upstream if >1 run; here we just
    # place whatever run files classification found — single run for the SAMPLE subject) ---
    func_rel_path = None
    if func_runs:
        func_dir.mkdir(parents=True, exist_ok=True)
        if len(func_runs) > 1:
            print(f"  [INFO] {len(func_runs)} resting-state runs found — run merge_runs.py "
                  f"first and re-run pointing at the merged file instead.", file=sys.stderr)
        nii_path, sidecar = func_runs[0]
        dest = func_dir / f"sub-{sub_label}_ses-{ses_label}_task-rest_bold.nii.gz"
        shutil.copy(nii_path, dest)
        sidecar = {**sidecar, "TaskName": "rest"}
        write_json(dest.with_suffix("").with_suffix(".json"), sidecar)
        func_rel_path = f"ses-{ses_label}/func/{dest.name}"

    # --- fieldmap pair ---
    # A subject can have MORE than one GRE fieldmap acquisition (this SAMPLE has both a 3.5iso
    # and a 3iso fieldmap). magnitude1/magnitude2 MUST come from the *same* acquisition, or
    # fMRIPrep's magnitude-merge node crashes on a shape mismatch. So group magnitudes and
    # phasediffs by voxel geometry first, then select the single fieldmap whose geometry matches
    # the BOLD run (Siemens GRE fieldmaps are acquired at the EPI's resolution), and emit only
    # that one. Other fieldmap acquisitions are dropped (logged).
    if fmap_magnitudes:
        func_voxel = voxel_size(func_runs[0][0]) if func_runs else None
        mag_groups = group_by_voxel(fmap_magnitudes)
        phase_groups = group_by_voxel(fmap_phasediffs)

        if func_voxel in mag_groups:
            chosen_voxel = func_voxel
        elif len(mag_groups) == 1:
            chosen_voxel = next(iter(mag_groups))
        else:
            chosen_voxel = max(mag_groups, key=lambda k: len(mag_groups[k]))
            print(f"  [WARN] {len(mag_groups)} fieldmap geometries {list(mag_groups)} and none "
                  f"matches the func voxel {func_voxel}; using {chosen_voxel}. Review manually.",
                  file=sys.stderr)

        dropped = [k for k in mag_groups if k != chosen_voxel]
        if dropped:
            print(f"  [INFO] using fieldmap geometry {chosen_voxel}, dropping other fieldmap "
                  f"acquisition(s) at {dropped}.", file=sys.stderr)

        fmap_dir.mkdir(parents=True, exist_ok=True)
        chosen_mags = sorted(mag_groups[chosen_voxel], key=lambda pair: pair[1].get("EchoTime", 0))

        for i, (nii_path, sidecar) in enumerate(chosen_mags[:2], start=1):
            dest = fmap_dir / f"sub-{sub_label}_ses-{ses_label}_magnitude{i}.nii.gz"
            shutil.copy(nii_path, dest)
            write_json(dest.with_suffix("").with_suffix(".json"), sidecar)

        chosen_phase = phase_groups.get(chosen_voxel, [])
        if chosen_phase:
            nii_path, sidecar = chosen_phase[0]
            dest = fmap_dir / f"sub-{sub_label}_ses-{ses_label}_phasediff.nii.gz"
            shutil.copy(nii_path, dest)
            phasediff_json = {
                **sidecar,
                "EchoTime1": chosen_mags[0][1].get("EchoTime"),
                "EchoTime2": chosen_mags[-1][1].get("EchoTime"),
            }
            if func_rel_path:
                phasediff_json["IntendedFor"] = func_rel_path
            write_json(dest.with_suffix("").with_suffix(".json"), phasediff_json)
        else:
            print(f"  [WARN] no phasediff found at the chosen fieldmap geometry {chosen_voxel} — "
                  f"magnitudes written without a phasediff; fMRIPrep SDC will be skipped.",
                  file=sys.stderr)

    if unmapped:
        print(f"  [INFO] {len(unmapped)} unmapped series left out of BIDS tree (FLASH/IR-EPI/T2/"
              f"unmatched fieldmap etc.) — see below. Re-run series_classification.classify() "
              f"rules if any of these should be included:")
        for nii_path, desc in unmapped:
            print(f"    - {desc}  ({nii_path.name})")

    # append participant row
    participants_tsv = args.bids_root / "participants.tsv"
    if participants_tsv.exists():
        with open(participants_tsv, "a") as f:
            f.write(f"sub-{sub_label}\n")

    print(f"Built BIDS tree for sub-{sub_label} at {sub_dir}")


if __name__ == "__main__":
    main()
