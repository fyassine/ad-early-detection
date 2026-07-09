#!/usr/bin/env python3
"""
split_batch_zips.py
====================
One-time repair for zips saved by the pre-fix save_dicom_zip(), which named
a whole LONI batch download (--batch-size > 1 packs several images into one
"1-CLICK" archive) after only the first image_id/subject_id it found in the
archive, silently hiding the other bundled images under that single
misleading filename.

Splits every zip in --zip-dir into one {subject_id}_{image_id}.zip per image,
identified from each member's internal LONI path
(ADNI/{subject_id}/.../I{image_id}/*.dcm) -- the same logic the fixed
save_dicom_zip() now applies at download time. Already-correct single-image
zips are left untouched, so this is safe to re-run over an entire zip
directory.

Usage
-----
    python split_batch_zips.py --zip-dir ../../__smri_dicom_zips_flat__
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def member_ids(name: str) -> tuple[str | None, int | None]:
    """(subject_id, image_id) encoded in one zip member's internal LONI path
    (ADNI/{subject_id}/.../I{image_id}/*.dcm), or (None, None) if absent."""
    subject_id = None
    image_id = None
    for part in Path(name).parts:
        if image_id is None:
            m = re.match(r"^I(\d+)$", part)
            if m:
                image_id = int(m.group(1))
        if subject_id is None:
            m = re.match(r"^(\d{3}_S_\d{4})$", part)
            if m:
                subject_id = m.group(1)
    return subject_id, image_id


def split_zip(zip_path: Path) -> list[Path]:
    """
    Split zip_path into one {subject_id}_{image_id}.zip per bundled image.
    New files are written under temp names first -- a bundled zip's final
    name for one of its images very often collides with zip_path's own
    (misleading) name, so writing directly would partially overwrite the
    archive still being read. zip_path is only removed, and the temp files
    only renamed into place, once every image has been fully extracted.

    Returns the final destination paths written (empty if zip_path already
    contains exactly one image -- nothing to split).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members_by_image: dict[tuple[str, int], list[str]] = {}
        for name in zf.namelist():
            subject_id, image_id = member_ids(name)
            if subject_id is None or image_id is None:
                continue
            members_by_image.setdefault((subject_id, image_id), []).append(name)

        if len(members_by_image) <= 1:
            return []

        tmp_to_final: list[tuple[Path, Path]] = []
        for (subject_id, image_id), member_names in members_by_image.items():
            final = zip_path.parent / f"{subject_id}_{image_id}.zip"
            tmp = zip_path.parent / f".split_{subject_id}_{image_id}.zip.tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out_zf:
                for name in member_names:
                    out_zf.writestr(name, zf.read(name))
            tmp_to_final.append((tmp, final))

    zip_path.unlink()
    written: list[Path] = []
    for tmp, final in tmp_to_final:
        if final.exists():
            tmp.unlink()
            continue
        tmp.rename(final)
        written.append(final)
    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--zip-dir", required=True)
    args = p.parse_args()

    zip_dir = Path(args.zip_dir)
    zips = sorted(zip_dir.glob("*.zip"))
    print(f"Scanning {len(zips)} zip(s) in {zip_dir}...")

    split_count = 0
    image_count = 0
    for zp in zips:
        written = split_zip(zp)
        if written:
            split_count += 1
            image_count += len(written)
            print(f"  {zp.name}: split into {len(written)} image(s): {[w.name for w in written]}")

    print(f"Done: {split_count} bundled zip(s) split into {image_count} per-image zip(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
