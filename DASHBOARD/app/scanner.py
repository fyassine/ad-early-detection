"""
scanner.py — Discover and scan data folders for fMRI / parcellated files.

Dataset-agnostic. Uses a single os.walk pass for performance on large data trees.
Discovers:
  - CSV files (potential metadata)
  - Directories containing .nii.gz or .npz files
  - Subject IDs extracted from filenames or paths (sub-XXXXXXX pattern)
  - Scan format details (matrix dimensions, file sizes)
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Optional


# Pattern to extract subject ID (sub-XXXXXXXXX) from paths/filenames
SUBJECT_PATTERN = re.compile(r"sub-([a-f0-9]+)", re.IGNORECASE)
VISIT_PATTERN = re.compile(r"_(M\d+)_", re.IGNORECASE)


def discover_csvs(data_root: str) -> list[dict]:
    """
    Recursively find all CSV files under data_root.
    Returns list of dicts with path info.
    """
    csvs = []
    for dirpath, _, filenames in os.walk(data_root):
        for fn in sorted(filenames):
            if fn.lower().endswith(".csv"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, data_root)
                csvs.append({
                    "path": rel,
                    "abs_path": full,
                    "name": fn,
                    "size_bytes": os.path.getsize(full),
                })
    return csvs


def discover_scan_folders(data_root: str) -> list[dict]:
    """
    Find directories containing scan files (.nii.gz or .npz).
    Single os.walk pass, grouped by top-level folder structure.
    Includes format information for each scan type.
    """
    # Collect all scan files in one pass
    scan_files: list[dict] = []
    for dirpath, _, filenames in os.walk(data_root):
        for fn in filenames:
            ext = None
            if fn.endswith(".nii.gz"):
                ext = "nii.gz"
            elif fn.endswith(".npz"):
                ext = "npz"
            else:
                continue

            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, data_root)
            subj = _extract_subject_from_path(rel)

            scan_files.append({
                "rel_path": rel,
                "abs_path": full,
                "ext": ext,
                "subject": subj,
                "filename": fn,
            })

    if not scan_files:
        return []

    # Group by meaningful folder prefix (up to 3 levels deep)
    groups: dict[str, list[dict]] = {}
    for sf in scan_files:
        parts = sf["rel_path"].split(os.sep)
        dir_parts = parts[:-1]
        key_depth = min(3, len(dir_parts))
        key = os.sep.join(dir_parts[:key_depth]) if key_depth > 0 else dir_parts[0] if dir_parts else "."
        groups.setdefault(key, []).append(sf)

    # Build result with format info
    results = []
    for group_path, files in sorted(groups.items()):
        exts = set(f["ext"] for f in files)
        if len(exts) > 1:
            file_type = "mixed"
        else:
            file_type = exts.pop()

        subjects = sorted(set(f["subject"] for f in files if f["subject"]))

        # Extract visit distribution
        visits = {}
        for f in files:
            vm = VISIT_PATTERN.search(f["filename"])
            v = vm.group(1) if vm else "unknown"
            visits[v] = visits.get(v, 0) + 1

        # Get format details by sampling one file
        format_info = _get_format_info(files[0], file_type)

        results.append({
            "path": group_path,
            "abs_path": os.path.join(data_root, group_path),
            "file_type": file_type,
            "scan_count": len(files),
            "subject_count": len(subjects),
            "visit_distribution": dict(sorted(visits.items())),
            "format_info": format_info,
        })

    return results


def _get_format_info(sample_file: dict, file_type: str) -> dict:
    """
    Get format details by sampling one file.
    For .npz: matrix shape, key name.
    For .nii.gz: file size.
    """
    info = {
        "type": file_type,
        "sample_file": sample_file["filename"],
        "sample_size_mb": round(os.path.getsize(sample_file["abs_path"]) / (1024 * 1024), 2),
    }

    if file_type == "npz":
        try:
            data = np.load(sample_file["abs_path"])
            key = list(data.keys())[0]
            arr = data[key]
            info["matrix_shape"] = list(arr.shape)
            info["array_key"] = key
            info["dtype"] = str(arr.dtype)
            if len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]:
                info["description"] = f"{arr.shape[0]}×{arr.shape[0]} correlation matrix"
                if arr.shape[0] <= 50:
                    info["parcellation"] = "DMN-only (Schaefer)"
                elif arr.shape[0] == 200:
                    info["parcellation"] = "Schaefer 200 parcels"
                elif arr.shape[0] == 400:
                    info["parcellation"] = "Schaefer 400 parcels"
                else:
                    info["parcellation"] = f"{arr.shape[0]} ROIs"
        except Exception as e:
            info["error"] = str(e)
    elif file_type == "nii.gz":
        info["description"] = "Preprocessed fMRI volume (NIfTI)"

    return info


def scan_selected_folders(data_root: str, folder_paths: list[str]) -> dict:
    """
    Scan the selected folders, aggregate all scans.
    Returns summary with per-subject scan counts, file type, totals, and format info.
    """
    all_files = []
    subject_scan_counts: dict[str, int] = {}
    subject_visits: dict[str, list[str]] = {}
    detected_types = set()
    format_info = None

    for folder_rel in folder_paths:
        folder = os.path.join(data_root, folder_rel)
        if not os.path.isdir(folder):
            continue

        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                ext = None
                if fn.endswith(".nii.gz"):
                    ext = "nii.gz"
                elif fn.endswith(".npz"):
                    ext = "npz"
                else:
                    continue

                detected_types.add(ext)
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, data_root)
                subj = _extract_subject_from_path(rel)

                # Extract visit
                vm = VISIT_PATTERN.search(fn)
                visit = vm.group(1) if vm else "unknown"

                all_files.append({
                    "file": rel,
                    "subject": subj,
                    "type": ext,
                    "visit": visit,
                })

                if subj:
                    subject_scan_counts[subj] = subject_scan_counts.get(subj, 0) + 1
                    subject_visits.setdefault(subj, []).append(visit)

                # Sample format info from first file
                if format_info is None:
                    format_info = _get_format_info(
                        {"abs_path": full, "filename": fn}, ext
                    )

    if len(detected_types) > 1:
        file_type = "mixed"
    elif detected_types:
        file_type = detected_types.pop()
    else:
        file_type = "none"

    # Subjects with multiple visits (longitudinal)
    multi_visit_subjects = {
        k: sorted(set(v)) for k, v in subject_visits.items() if len(set(v)) > 1
    }

    return {
        "total_scans": len(all_files),
        "total_subjects": len(subject_scan_counts),
        "file_type": file_type,
        "format_info": format_info,
        "subject_scan_counts": subject_scan_counts,
        "scans_per_subject_distribution": _distribution(subject_scan_counts),
        "longitudinal_subjects": len(multi_visit_subjects),
        "visit_distribution": _visit_distribution(all_files),
    }


def _visit_distribution(files: list[dict]) -> dict[str, int]:
    """Count files per visit."""
    dist: dict[str, int] = {}
    for f in files:
        v = f.get("visit", "unknown")
        dist[v] = dist.get(v, 0) + 1
    return dict(sorted(dist.items()))


def _extract_subject_from_path(rel_path: str) -> Optional[str]:
    """Extract subject ID from a relative file path."""
    match = SUBJECT_PATTERN.search(rel_path)
    return match.group(1) if match else None


def _distribution(counts: dict[str, int]) -> dict[str, int]:
    """Convert subject->count to count_value->frequency."""
    dist: dict[str, int] = {}
    for count in counts.values():
        key = str(count)
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: int(x[0])))
