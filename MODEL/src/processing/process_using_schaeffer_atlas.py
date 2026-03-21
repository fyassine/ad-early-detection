from pathlib import Path
from typing import Any, cast

import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


BASELINE_ROOT = Path("/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/fmri/baseline")
SUBJECT_GLOB = "sub-*"
OUTPUT_RAW_SUFFIX = "_whole_brain_correlation_matrix.npz"
OUTPUT_Z_SUFFIX = "_whole_brain_correlation_matrix_z_transformed.npz"
OVERWRITE_EXISTING = False


def build_masker() -> NiftiLabelsMasker:
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    return NiftiLabelsMasker(
        labels_img=schaefer.maps,
        standardize=cast(Any, "zscore_sample"),
    )


def strip_nifti_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


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


def should_skip_file(bold_path: Path) -> bool:
    raw_output_path, z_output_path = build_output_paths(bold_path)
    return raw_output_path.exists() and z_output_path.exists()


def build_progress_iterator(bold_files: list[Path]) -> Any:
    if tqdm is None:
        return bold_files
    return tqdm(bold_files, total=len(bold_files), unit="file", dynamic_ncols=True)


def build_output_paths(bold_path: Path) -> tuple[Path, Path]:
    prefix = strip_nifti_suffix(bold_path.name)
    return (
        bold_path.parent / f"{prefix}{OUTPUT_RAW_SUFFIX}",
        bold_path.parent / f"{prefix}{OUTPUT_Z_SUFFIX}",
    )


def compute_connectivity_matrices(
    bold_path: Path,
    masker: NiftiLabelsMasker,
    correlation_measure: ConnectivityMeasure,
) -> tuple[np.ndarray, np.ndarray]:
    time_series = masker.fit_transform(str(bold_path))
    corr_matrix = correlation_measure.fit_transform([time_series])[0]

    clipped_corr_matrix = np.clip(corr_matrix, -0.999999, 0.999999)
    z_matrix = np.arctanh(clipped_corr_matrix)
    np.fill_diagonal(z_matrix, 0.0)
    z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return corr_matrix, z_matrix


def process_file(
    bold_path: Path,
    masker: NiftiLabelsMasker,
    correlation_measure: ConnectivityMeasure,
) -> str:
    raw_output_path, z_output_path = build_output_paths(bold_path)
    if not OVERWRITE_EXISTING and should_skip_file(bold_path):
        return f"SKIP {bold_path}"

    corr_matrix, z_matrix = compute_connectivity_matrices(
        bold_path=bold_path,
        masker=masker,
        correlation_measure=correlation_measure,
    )

    np.savez_compressed(raw_output_path, array=corr_matrix)
    np.savez_compressed(z_output_path, array=z_matrix)
    return f"DONE {bold_path} -> {raw_output_path.name}, {z_output_path.name}"


def main() -> None:
    if not BASELINE_ROOT.exists():
        raise FileNotFoundError(f"Baseline directory not found: {BASELINE_ROOT}")

    bold_files = list(iter_bold_files(BASELINE_ROOT))
    if not bold_files:
        print(f"No rest-state BOLD files found under {BASELINE_ROOT}")
        return

    masker = build_masker()
    correlation_measure = ConnectivityMeasure(kind="correlation")

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    progress: Any = build_progress_iterator(bold_files)

    for index, bold_path in enumerate(progress, start=1):
        try:
            message = process_file(
                bold_path=bold_path,
                masker=masker,
                correlation_measure=correlation_measure,
            )
            if message.startswith("SKIP"):
                skipped_count += 1
            else:
                processed_count += 1

            if tqdm is None:
                remaining = len(bold_files) - index
                print(
                    f"[{index}/{len(bold_files)}] {message} "
                    f"(processed={processed_count}, skipped={skipped_count}, failed={failed_count}, left={remaining})"
                )
            else:
                progress.set_postfix(
                    processed=processed_count,
                    skipped=skipped_count,
                    failed=failed_count,
                    left=len(bold_files) - index,
                )
        except Exception as exc:
            failed_count += 1
            if tqdm is None:
                remaining = len(bold_files) - index
                print(
                    f"[{index}/{len(bold_files)}] ERROR {bold_path}: {exc} "
                    f"(processed={processed_count}, skipped={skipped_count}, failed={failed_count}, left={remaining})"
                )
            else:
                progress.write(f"ERROR {bold_path}: {exc}")
                progress.set_postfix(
                    processed=processed_count,
                    skipped=skipped_count,
                    failed=failed_count,
                    left=len(bold_files) - index,
                )

    if tqdm is not None:
        progress.close()

    print(
        "Finished DELCODE baseline processing: "
        f"processed={processed_count}, skipped={skipped_count}, failed={failed_count}"
    )


if __name__ == "__main__":
    main()