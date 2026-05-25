"""
build_baseline_embeddings.py — CLI to precompute GAAE baseline embeddings
for all subjects, cached as parquet for fast Cox/RSF/DeepSurv sweeps.

Usage:
    # Single combo:
    python -m PROGNOSER.src.build_baseline_embeddings --combo dmn_hippo

    # All 8 combos:
    python -m PROGNOSER.src.build_baseline_embeddings --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PROGNOSER.common.embeddings import (
    extract_baseline_embeddings,
    cache_embeddings,
)


REPO_ROOT = Path('/mnt/e/fyassine/ad-early-detection')
CACHE_DIR = REPO_ROOT / 'PROGNOSER' / 'notebooks' / '_embeddings_cache_'


COMBO_TABLE = {
    "dmn":              ("__fc_dmn_sch200_flat__",                        "_dmn_correlation_matrix_z_transformed.npz"),
    "hippo":            ("__fc_hippo_tian2_flat__",                       "_hippocampus_correlation_matrix_z_transformed.npz"),
    "limbic":           ("__fc_limbic_sch200_flat__",                     "_limbic_correlation_matrix_z_transformed.npz"),
    "dan":              ("__fc_dan_sch200_flat__",                        "_dorsal_attention_correlation_matrix_z_transformed.npz"),
    "dmn_hippo":        ("__fc_dmn-hippo_sch200-tian2_flat__",            "_dmn_hippo_correlation_matrix_z_transformed.npz"),
    "dmn_limbic":       ("__fc_dmn-limbic_sch200_flat__",                 "_dmn_limbic_correlation_matrix_z_transformed.npz"),
    "dmn_limbic_hippo": ("__fc_dmn-hippo-limbic_sch200-tian2_flat__",    "_dmn_limbic_hippo_correlation_matrix_z_transformed.npz"),
    "all_combined":     ("__fc_dmn-hippo-limbic-dan_sch200-tian2_flat__", "_all_combined_correlation_matrix_z_transformed.npz"),
}


def build_one(combo: str, knn_k: int = 8, device: str = 'cuda') -> Path:
    if combo not in COMBO_TABLE:
        raise ValueError(f'Unknown combo: {combo}. Options: {list(COMBO_TABLE)}')
    data_version, file_suffix = COMBO_TABLE[combo]

    print(f'\n{"="*60}\n  {combo}  ({data_version}, knn_k={knn_k})\n{"="*60}')
    df = extract_baseline_embeddings(
        network_combo=combo,
        data_version=data_version,
        file_suffix=file_suffix,
        cohort_subjects=None,
        repo_root=REPO_ROOT,
        device=device,
        knn_k=knn_k,
    )
    out_path = CACHE_DIR / f'{combo}_baseline_embeddings.parquet'
    cache_embeddings(df, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--combo', type=str, default=None, help='Network combo name (e.g. dmn_hippo)')
    parser.add_argument('--all', action='store_true', help='Build embeddings for all 8 combos')
    parser.add_argument('--knn-k', type=int, default=8, help='kNN k for adjacency (default 8)')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    args = parser.parse_args()

    if args.all:
        for combo in COMBO_TABLE:
            try:
                build_one(combo, knn_k=args.knn_k, device=args.device)
            except FileNotFoundError as exc:
                print(f'[skip] {combo}: {exc}')
    elif args.combo:
        build_one(args.combo, knn_k=args.knn_k, device=args.device)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
