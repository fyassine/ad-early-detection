"""
build_subject_embeddings.py — CLI to precompute GAAE embeddings for all subjects
using a given strategy, cached as parquet for fast Cox/RSF/LSTM sweeps.

Usage:
    # Single combo, last-visit strategy:
    python -m PROGNOSER.src.build_subject_embeddings --combo dmn_hippo --strategy last

    # All 8 combos, all aggregations:
    python -m PROGNOSER.src.build_subject_embeddings --all --strategy all_aggs

    # Sequence embeddings for LSTM:
    python -m PROGNOSER.src.build_subject_embeddings --all --strategy sequence

Strategies:
    baseline  — M0 only (backward-compatible)
    last      — latest visit in at-risk window
    mean      — mean across window visits
    slope     — embedding direction (last - baseline)
    all_aggs  — concat of baseline+last+mean+slope (4×latent_dim features)
    sequence  — long-format, one row per visit (for LSTM)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PROGNOSER.common.embeddings import extract_subject_embeddings, cache_embeddings
from PROGNOSER.common.survival_table import build_survival_table


REPO_ROOT = Path('/mnt/e/fyassine/ad-early-detection')
CACHE_DIR = REPO_ROOT / 'PROGNOSER' / 'notebooks' / '_embeddings_cache_'
COHORTS_CSV = REPO_ROOT / 'DATA' / 'DELCODE' / '__v3__' / 'metadata' / 'cohorts.csv'
SPLITS_DIR = REPO_ROOT / 'DATA' / 'DELCODE' / '__v3__' / 'metadata' / 'splits_gec'

COMBO_TABLE = {
    "dmn":              ("__v4__",  "_dmn_correlation_matrix_z_transformed.npz"),
    "hippo":            ("__v5__",  "_hippocampus_correlation_matrix_z_transformed.npz"),
    "limbic":           ("__v6__",  "_limbic_correlation_matrix_z_transformed.npz"),
    "dan":              ("__v7__",  "_dorsal_attention_correlation_matrix_z_transformed.npz"),
    "dmn_hippo":        ("__v8__",  "_dmn_hippo_correlation_matrix_z_transformed.npz"),
    "dmn_limbic":       ("__v9__",  "_dmn_limbic_correlation_matrix_z_transformed.npz"),
    "dmn_limbic_hippo": ("__v10__", "_dmn_limbic_hippo_correlation_matrix_z_transformed.npz"),
    "all_combined":     ("__v11__", "_all_combined_correlation_matrix_z_transformed.npz"),
}


def _build_at_risk_windows() -> dict[str, int]:
    """Build per-subject at-risk window from full cohort (all splits)."""
    table = build_survival_table(str(COHORTS_CSV))
    return {str(row['subject_id']): int(row['duration']) for _, row in table.iterrows()}


def build_one(combo: str, strategy: str, knn_k: int = 8, device: str = 'cuda') -> Path:
    if combo not in COMBO_TABLE:
        raise ValueError(f'Unknown combo: {combo}. Options: {list(COMBO_TABLE)}')
    data_version, file_suffix = COMBO_TABLE[combo]

    print(f'\n{"="*60}\n  {combo} | {strategy} ({data_version}, knn_k={knn_k})\n{"="*60}')

    at_risk_windows = _build_at_risk_windows()

    df = extract_subject_embeddings(
        network_combo=combo,
        data_version=data_version,
        file_suffix=file_suffix,
        cohort_subjects=None,
        at_risk_windows=at_risk_windows,
        strategy=strategy,
        repo_root=REPO_ROOT,
        device=device,
        knn_k=knn_k,
    )
    out_path = CACHE_DIR / f'{combo}_{strategy}_embeddings.parquet'
    cache_embeddings(df, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--combo', type=str, default=None, help='Network combo name')
    parser.add_argument('--all', action='store_true', help='Build embeddings for all 8 combos')
    parser.add_argument('--strategy', type=str, default='last',
                        choices=['baseline', 'last', 'mean', 'slope', 'all_aggs', 'sequence'],
                        help='Embedding aggregation strategy')
    parser.add_argument('--knn-k', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.all:
        for combo in COMBO_TABLE:
            try:
                build_one(combo, strategy=args.strategy, knn_k=args.knn_k, device=args.device)
            except FileNotFoundError as exc:
                print(f'[skip] {combo}: {exc}')
    elif args.combo:
        build_one(args.combo, strategy=args.strategy, knn_k=args.knn_k, device=args.device)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
