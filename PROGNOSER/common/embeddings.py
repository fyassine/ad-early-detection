"""
embeddings.py — Extract per-subject GAAE embeddings with multiple strategies.

Strategy options:
    'baseline'  — M0 only (original behaviour)
    'last'      — latest visit within the at-risk window
    'mean'      — mean of all visit embeddings within window
    'slope'     — linear slope of embedding trajectory (visit 0 → last)
    'all_aggs'  — all four above concatenated (4 × latent_dim features)
    'sequence'  — long-format: one row per visit, for LSTM training

At-risk window: visits with month < window_end are used. For converters
window_end = event_month; for non-converters window_end = last_visit_month.
This applies the same logic to both groups.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse

# Repo root resolves from this file (PROGNOSER/common/embeddings.py -> repo root),
# overridable via the AD_REPO_ROOT env var for non-standard checkouts.
REPO_ROOT_DEFAULT = Path(os.environ.get("AD_REPO_ROOT", Path(__file__).resolve().parents[2]))

EmbeddingStrategy = Literal['baseline', 'last', 'mean', 'slope', 'all_aggs', 'sequence']


def find_latest_checkpoint(network_combo: str, repo_root: Path | str = REPO_ROOT_DEFAULT) -> tuple[Path, dict]:
    """Walk CLASSIFIER/notebooks/checkpoints_gaae_{combo}/*/, return newest (model_path, run_config)."""
    repo_root = Path(repo_root)
    base = repo_root / 'CLASSIFIER' / 'notebooks' / f'checkpoints_gaae_{network_combo}'
    if not base.is_dir():
        candidates = sorted((repo_root / 'CLASSIFIER' / 'notebooks').glob(f'checkpoints_gaae_{network_combo}*'))
        if not candidates:
            raise FileNotFoundError(
                f'No GAAE checkpoint dir found for network_combo={network_combo}.\n'
                f'Run NETWORK_GAAE_RUNNER.ipynb with network_combo="{network_combo}" first.'
            )
        base = candidates[-1]

    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f'No run subdirs in {base}')
    run_dir = run_dirs[-1]

    model_path = run_dir / f'model_{run_dir.name}.pth'
    if not model_path.exists():
        pth_files = list(run_dir.glob('*.pth'))
        if not pth_files:
            raise FileNotFoundError(f'No .pth file in {run_dir}')
        model_path = pth_files[0]

    run_config_path = run_dir / 'run_config.json'
    if not run_config_path.exists():
        raise FileNotFoundError(f'run_config.json missing in {run_dir}')
    with open(run_config_path) as f:
        run_config = json.load(f)
    return model_path, run_config


def build_gaae_model(run_config: dict, device: str = 'cpu'):
    import sys
    classifier_root = REPO_ROOT_DEFAULT / 'CLASSIFIER'
    if str(classifier_root) not in sys.path:
        sys.path.insert(0, str(classifier_root))
    from model.GAAE.models import GraphAttentionAutoencoderConditioned

    mc = run_config.get('model_config', {})
    model = GraphAttentionAutoencoderConditioned(
        in_features=int(mc.get('in_features')),
        hidden_dim=int(mc.get('hidden_size', mc.get('in_features'))),
        out_features=int(mc.get('latent_dim', 64)),
        cond_dim=int(mc.get('cond_dim', 2)),
        num_heads=int(mc.get('attention_heads', 2)),
        dropout=float(mc.get('dropout', 0.3)),
    ).to(device)
    return model


def _knn_binary_adjacency(corr: np.ndarray, k: int = 8) -> np.ndarray:
    abs_corr = np.abs(corr)
    N = abs_corr.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        row = abs_corr[i].copy()
        row[i] = -np.inf
        nn = np.argsort(-row)[:k]
        A[i, nn] = 1
    return np.maximum(A, A.T)


def _parse_visit(filename: str) -> int:
    """Extract M-month number from filename. Lower = earlier visit."""
    m = re.search(r'_M(\d+)_', filename)
    return int(m.group(1)) if m else 99999


def _encode_one(npz_path: Path, model, device: str, knn_k: int) -> np.ndarray | None:
    """Encode a single .npz correlation matrix to a 64-dim pooled embedding."""
    from torch_geometric.nn import global_mean_pool
    try:
        feat = np.load(npz_path)['array']
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(feat, dtype=torch.float, device=device)
        A = _knn_binary_adjacency(np.abs(feat), k=knn_k)
        edge_index, _ = dense_to_sparse(torch.tensor(A, dtype=torch.float))
        edge_index = edge_index.to(device)
        with torch.no_grad():
            z = model.encode(x, edge_index)
            batch_mask = torch.zeros(z.size(0), dtype=torch.long, device=device)
            pooled = global_mean_pool(z, batch_mask)
        return pooled.cpu().numpy().flatten()
    except Exception:
        return None


def extract_subject_embeddings(
    network_combo: str,
    data_version: str,
    file_suffix: str,
    cohort_subjects: list[str] | None = None,
    at_risk_windows: dict[str, int] | None = None,
    strategy: EmbeddingStrategy = 'last',
    repo_root: Path | str = REPO_ROOT_DEFAULT,
    device: str = 'cuda',
    knn_k: int = 8,
) -> pd.DataFrame:
    """
    Extract GAAE embeddings per subject using the given strategy.

    Parameters
    ----------
    at_risk_windows : dict {subject_id: window_end_months}
        Restricts which visits are used per subject (visits with month < window_end).
        Pass None to use all available visits (e.g. for inference at test time).
    strategy :
        'baseline' — M0 embedding only
        'last'     — latest visit within at-risk window
        'mean'     — mean across all visits in window
        'slope'    — linear slope (last - first embedding per dimension)
        'all_aggs' — concat of [baseline, last, mean, slope] → 4×latent_dim
        'sequence' — long-format DataFrame with one row per (subject_id, visit_month)

    Returns
    -------
    For all strategies except 'sequence':
        DataFrame indexed by subject_id with columns z_<agg>_0..z_<agg>_{latent-1}
        (or just z_0..z_{latent-1} for single-agg strategies).
    For 'sequence':
        Long-format DataFrame with columns: subject_id, visit_month, z_0..z_{latent-1}.
    """
    repo_root = Path(repo_root)
    matrices_dir = repo_root / 'DATA' / 'DELCODE' / data_version / 'matrices'
    if not matrices_dir.is_dir():
        raise FileNotFoundError(f'No matrices dir at {matrices_dir}')

    # Map subject_id → sorted list of (visit_month, Path)
    subject_visits: dict[str, list[tuple[int, Path]]] = {}
    for npz in sorted(matrices_dir.glob(f'*{file_suffix}')):
        first_token = npz.name.split('_')[0]
        if not first_token.startswith('sub-'):
            continue
        sid = first_token[4:]
        visit_m = _parse_visit(npz.name)
        subject_visits.setdefault(sid, []).append((visit_m, npz))

    for sid in subject_visits:
        subject_visits[sid].sort(key=lambda x: x[0])

    if cohort_subjects is not None:
        cohort_set = set(map(str, cohort_subjects))
        subject_visits = {s: v for s, v in subject_visits.items() if s in cohort_set}

    if not subject_visits:
        raise RuntimeError(f'No matrix files matched suffix {file_suffix} in {matrices_dir}.')

    device = device if (device == 'cpu' or not torch.cuda.is_available()) else 'cuda'
    model_path, run_config = find_latest_checkpoint(network_combo, repo_root)
    model = build_gaae_model(run_config, device=device)
    state = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(state, torch.nn.Module):
        model = state.to(device)
    else:
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
    model.eval()

    latent_dim = int(run_config['model_config'].get('latent_dim', 64))
    print(f'[embeddings] {network_combo}/{strategy}: latent={latent_dim}, n_subjects={len(subject_visits)}')

    if strategy == 'sequence':
        return _build_sequence_embeddings(subject_visits, at_risk_windows, model, device, knn_k, latent_dim)

    records: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for sid, visit_list in subject_visits.items():
        window_end = (at_risk_windows or {}).get(sid, 99999)
        # Filter to visits within at-risk window
        in_window = [(m, p) for m, p in visit_list if m < window_end]
        if not in_window:
            # Fall back to earliest available visit
            in_window = [visit_list[0]] if visit_list else []

        if not in_window:
            skipped.append(sid)
            continue

        if strategy == 'baseline':
            emb = _encode_one(in_window[0][1], model, device, knn_k)
            if emb is None:
                skipped.append(sid)
            else:
                records[sid] = emb

        elif strategy == 'last':
            emb = _encode_one(in_window[-1][1], model, device, knn_k)
            if emb is None:
                skipped.append(sid)
            else:
                records[sid] = emb

        elif strategy in ('mean', 'slope', 'all_aggs'):
            embs = []
            for _, p in in_window:
                e = _encode_one(p, model, device, knn_k)
                if e is not None:
                    embs.append(e)
            if not embs:
                skipped.append(sid)
                continue

            baseline_emb = embs[0]
            last_emb = embs[-1]
            mean_emb = np.mean(embs, axis=0)
            if len(embs) >= 2:
                slope_emb = last_emb - baseline_emb
            else:
                slope_emb = np.zeros_like(baseline_emb)

            if strategy == 'mean':
                records[sid] = mean_emb
            elif strategy == 'slope':
                records[sid] = slope_emb
            elif strategy == 'all_aggs':
                records[sid] = np.concatenate([baseline_emb, last_emb, mean_emb, slope_emb])

    if skipped:
        print(f'[embeddings] skipped {len(skipped)}: {skipped[:3]}{"..." if len(skipped) > 3 else ""}')

    if strategy == 'all_aggs':
        col_names = [f'z_baseline_{i}' for i in range(latent_dim)] \
                  + [f'z_last_{i}' for i in range(latent_dim)] \
                  + [f'z_mean_{i}' for i in range(latent_dim)] \
                  + [f'z_slope_{i}' for i in range(latent_dim)]
    else:
        col_names = [f'z_{i}' for i in range(latent_dim)]

    df = pd.DataFrame.from_dict(records, orient='index', columns=col_names)
    df.index.name = 'subject_id'
    return df


def _build_sequence_embeddings(
    subject_visits: dict[str, list[tuple[int, Path]]],
    at_risk_windows: dict[str, int] | None,
    model,
    device: str,
    knn_k: int,
    latent_dim: int,
) -> pd.DataFrame:
    """Build long-format sequence DataFrame: one row per (subject_id, visit_month)."""
    col_names = [f'z_{i}' for i in range(latent_dim)]
    rows = []
    for sid, visit_list in subject_visits.items():
        window_end = (at_risk_windows or {}).get(sid, 99999)
        in_window = [(m, p) for m, p in visit_list if m < window_end]
        if not in_window:
            in_window = [visit_list[0]] if visit_list else []
        for visit_m, p in in_window:
            emb = _encode_one(p, model, device, knn_k)
            if emb is not None:
                row = {'subject_id': sid, 'visit_month': visit_m}
                row.update(dict(zip(col_names, emb.tolist())))
                rows.append(row)
    return pd.DataFrame(rows)


def cache_embeddings(df: pd.DataFrame, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f'[embeddings] cached {df.shape} → {out_path}')


def load_embeddings(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


# Backward-compatibility shim
def extract_baseline_embeddings(
    network_combo: str,
    data_version: str,
    file_suffix: str,
    cohort_subjects: list[str] | None = None,
    repo_root: Path | str = REPO_ROOT_DEFAULT,
    device: str = 'cuda',
    knn_k: int = 8,
) -> pd.DataFrame:
    """Deprecated shim — calls extract_subject_embeddings(strategy='baseline')."""
    return extract_subject_embeddings(
        network_combo=network_combo,
        data_version=data_version,
        file_suffix=file_suffix,
        cohort_subjects=cohort_subjects,
        at_risk_windows=None,
        strategy='baseline',
        repo_root=repo_root,
        device=device,
        knn_k=knn_k,
    )
