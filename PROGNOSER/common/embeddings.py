"""
embeddings.py — Extract per-subject baseline GAAE embeddings.

Loads a trained GAAE checkpoint, runs the encoder on each subject's
baseline (M0) correlation matrix, applies global mean pool over nodes,
and returns a fixed-size 64-dim feature per subject.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


REPO_ROOT_DEFAULT = Path('/mnt/e/fyassine/ad-early-detection')


def find_latest_checkpoint(network_combo: str, repo_root: Path | str = REPO_ROOT_DEFAULT) -> tuple[Path, dict]:
    """Walk CLASSIFIER/notebooks/checkpoints_gaae_{combo}/*/, return newest run dir's
    (model_path, run_config_dict). Falls back to checkpoints_gaae_{combo}_* if needed."""
    repo_root = Path(repo_root)
    base = repo_root / 'CLASSIFIER' / 'notebooks' / f'checkpoints_gaae_{network_combo}'
    if not base.is_dir():
        # Try common variants
        alt = repo_root / 'CLASSIFIER' / 'notebooks'
        candidates = sorted(alt.glob(f'checkpoints_gaae_{network_combo}*'))
        if not candidates:
            raise FileNotFoundError(
                f'No GAAE checkpoint dir found for network_combo={network_combo}.\n'
                f'Looked in: {base}\n'
                f'Run NETWORK_GAAE_RUNNER.ipynb with network_combo="{network_combo}" first.'
            )
        base = candidates[-1]

    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f'No run subdirs in {base}')

    run_dir = run_dirs[-1]
    model_path = run_dir / f'model_{run_dir.name}.pth'
    if not model_path.exists():
        # Find any .pth in the dir
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
    """Instantiate GraphAttentionAutoencoderConditioned from run_config and load weights.
    Caller must call .load_state_dict(torch.load(model_path)) afterwards."""
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
    """Same as CLASSIFIER/common/utils.py:knn_binary_adjacency_matrix_no_diag, inlined."""
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
    """Extract M-month integer from filename like 'sub-XXX_ses-01_M12_..._.npz'.
    Returns very large number if not found (so M0 sorts first)."""
    m = re.search(r'_M(\d+)_', filename)
    return int(m.group(1)) if m else 99999


def extract_baseline_embeddings(
    network_combo: str,
    data_version: str,
    file_suffix: str,
    cohort_subjects: list[str] | None = None,
    repo_root: Path | str = REPO_ROOT_DEFAULT,
    device: str = 'cuda',
    knn_k: int = 8,
) -> pd.DataFrame:
    """
    For each subject in `cohort_subjects` (or all subjects with matrix files if None),
    locate baseline (earliest M-month) graph, run GAAE encoder, mean-pool to 64-dim.

    Returns DataFrame indexed by subject_id with columns z_0..z_{latent-1}.
    Subjects without a usable scan are dropped.
    """
    from torch_geometric.nn import global_mean_pool

    repo_root = Path(repo_root)
    matrices_dir = repo_root / 'DATA' / 'DELCODE' / data_version / 'matrices'
    if not matrices_dir.is_dir():
        raise FileNotFoundError(f'No matrices dir at {matrices_dir}')

    # Build subject → list of npz files (one per visit)
    subject_files: dict[str, list[Path]] = {}
    for npz in sorted(matrices_dir.glob(f'*{file_suffix}')):
        # filename starts with 'sub-{ID}_'
        first_token = npz.name.split('_')[0]
        if not first_token.startswith('sub-'):
            continue
        sid = first_token[4:]  # strip 'sub-'
        subject_files.setdefault(sid, []).append(npz)

    if cohort_subjects is not None:
        cohort_subjects = set(map(str, cohort_subjects))
        subject_files = {s: f for s, f in subject_files.items() if s in cohort_subjects}

    if not subject_files:
        raise RuntimeError(
            f'No matrix files matched suffix {file_suffix} in {matrices_dir} '
            f'for the requested cohort.'
        )

    # Load model
    device = device if torch.cuda.is_available() else 'cpu'
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
    print(f'[embeddings] {network_combo}: latent_dim={latent_dim}, n_subjects={len(subject_files)}')

    embeddings: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    with torch.no_grad():
        for sid, files in subject_files.items():
            # Pick earliest visit (M0 if available)
            files_sorted = sorted(files, key=lambda p: _parse_visit(p.name))
            npz_path = files_sorted[0]
            try:
                feat = np.load(npz_path)['array']
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                x = torch.tensor(feat, dtype=torch.float, device=device)

                A = _knn_binary_adjacency(np.abs(feat), k=knn_k)
                edge_index, _ = dense_to_sparse(torch.tensor(A, dtype=torch.float))
                edge_index = edge_index.to(device)

                z = model.encode(x, edge_index)            # [n_nodes, latent_dim]
                batch_mask = torch.zeros(z.size(0), dtype=torch.long, device=device)
                pooled = global_mean_pool(z, batch_mask)   # [1, latent_dim]
                embeddings[sid] = pooled.cpu().numpy().flatten()
            except Exception as exc:
                skipped.append(f'{sid}: {exc}')
                continue

    if skipped:
        print(f'[embeddings] skipped {len(skipped)} subjects (first 3): {skipped[:3]}')

    df = pd.DataFrame.from_dict(embeddings, orient='index',
                                columns=[f'z_{i}' for i in range(latent_dim)])
    df.index.name = 'subject_id'
    return df


def cache_embeddings(df: pd.DataFrame, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f'[embeddings] cached {df.shape} → {out_path}')


def load_embeddings(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)
