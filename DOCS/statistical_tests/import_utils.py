from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

COHORTS = [
    {'name': 'Healthy', 'key': 'healthy'},
    {'name': 'AD', 'key': 'ad'},
    {'name': 'MCI', 'key': 'mci'},
    {'name': 'Converters', 'key': 'converter'}
]


def get_cohort_raw_path(cohort_key: str, config: Any) -> Path:
    path_map = {
        'healthy': config.DELCODE_HEALTHY_GRAPH_DATA,
        'ad': config.DELCODE_AD_GRAPH_DATA,
        'mci': config.DELCODE_MCI_SCD_GRAPH_DATA,
        'converter': config.DELCODE_CONVERTER_GRAPH_DATA
    }
    return Path(path_map[cohort_key]) / 'raw'


def load_correlation_matrices(raw_dir: Path, correlation_type: str = 'pearson') -> Tuple[np.ndarray, np.ndarray]:
    pattern = f"*_{correlation_type}_correlation_matrix.npz"
    files = sorted(raw_dir.glob(pattern))

    if not files:
        print('  (no files found)')
        return np.array([]), np.array([])

    matrices = []
    ids = []

    for file_path in files:
        with np.load(file_path) as data:
            matrix = data.get('array', data.get('adjacency'))
            if matrix is None:
                continue
            matrices.append(matrix)

        subj_id = file_path.name.split('_')[0].replace('sub-', '')
        ids.append(subj_id)

    return np.array(matrices), np.array(ids)


def find_missing_ids(patient_df, cohorts) -> list:

    fmri_ids = set()
    for cohort in cohorts:
        current_ids = [str(x) for x in cohorts[cohort]['ids']]
        fmri_ids.update(current_ids)

    csv_ids = set(patient_df['Repseudonym'])

    missing_ids = list(csv_ids - fmri_ids)

    return missing_ids


def extract_upper_triangular(matrices):
    stack = np.asanyarray(matrices)
    if stack.size == 0:
        return np.array([]), None

    rows, cols = np.triu_indices(stack.shape[1], k=1)
    return stack[:, rows, cols], (rows, cols)

def load_and_process_cohorts(config: Any, correlation_type: str = 'pearson', remove_nan: bool = True) -> Dict[str, Any]:
    cohort_data = {}
    global_triu_idx = None  # We only need to store the indices once

    for cohort in COHORTS:
        name, key = cohort['name'], cohort['key']
        path = get_cohort_raw_path(key, config)

        matrices, ids = load_correlation_matrices(path, correlation_type)

        if remove_nan and matrices.size > 0:
            has_nan = np.isnan(matrices).any(axis=(1, 2))
            n_removed = np.sum(has_nan)

            if n_removed > 0:
                print(f"{name}: Removing {n_removed} subjects with NaN values.")
                matrices = matrices[~has_nan]
                ids = ids[~has_nan]

        edges, triu_idx = extract_upper_triangular(matrices)

        if global_triu_idx is None and triu_idx is not None:
            global_triu_idx = triu_idx

        cohort_data[key] = {
            'name': name,
            'ids': ids,  # Now guaranteed to be clean strings
            'matrices': matrices,  # Original 3D matrices
            'edges': edges  # New 2D array (Subjects x Edges)
        }

        shape = matrices.shape if matrices.size > 0 else "Empty"
        print(f"{name}: {len(matrices)} subjects, Matrix shape: {shape}, Edges shape: {edges.shape}")


    if cohort_data and global_triu_idx is not None:
        first_key = list(cohort_data.keys())[0]
        cohort_data[first_key]['triu_idx'] = global_triu_idx

    return cohort_data