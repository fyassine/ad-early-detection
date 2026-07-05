import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse

from .utils import knn_binary_adjacency_matrix_no_diag


class GraphDatasetInMemoryFiltered(InMemoryDataset):
    def __init__(
        self,
        root,
        adjacency_function=knn_binary_adjacency_matrix_no_diag,
        adjacency_args={'k': 16},
        filter_csv_path=None,
        patient_info_path=None,
        separator=",",
        file_variant="z_transformed",
        file_suffix=None,
        transform=None,
        pre_transform=None,
    ):
        self.adjacency_function = adjacency_function
        self.adjacency_args = adjacency_args or {}
        self.filter_csv_path = filter_csv_path
        if self.filter_csv_path is None:
            raise ValueError("filter_csv_path is required")
        self.patient_info = None
        self.separator = separator
        self.file_variant = file_variant
        self.file_suffix = file_suffix  # overrides variant lookup when set
        if patient_info_path:
            self.patient_info = pd.read_csv(patient_info_path, sep=self.separator)
            self.patient_info.set_index("Pseudonym", inplace=True)
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        # Support both standard PyG layout (<root>/raw) and flat layout (<root>).
        default_raw_dir = super().raw_dir
        if os.path.isdir(default_raw_dir):
            return default_raw_dir
        return self.root

    def _get_patient_info(self, patient_id):
        default_sex = torch.tensor(0, dtype=torch.long)  # 0 for female
        default_age = torch.tensor(0.5, dtype=torch.float)  # normalized 50 years
        
        if self.patient_info is None or patient_id not in self.patient_info.index:
            logging.warning(f"Patient {patient_id} not found in metadata; using defaults.")
            return default_sex, default_age
        
        patient_row = self.patient_info.loc[patient_id]
        if isinstance(patient_row, pd.DataFrame):
            patient_row = patient_row.iloc[0]
        
        sex = patient_row.get("sex")
        sex_tensor = torch.tensor(1 if sex == "m" else 0, dtype=torch.long)
        
        age = patient_row.get("age")
        age_tensor = torch.tensor(min(max(float(age) / 100.0, 0.0), 1.0) if age is not None else 0.5, dtype=torch.float)
        
        return sex_tensor, age_tensor

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            feature_matrix = np.load(raw_path)['array']
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)
            abs_feature_matrix = torch.abs(torch.clone(feature_matrix))

            adjacency_matrix = self.adjacency_function(abs_feature_matrix, **self.adjacency_args)
            if isinstance(adjacency_matrix, np.ndarray):
                adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

            edge_index, edge_weight = dense_to_sparse(adjacency_matrix)
            raw_filename = os.path.basename(raw_path)
            patient_id = raw_filename.split('_')[0].replace('sub-', '')

            data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_weight)
            data.patient_id = patient_id

            sex_tensor, age_tensor = self._get_patient_info(patient_id)
            data.patient_sex = sex_tensor
            data.patient_age = age_tensor
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def raw_file_names(self):
        all_npz_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.npz')])

        if self.file_suffix:
            all_files = [f for f in all_npz_files if f.endswith(self.file_suffix)]
        else:
            variant_map = {
                "raw": "_whole_brain_correlation_matrix.npz",
                "z_transformed": "_whole_brain_correlation_matrix_z_transformed.npz",
            }
            variant_key = str(self.file_variant).lower()
            if variant_key not in variant_map:
                raise ValueError("file_variant must be one of: raw, z_transformed")
            all_files = [f for f in all_npz_files if f.endswith(variant_map[variant_key])]

        if not os.path.exists(self.filter_csv_path):
            raise FileNotFoundError(f"Filter CSV not found at {self.filter_csv_path}")
        filter_df = pd.read_csv(self.filter_csv_path, sep=self.separator)
        if 'Pseudonym' not in filter_df.columns:
            raise ValueError(f"Filter CSV must contain 'Pseudonym' column. Found: {filter_df.columns}")
        allowed_ids = set(filter_df['Pseudonym'].astype(str))

        return [f for f in all_files if f.split('_')[0].replace('sub-', '') in allowed_ids]

    @property
    def processed_file_names(self):
        variant_tag = str(self.file_variant).lower().replace("-", "_")
        split_tag = os.path.splitext(os.path.basename(self.filter_csv_path))[0]
        fn_name = getattr(self.adjacency_function, "__name__", "adj")
        args_str = json.dumps(self.adjacency_args, sort_keys=True)
        graph_tag = hashlib.md5(f"{fn_name}_{args_str}".encode()).hexdigest()[:8]
        with open(self.filter_csv_path, "rb") as _f:
            csv_hash = hashlib.md5(_f.read()).hexdigest()[:8]
        return [f"data_filtered_{variant_tag}_{split_tag}_{csv_hash}_graph{graph_tag}.pt"]


class GraphDMNDatasetInMemoryFiltered(GraphDatasetInMemoryFiltered):
    """Dataset for DMN-only correlation matrices (46 nodes).

    Identical to GraphDatasetInMemoryFiltered except that it defaults to
    DMN file suffixes. Prefer GraphDatasetInMemoryFiltered with file_suffix
    for new network experiments.
    """

    @property
    def raw_file_names(self):
        all_npz_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.npz')])

        if self.file_suffix:
            all_files = [f for f in all_npz_files if f.endswith(self.file_suffix)]
        else:
            variant_map = {
                "raw": "_dmn_correlation_matrix.npz",
                "z_transformed": "_dmn_correlation_matrix_z_transformed.npz",
            }
            variant_key = str(self.file_variant).lower()
            if variant_key not in variant_map:
                raise ValueError("file_variant must be one of: raw, z_transformed")
            all_files = [f for f in all_npz_files if f.endswith(variant_map[variant_key])]

        if not os.path.exists(self.filter_csv_path):
            raise FileNotFoundError(f"Filter CSV not found at {self.filter_csv_path}")
        filter_df = pd.read_csv(self.filter_csv_path, sep=self.separator)
        if 'Pseudonym' not in filter_df.columns:
            raise ValueError(f"Filter CSV must contain 'Pseudonym' column. Found: {filter_df.columns}")
        allowed_ids = set(filter_df['Pseudonym'].astype(str))

        return [f for f in all_files if f.split('_')[0].replace('sub-', '') in allowed_ids]
