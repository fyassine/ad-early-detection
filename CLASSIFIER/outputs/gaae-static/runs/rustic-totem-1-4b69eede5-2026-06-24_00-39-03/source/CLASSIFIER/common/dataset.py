import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse


class ClassificationDataset(InMemoryDataset):
    def __init__(self, root, adjacency_function, adjacency_args=None, transform=None,
                 pre_transform=None, patient_info_path=None, converter_list_path=None,
                 is_converter_dataset=False, separator=",", correlation_type="pearson",
                 filter_csv_path=None, subject_ids=None, file_variant="z_transformed",
                 file_suffix=None):
        self.adjacency_function = adjacency_function
        self.adjacency_args = adjacency_args or {}
        self.patient_info = None
        self.converter_ids = set()
        self.is_converter_dataset = is_converter_dataset
        self.correlation_type = correlation_type
        self.separator = separator
        self.filter_csv_path = filter_csv_path
        self.file_variant = str(file_variant).lower()
        self.file_suffix = file_suffix  # overrides variant/correlation_type matching when set
        self.allowed_ids = set()

        if subject_ids is not None:
            self.allowed_ids.update(str(s) for s in subject_ids)

        if self.filter_csv_path is not None:
            if not os.path.exists(self.filter_csv_path):
                raise FileNotFoundError(f"Filter CSV not found at {self.filter_csv_path}")
            filter_df = pd.read_csv(self.filter_csv_path, sep=self.separator)
            if "Pseudonym" not in filter_df.columns:
                raise ValueError(
                    f"Filter CSV must contain 'Pseudonym' column. Found: {list(filter_df.columns)}"
                )
            self.allowed_ids.update(filter_df["Pseudonym"].astype(str))

        if patient_info_path:
            self.patient_info = pd.read_csv(patient_info_path, sep=self.separator)
            id_col = None
            for candidate in ["Pseudonym", "ID"]:
                if candidate in self.patient_info.columns:
                    id_col = candidate
                    break
            if id_col is None:
                raise ValueError(
                    "Patient info CSV must contain one of: Pseudonym, ID. "
                    f"Found: {list(self.patient_info.columns)}"
                )
            self.patient_info.set_index(id_col, inplace=True)

        if converter_list_path and os.path.exists(converter_list_path):
            converter_df = pd.read_csv(converter_list_path)
            if 'ID' in converter_df.columns:
                self.converter_ids = set(converter_df['ID'].astype(str))
            elif 'Pseudonym' in converter_df.columns:
                self.converter_ids = set(converter_df['Pseudonym'].astype(str))

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        # Support both standard PyG layout (<root>/raw) and flat layout (<root>).
        default_raw_dir = super().raw_dir
        if os.path.isdir(default_raw_dir):
            return default_raw_dir
        return self.root

    def _resolve_candidate_files(self, all_files):
        if self.file_suffix:
            return [f for f in all_files if f.endswith(self.file_suffix)]

        variant_suffixes = {
            "raw": [
                "_dmn_correlation_matrix.npz",
                "_whole_brain_correlation_matrix.npz",
            ],
            "z_transformed": [
                "_dmn_correlation_matrix_z_transformed.npz",
                "_whole_brain_correlation_matrix_z_transformed.npz",
            ],
            "both": [
                "_dmn_correlation_matrix.npz",
                "_dmn_correlation_matrix_z_transformed.npz",
                "_whole_brain_correlation_matrix.npz",
                "_whole_brain_correlation_matrix_z_transformed.npz",
            ],
        }

        if self.file_variant in variant_suffixes:
            suffixes = tuple(variant_suffixes[self.file_variant])
            return [f for f in all_files if f.endswith(suffixes)]

        correlation_pattern = f"{self.correlation_type}_correlation_matrix"
        filtered_files = [f for f in all_files if correlation_pattern in f]
        return filtered_files if filtered_files else all_files

    @property
    def raw_file_names(self):
        all_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.npz')])
        candidate_files = self._resolve_candidate_files(all_files)

        if self.allowed_ids:
            candidate_files = [
                f for f in candidate_files
                if f.split('_')[0].replace('sub-', '') in self.allowed_ids
            ]

        if not candidate_files:
            raise FileNotFoundError(
                "No matching .npz files found for the dataset settings. "
                f"root={self.raw_dir}, file_variant={self.file_variant}, "
                f"correlation_type={self.correlation_type}, "
                f"num_allowed_ids={len(self.allowed_ids)}"
            )

        return candidate_files

    @property
    def processed_file_names(self):
        suffix = '_converter' if self.is_converter_dataset else '_nonconverter'
        variant_tag = self.file_variant.replace('-', '_')
        split_tag = "all"
        if self.filter_csv_path:
            split_tag = os.path.splitext(os.path.basename(self.filter_csv_path))[0]
        return [f'data_classification_{variant_tag}_{split_tag}{suffix}.pt']

    def process(self):
        data_list = []
        for _idx, raw_path in enumerate(self.raw_paths):
            data_npz = np.load(raw_path)
            if 'array' in data_npz:
                feature_matrix = data_npz['array']
            else:
                key = list(data_npz.keys())[0]
                feature_matrix = data_npz[key]
            feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)
            abs_feature_matrix = torch.abs(feature_matrix.clone())

            adjacency_matrix = self.adjacency_function(abs_feature_matrix, **self.adjacency_args)
            if isinstance(adjacency_matrix, np.ndarray):
                adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

            edge_index, _ = dense_to_sparse(adjacency_matrix)
            raw_filename = os.path.basename(raw_path)
            patient_id = raw_filename.split('_')[0].replace('sub-', '')

            data = Data(x=feature_matrix, edge_index=edge_index)
            data.patient_id = patient_id

            if self.patient_info is not None and patient_id in self.patient_info.index:
                patient_row = self.patient_info.loc[patient_id]
                if isinstance(patient_row, pd.DataFrame):
                    patient_row = patient_row.iloc[0]

                sex_tensor = torch.tensor(1 if patient_row.get("sex", "f") == "m" else 0, dtype=torch.long)
                age_value = patient_row.get("age", 50.0)
                age_tensor = torch.tensor(min(max(age_value / 100.0, 0.0), 1.0), dtype=torch.float)
            else:
                sex_tensor = torch.tensor(0, dtype=torch.long)
                age_tensor = torch.tensor(0.5, dtype=torch.float)

            data.patient_sex = sex_tensor
            data.patient_age = age_tensor

            if self.is_converter_dataset:
                is_converter = 1
            else:
                is_converter = 0

            data.is_converter = torch.tensor(is_converter, dtype=torch.float)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_labels(self):
        labels = []
        for i in range(len(self)):
            data = self.get(i)
            labels.append(data.is_converter.item())
        return labels


class CombinedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, converter_dataset, non_converter_dataset):
        self.converter_dataset = converter_dataset
        self.non_converter_dataset = non_converter_dataset
        self.converter_len = len(converter_dataset)
        self.non_converter_len = len(non_converter_dataset)

    def __len__(self):
        return self.converter_len + self.non_converter_len

    def __getitem__(self, idx):
        if idx < self.converter_len:
            return self.converter_dataset[idx]
        else:
            return self.non_converter_dataset[idx - self.converter_len]

    def get_labels(self):
        labels = []
        for i in range(self.converter_len):
            data = self.converter_dataset[i]
            labels.append(data.is_converter.item())
        for i in range(self.non_converter_len):
            data = self.non_converter_dataset[i]
            labels.append(data.is_converter.item())
        return labels
