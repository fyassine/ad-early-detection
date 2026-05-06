import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse

from .utils import dense_adjacency_from_corr, build_complete_edge_index


class GraphDatasetInMemoryFilteredDense(InMemoryDataset):
    def __init__(
        self,
        root,
        filter_csv_path=None,
        patient_info_path=None,
        separator=",",
        file_variant="z_transformed",
        use_abs=False,
        zero_diag=True,
        transform=None,
        pre_transform=None,
    ):
        self.filter_csv_path = filter_csv_path
        if self.filter_csv_path is None:
            raise ValueError("filter_csv_path is required")
        self.patient_info = None
        self.separator = separator
        self.file_variant = file_variant
        self.use_abs = use_abs
        self.zero_diag = zero_diag

        if patient_info_path:
            self.patient_info = pd.read_csv(patient_info_path, sep=self.separator)
            self.patient_info.set_index("Repseudonym", inplace=True)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _get_patient_info(self, patient_id):
        default_sex = torch.tensor(0, dtype=torch.long)
        default_age = torch.tensor(0.5, dtype=torch.float)

        if self.patient_info is None or patient_id not in self.patient_info.index:
            return default_sex, default_age

        patient_row = self.patient_info.loc[patient_id]
        if isinstance(patient_row, pd.DataFrame):
            patient_row = patient_row.iloc[0]

        sex = patient_row.get("sex")
        sex_tensor = torch.tensor(1 if sex == "m" else 0, dtype=torch.long)

        age = patient_row.get("age")
        age_tensor = torch.tensor(
            min(max(float(age) / 100.0, 0.0), 1.0) if age is not None else 0.5,
            dtype=torch.float,
        )

        return sex_tensor, age_tensor

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            corr_matrix = np.load(raw_path)["array"]
            corr_matrix = corr_matrix.astype(np.float32)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            feature_matrix = torch.tensor(corr_matrix, dtype=torch.float)

            adjacency_matrix = dense_adjacency_from_corr(
                corr_matrix,
                use_abs=self.use_abs,
                zero_diag=self.zero_diag,
            )
            adjacency_matrix = np.nan_to_num(adjacency_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

            num_nodes = adjacency_matrix.shape[0]
            edge_index = build_complete_edge_index(num_nodes)
            edge_attr = adjacency_matrix[edge_index[0], edge_index[1]]

            raw_filename = os.path.basename(raw_path)
            patient_id = raw_filename.split("_")[0].replace("sub-", "")

            data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_attr)
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
        variant_map = {
            "raw": "_whole_brain_correlation_matrix.npz",
            "z": "_whole_brain_correlation_matrix_z_transformed.npz",
            "z_transformed": "_whole_brain_correlation_matrix_z_transformed.npz",
        }
        variant_key = str(self.file_variant).lower()
        if variant_key not in variant_map:
            raise ValueError("file_variant must be one of: raw, z, z_transformed")

        all_npz_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".npz")])
        suffix = variant_map[variant_key]
        all_files = [f for f in all_npz_files if f.endswith(suffix)]

        if not os.path.exists(self.filter_csv_path):
            raise FileNotFoundError(f"Filter CSV not found at {self.filter_csv_path}")
        filter_df = pd.read_csv(self.filter_csv_path, sep=self.separator)
        if "Repseudonym" not in filter_df.columns:
            raise ValueError(
                f"Filter CSV must contain 'Repseudonym' column. Found: {filter_df.columns}"
            )
        allowed_ids = set(filter_df["Repseudonym"].astype(str))

        return [f for f in all_files if f.split("_")[0].replace("sub-", "") in allowed_ids]

    @property
    def processed_file_names(self):
        variant_tag = str(self.file_variant).lower().replace("-", "_")
        abs_tag = "abs" if self.use_abs else "signed"
        split_tag = os.path.splitext(os.path.basename(self.filter_csv_path))[0]
        return [f"data_dense_{variant_tag}_{abs_tag}_{split_tag}.pt"]
