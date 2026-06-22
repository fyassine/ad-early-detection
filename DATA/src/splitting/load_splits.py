"""
Utility functions to load data splits and convert them to indices for datasets.
Use these to filter datasets based on the pre-defined train/val/test patient splits.
"""
from pathlib import Path

import pandas as pd

SPLITS_ROOT = Path(__file__).parents[3] / "DATA" / "DELCODE" / "SPLITS"
_MODEL_DIRS = {"pretrain": SPLITS_ROOT / "pretrain", "downstream": SPLITS_ROOT / "downstream"}

_SPLIT_FILES = [("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")]


def splits_dir(model: str) -> Path:
    """Return the directory holding {train,val,test}.csv for a model ('pretrain' | 'downstream')."""
    if model not in _MODEL_DIRS:
        raise ValueError(f"Unknown model '{model}'. Expected one of {sorted(_MODEL_DIRS)}.")
    return _MODEL_DIRS[model]


def split_csv_paths(model: str) -> dict:
    """{'train','val','test'} -> absolute CSV path (str) for the given model ('pretrain' | 'downstream')."""
    d = splits_dir(model)
    return {"train": str(d / "train.csv"), "val": str(d / "val.csv"), "test": str(d / "test.csv")}


_DOWNSTREAM_SPLITS_DIR = splits_dir("downstream")
_PRETRAIN_SPLITS_DIR = splits_dir("pretrain")


def _load_splits(splits_dir: Path) -> dict:
    result = {}
    for split_name, fname in _SPLIT_FILES:
        df = pd.read_csv(splits_dir / fname)
        result[split_name] = {
            row["Pseudonym"]: row.drop("Pseudonym").to_dict()
            for _, row in df.iterrows()
        }
    return result


def load_downstream_splits(splits_dir: Path = None) -> dict:
    """Load downstream (mci/converter) splits. Returns {split_name: {patient_id: {metadata...}}}."""
    return _load_splits(splits_dir if splits_dir is not None else _DOWNSTREAM_SPLITS_DIR)


def load_pretrain_splits(splits_dir: Path = None) -> dict:
    """Load pretrain (all-cohort) splits. Returns {split_name: {patient_id: {metadata...}}}."""
    return _load_splits(splits_dir if splits_dir is not None else _PRETRAIN_SPLITS_DIR)


def get_split_indices_for_dataset(dataset, split_data: dict, split_name: str = "train") -> list:
    """
    Get indices into a dataset for patients in the given split.

    Args:
        dataset: dataset whose items expose a `patient_id` or `id` attribute
        split_data: loaded splits dict (from load_downstream_splits / load_pretrain_splits)
        split_name: "train", "validation", or "test"

    Returns:
        List of indices into the dataset that belong to the specified split.
    """
    if split_name not in split_data:
        available = [k for k in split_data if k != "metadata"]
        raise ValueError(f"Split '{split_name}' not found. Available: {available}")

    split_ids = set(split_data[split_name].keys())
    indices = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        patient_id = None
        if hasattr(data, "patient_id"):
            patient_id = str(data.patient_id).removeprefix("sub-")
        elif hasattr(data, "id"):
            patient_id = str(data.id).removeprefix("sub-")

        if patient_id and patient_id in split_ids:
            indices.append(idx)

    return indices


def get_all_split_indices(dataset, split_data: dict) -> dict:
    """Get indices for train/validation/test at once."""
    return {
        split_name: get_split_indices_for_dataset(dataset, split_data, split_name)
        for split_name in ["train", "validation", "test"]
    }


def get_split_patient_ids(split_data: dict, split_name: str) -> set:
    """Get all patient IDs for a specific split."""
    if split_name not in split_data:
        raise ValueError(f"Split '{split_name}' not found.")
    return set(split_data[split_name].keys())


if __name__ == "__main__":
    downstream = load_downstream_splits()
    print("Downstream splits:")
    for name in ["train", "validation", "test"]:
        print(f"  {name}: {len(downstream[name])} patients")

    pretrain = load_pretrain_splits()
    print("Pretrain splits:")
    for name in ["train", "validation", "test"]:
        n_scans = sum(v.get("n_scans", 1) for v in pretrain[name].values())
        print(f"  {name}: {len(pretrain[name])} patients, {n_scans} scans")
