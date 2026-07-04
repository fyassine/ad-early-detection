"""
Reproducibility helpers: process-wide seeding and RNG factories.

Notebooks and training scripts should call ``set_seed(SEED)`` at the very top
and then thread ``make_rng(SEED)`` / ``make_torch_generator(SEED)`` /
``seed_worker`` into any function that needs randomness (batch shuffling,
DataLoaders, etc.) rather than relying on global state.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
