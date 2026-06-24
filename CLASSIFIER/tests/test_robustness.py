import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from CLASSIFIER.common.robustness import perturb_graph


def _small_graph():
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    x = torch.randn(6, 4)
    return Data(x=x, edge_index=edge_index)


def test_edge_perturbation_dense_adjacency_has_no_values_above_one():
    sample = _small_graph()
    rng = np.random.default_rng(0)
    for noise_level in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        perturbed = perturb_graph(sample, "edge_perturbation", noise_level, rng=rng)
        dense = to_dense_adj(perturbed.edge_index, max_num_nodes=sample.x.size(0))
        assert dense.max() <= 1.0, (
            f"noise_level={noise_level} produced a dense-adjacency value > 1, "
            "which violates BCE's target-in-[0,1] requirement"
        )


def test_edge_perturbation_edge_index_has_no_duplicate_columns():
    sample = _small_graph()
    rng = np.random.default_rng(1)
    perturbed = perturb_graph(sample, "edge_perturbation", 0.3, rng=rng)
    num_edges = perturbed.edge_index.size(1)
    num_unique = torch.unique(perturbed.edge_index, dim=1).size(1)
    assert num_edges == num_unique
