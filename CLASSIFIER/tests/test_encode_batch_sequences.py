"""Regression guard: encode_batch_sequences must NOT leave the model in
``.eval()`` mode after it returns. This caught the bug where the old code
silently mutated the caller's model state and required a defensive
``model.train()`` in the training loop.
"""
import torch
from torch_geometric.data import Data

from CLASSIFIER.model.GELSTM.utils import encode_batch_sequences, eval_mode


class _StubEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)

    def encode_visit(self, x, edge_index, edge_attr=None, pool="mean"):
        return torch.zeros(1)


def _make_batch():
    g = Data(x=torch.zeros(1, 1), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=None)
    return [{
        "subject_id": "s0",
        "graphs":     [g],
        "delta_t":    [0.0],
        "visit_months": [0.0],
        "label":      0.0,
    }]


def test_training_mode_preserved_when_caller_in_train():
    m = _StubEncoder()
    m.train()
    assert m.training is True
    _ = encode_batch_sequences(_make_batch(), m, device=torch.device("cpu"), use_time_delta=False)
    assert m.training is True, "encode_batch_sequences must not flip the caller's mode"


def test_eval_mode_context_manager_restores_state():
    m = _StubEncoder()
    m.train()
    with eval_mode(m):
        assert m.training is False
    assert m.training is True
