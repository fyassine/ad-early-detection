"""Tests for the Youden's-J path in GELSTM.evaluate using a stub model."""
import numpy as np
import torch

from CLASSIFIER.configs.gelstm import EvalConfig
from CLASSIFIER.model.GELSTM.train import evaluate


class _StubModel(torch.nn.Module):
    """Model that returns canned logits in the order it receives batches."""
    def __init__(self, logits_by_call):
        super().__init__()
        self._logits = list(logits_by_call)
        self._i = 0
        # Provide encode_visit so encode_batch_sequences can run.
        self.fc = torch.nn.Linear(1, 1)

    def encode_visit(self, x, edge_index, edge_attr=None, pool="mean"):
        return torch.zeros(1)  # 1-dim "embedding"

    def forward(self, packed):
        out = torch.tensor(self._logits[self._i], dtype=torch.float)
        self._i += 1
        return out


def _make_batch(labels):
    """Build a minimal batch of subject dicts compatible with encode_batch_sequences."""
    from torch_geometric.data import Data
    items = []
    for sid, y in enumerate(labels):
        g = Data(
            x=torch.zeros(1, 1),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=None,
        )
        items.append({
            "subject_id": str(sid),
            "graphs":      [g],
            "delta_t":     [0.0],
            "visit_months":[0.0],
            "label":       float(y),
        })
    return items


def test_youden_threshold_selected():
    # 4 well-separated probabilities; threshold should land between groups.
    # logits: sigmoid(-2)=.12, sigmoid(-1)=.27, sigmoid(1)=.73, sigmoid(2)=.88
    batch = _make_batch([0, 0, 1, 1])
    stub = _StubModel(logits_by_call=[[-2.0, -1.0, 1.0, 2.0]])
    eval_cfg = EvalConfig(use_time_delta=False, threshold_mode="youden")
    out = evaluate(stub, [batch], device=torch.device("cpu"), eval_cfg=eval_cfg)
    assert out["auc"] == 1.0
    # Youden threshold should separate the two groups; with these logits the
    # ROC curve places it at the upper-class boundary (~sigmoid(1)=0.7311).
    assert 0.27 < out["best_threshold"] <= 0.74
    assert out["threshold_used"] == out["best_threshold"]
    assert out["sensitivity"] == 1.0
    assert out["specificity"] == 1.0


def test_single_class_returns_safe_zeros():
    batch = _make_batch([1, 1, 1, 1])
    stub = _StubModel(logits_by_call=[[0.5, 0.6, 0.7, 0.8]])
    eval_cfg = EvalConfig(use_time_delta=False, threshold_mode="youden")
    out = evaluate(stub, [batch], device=torch.device("cpu"), eval_cfg=eval_cfg)
    assert out["auc"] == 0.0
    # No crash; sens/spec/f1 numerically defined.
    assert 0.0 <= out["sensitivity"] <= 1.0
    assert 0.0 <= out["specificity"] <= 1.0
