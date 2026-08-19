"""
model/grcu.py — GIVE / INE: EvolveGCN-H recurrent graph-convolution unit.

PORT of ``Brain-TokenGT/model_grcu.py`` (Dong et al., MICCAI 2023). The upstream
file is kept pristine at the repo root and is the reference for
``tests/test_upstream_equivalence.py``.

Port rules (see BRAINTOKENGT/README.md §"What was changed"):

  * The arithmetic of every ``forward`` is byte-for-byte the upstream arithmetic.
  * Only device handling and parameter registration changed:

      upstream                                    here
      --------------------------------------------------------------------
      ``Parameter(torch.Tensor(r, c)).to(dev)``   ``nn.Parameter(torch.empty(r, c))``
      ``node_embs.cuda()`` inside forward         caller places tensors

    Upstream's ``.to(device)`` on a freshly-constructed ``Parameter`` returns a
    plain ``Tensor`` whenever the device actually changes, so the parameter is
    never registered on the module. The measurable consequence upstream is that
    ``GRCU.parameters()`` is EMPTY and the whole INE module stays frozen at its
    random initialisation — the optimiser never receives it. Registering the
    parameters properly (as here) makes them trainable, which is a behavioural
    change, so it is gated behind ``BrainTokenGT(train_give=...)`` rather than
    applied silently. ``train_give=False`` reproduces the upstream optimiser's
    parameter set exactly.

Nothing else about the module changed: the same gates, the same TopK selection,
the same initialisation distributions in the same order.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


@torch.no_grad()
def orthogonal_matrix_chunk(cols: int, device=None) -> torch.Tensor:
    """One (cols, cols) orthogonal block via QR. Verbatim from upstream."""
    unstructured_block = torch.randn((cols, cols), device=device)
    q, _r = torch.linalg.qr(unstructured_block, mode="reduced")
    return q.t()


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_rows: int, nb_columns: int, device=None) -> torch.Tensor:
    """(nb_rows, nb_columns) row-normalised Gaussian orthogonal matrix.

    Verbatim from upstream. Used for the non-trainable node identifiers ``Q``.
    """
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    normalizer = final_matrix.norm(p=2, dim=1, keepdim=True)
    normalizer[normalizer == 0] = 1e-5
    return final_matrix / normalizer


def pad_with_last_val(vect: torch.Tensor, k: int) -> torch.Tensor:
    """Right-pad ``vect`` to length ``k`` by repeating its last entry.

    Inlined from upstream ``utils.pad_with_last_val`` so this package does not
    depend on upstream's ``utils.py`` (which also carries a time-seeded
    ``set_seeds`` we must not import — see .claude/rules/seeding.md).
    """
    pad = torch.ones(k - vect.size(0), dtype=torch.long, device=vect.device) * vect[-1]
    return torch.cat([vect, pad])


class mat_GRU_gate(nn.Module):
    """One GRU gate over a weight *matrix* (EvolveGCN-H). Upstream arithmetic."""

    def __init__(self, rows: int, cols: int, activation: nn.Module):
        super().__init__()
        self.activation = activation
        # PORT: nn.Parameter, registered. Upstream: Parameter(...).to(device) -> Tensor.
        self.W = Parameter(torch.empty(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.empty(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t: torch.Tensor) -> None:
        # Initialise based on the number of columns (upstream).
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)


class TopK(nn.Module):
    """Score-and-select the top-k nodes, scaled by tanh(score). Upstream arithmetic."""

    def __init__(self, feats: int, k: int):
        super().__init__()
        self.scorer = Parameter(torch.empty(feats, 1))
        self.reset_param(self.scorer)
        self.k = k

    def reset_param(self, t: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is None:
            mask = 0
        # PORT: no .cuda() — node_embs already carries the caller's device.
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.numel() == 0:
            # Diagnostic guard (see BRAINTOKENGT/README.md "fix-failloud" experiment):
            # every candidate score came back non-finite. With train_give=False this
            # is a no-op (the scorer is frozen at a bounded random init and never
            # produces NaN); with train_give=True it means the GIVE/GRCU parameters
            # diverged during training. Fail with the diagnosis instead of the
            # cryptic IndexError from indexing an empty tensor below.
            raise ValueError(
                "TopK.forward: every candidate node score is non-finite (NaN/-inf). "
                f"scores: min={scores.min().item()!r} max={scores.max().item()!r} "
                f"any_nan={bool(torch.isnan(scores).any())} any_inf={bool(torch.isinf(scores).any())}. "
                "This means the GIVE/GRCU parameters (scorer or upstream node "
                "embeddings) have diverged during training, not a data or masking "
                "issue — see the 'What was changed' / stability note in "
                "BRAINTOKENGT/README.md."
            )

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = nn.Tanh()
        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))
        return out.t()


class mat_GRU_cell(nn.Module):
    """GRU cell whose hidden state is the GCN weight matrix. Upstream arithmetic."""

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.update = mat_GRU_gate(rows, cols, nn.Sigmoid())
        self.reset = mat_GRU_gate(rows, cols, nn.Sigmoid())
        self.htilda = mat_GRU_gate(rows, cols, nn.Tanh())
        self.choose_topk = TopK(feats=rows, k=cols)

    def forward(self, prev_Q: torch.Tensor, prev_Z: torch.Tensor, mask) -> torch.Tensor:
        z_topk = self.choose_topk(prev_Z, mask)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        return (1 - update) * prev_Q + update * h_cap


class GRCU(nn.Module):
    """One evolving graph-convolution layer over the visit sequence.

    ``forward(A_list, node_embs_list)`` returns the per-visit node embeddings after
    a GCN whose weight matrix is carried across visits by ``mat_GRU_cell``.
    """

    def __init__(self, in_feats: int, out_feats: int, activation):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.evolve_weights = mat_GRU_cell(in_feats, out_feats)
        self.activation = activation
        self.GCN_init_weights = Parameter(torch.empty(in_feats, out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list, mask_list=None):
        if mask_list is None:
            mask_list = [None] * len(node_embs_list)

        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            # PORT: no .cuda() — tensors arrive on the caller's device.
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            out_seq.append(node_embs)

        return out_seq
