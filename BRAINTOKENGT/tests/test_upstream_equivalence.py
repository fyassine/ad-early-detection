"""
Regression test: the port reproduces the authors' released implementation.

This is the evidence behind the thesis claim "we ran the authors' implementation".
It instantiates upstream ``Brain-TokenGT/model_transformer.py::EvolveGCNH_Transformer``
and ``BRAINTOKENGT.model.BrainTokenGT`` at the authors' published configuration
(M=90 AAL ROIs, T=3 visits, binary edge weights), copies the upstream weights into
the port parameter-for-parameter, and asserts the forward passes agree to
floating-point tolerance.

If this test passes, every difference between the two files is confined to
(M, T) generalisation, device handling, and edge-feature ordering — none of which
changes the computation at the authors' settings.

Upstream requires CUDA (it hardcodes ``.cuda()`` throughout), so the equivalence
tests skip on CPU-only machines. The (M, T) generalisation tests do not, and run
everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_DIR = _REPO_ROOT / "Brain-TokenGT"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from BRAINTOKENGT.model import BrainTokenGT, time_alignment  # noqa: E402

# Authors' published configuration.
UPSTREAM_M = 90
UPSTREAM_T = 3
UPSTREAM_TOPK = 180

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="upstream Brain-TokenGT hardcodes .cuda(); equivalence can only be checked on GPU",
)
requires_upstream = pytest.mark.skipif(
    not (_UPSTREAM_DIR / "model_transformer.py").is_file(),
    reason=f"pristine upstream checkout not found at {_UPSTREAM_DIR}",
)


def _load_upstream():
    """Import the pristine upstream module.

    Upstream uses flat imports (``from model_grcu import ...``), so its directory
    must be on ``sys.path``. It is inserted at position 0 and left there for the
    session; the modules it shadows (``utils``) are not imported by this package.
    """
    if str(_UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM_DIR))
    import model_transformer  # noqa: PLC0415

    return model_transformer


def _copy_weights(upstream, port) -> None:
    """Copy every upstream weight into the port.

    Done explicitly rather than via ``state_dict()`` because upstream assigns an
    ``nn.ParameterList`` over ``nn.Module``'s internal ``_parameters`` registry and
    overrides ``parameters()``, so its ``state_dict()`` is not trustworthy. The
    explicit walk below doubles as documentation of the module correspondence.
    """
    with torch.no_grad():
        # Standard submodules — same classes, same shapes on both sides.
        for name in (
            "linear",
            "transformer_encoder",
            "classifier",
            "PoolingConvs",
            "type_embedding",
            "static_edge_topk",
            "projection",
            "graph_token",
        ):
            getattr(port, name).load_state_dict(getattr(upstream, name).state_dict())

        # Non-trainable node identifiers Q (a plain tensor upstream, a buffer here).
        port.orthogonal_matrix.copy_(upstream.orthogonal_matrix)

        # GIVE / INE: upstream holds these as plain tensors on a Python list.
        assert len(port.GRCU_layers) == len(upstream.GRCU_layers)
        for p_layer, u_layer in zip(port.GRCU_layers, upstream.GRCU_layers, strict=True):
            p_layer.GCN_init_weights.copy_(u_layer.GCN_init_weights)
            for gate in ("update", "reset", "htilda"):
                p_gate = getattr(p_layer.evolve_weights, gate)
                u_gate = getattr(u_layer.evolve_weights, gate)
                p_gate.W.copy_(u_gate.W)
                p_gate.U.copy_(u_gate.U)
                p_gate.bias.copy_(u_gate.bias)
            p_layer.evolve_weights.choose_topk.scorer.copy_(
                u_layer.evolve_weights.choose_topk.scorer
            )


def _make_inputs(M: int, T: int, device, seed: int = 7):
    """Binary adjacencies + FC-row node features, as upstream's training script builds them.

    ``main_optuna.py:84`` passes ``to_dense_adj(edge_index)``, i.e. a BINARY
    adjacency, and never forwards ``edge_attr`` — so binary inputs are the
    upstream-faithful case.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A_list, Nodes_list = [], []
    for _ in range(T):
        adj = (torch.rand(M, M, generator=gen) > 0.85).to(torch.float32)
        A_list.append(adj.to(device))
        Nodes_list.append(torch.rand(M, M, generator=gen).to(device))
    return A_list, Nodes_list


def _build_pair(device, *, output_sizes, num_layers, nhead, seed=0):
    upstream_mod = _load_upstream()
    torch.manual_seed(seed)
    upstream = upstream_mod.EvolveGCNH_Transformer(
        in_channels=UPSTREAM_M,
        output_sizes=list(output_sizes),
        nhead=nhead,
        num_layers=num_layers,
        num_nodes=UPSTREAM_M,
        static_edge_topk=UPSTREAM_TOPK,
    )
    port = BrainTokenGT(
        in_channels=UPSTREAM_M,
        output_sizes=list(output_sizes),
        num_nodes=UPSTREAM_M,
        nhead=nhead,
        num_layers=num_layers,
        static_edge_topk=UPSTREAM_TOPK,
        edge_weight_mode="binary",
        readout="mean",
        force_single_head=True,
        train_give=False,
    ).to(device)

    _copy_weights(upstream, port)

    # Dropout lives in the transformer encoder. Disable it explicitly on both
    # sides: upstream's own .eval() cannot be relied on, because it overrides
    # nn.Module's _parameters registry.
    upstream.transformer_encoder.eval()
    port.eval()
    port.transformer_encoder.eval()
    return upstream, port


# --------------------------------------------------------------------------- #
# Equivalence at the authors' published configuration
# --------------------------------------------------------------------------- #
@requires_upstream
@requires_cuda
@pytest.mark.parametrize(
    "output_sizes,num_layers,nhead",
    [
        ([32, 32], 2, 2),
        ([64], 1, 4),
        ([48, 24, 24], 3, 1),
    ],
)
def test_forward_matches_upstream(output_sizes, num_layers, nhead):
    """Port and upstream produce the same logit at (M=90, T=3), binary edges."""
    device = torch.device("cuda")
    upstream, port = _build_pair(
        device, output_sizes=output_sizes, num_layers=num_layers, nhead=nhead
    )
    A_list, Nodes_list = _make_inputs(UPSTREAM_M, UPSTREAM_T, device)

    with torch.no_grad():
        out_upstream = upstream(A_list, Nodes_list, None)
        out_port = port(A_list, Nodes_list, None)

    assert out_port.shape == out_upstream.shape
    torch.testing.assert_close(out_port, out_upstream, rtol=1e-5, atol=1e-6)


@requires_upstream
@requires_cuda
def test_forward_matches_upstream_across_inputs():
    """Equivalence holds across independent input draws, not just one lucky seed."""
    device = torch.device("cuda")
    upstream, port = _build_pair(device, output_sizes=[32, 32], num_layers=2, nhead=2)

    for seed in (1, 2, 3, 4, 5):
        A_list, Nodes_list = _make_inputs(UPSTREAM_M, UPSTREAM_T, device, seed=seed)
        with torch.no_grad():
            out_upstream = upstream(A_list, Nodes_list, None)
            out_port = port(A_list, Nodes_list, None)
        torch.testing.assert_close(
            out_port, out_upstream, rtol=1e-5, atol=1e-6, msg=f"diverged on input seed {seed}"
        )


@requires_upstream
@requires_cuda
def test_upstream_give_is_untrainable():
    """Document the upstream defect that ``train_give`` exists to expose.

    Upstream's ``Parameter(...).to(device)`` returns a plain tensor, so the INE /
    EvolveGCN module contributes NO parameters to the optimiser and stays frozen at
    random init. The port reproduces that parameter set with ``train_give=False``
    and repairs it with ``train_give=True``. If this test ever fails, upstream has
    been modified and the ``train_give`` flag needs revisiting.
    """
    upstream_mod = _load_upstream()
    torch.manual_seed(0)
    upstream = upstream_mod.EvolveGCNH_Transformer(
        in_channels=UPSTREAM_M,
        output_sizes=[32, 32],
        nhead=2,
        num_layers=2,
        num_nodes=UPSTREAM_M,
        static_edge_topk=UPSTREAM_TOPK,
    )
    assert len(list(upstream.GRCU_layers[0].parameters())) == 0
    assert not isinstance(upstream.GRCU_layers[0].GCN_init_weights, torch.nn.Parameter)

    frozen = BrainTokenGT(
        in_channels=UPSTREAM_M,
        output_sizes=[32, 32],
        num_nodes=UPSTREAM_M,
        static_edge_topk=UPSTREAM_TOPK,
        train_give=False,
    )
    trained = BrainTokenGT(
        in_channels=UPSTREAM_M,
        output_sizes=[32, 32],
        num_nodes=UPSTREAM_M,
        static_edge_topk=UPSTREAM_TOPK,
        train_give=True,
    )
    assert all(not p.requires_grad for p in frozen.GRCU_layers.parameters())
    assert all(p.requires_grad for p in trained.GRCU_layers.parameters())
    assert len(trained.get_trainable_params()) > len(frozen.get_trainable_params())


# --------------------------------------------------------------------------- #
# The (M, T) generalisation reduces to upstream's literals
# --------------------------------------------------------------------------- #
def test_time_alignment_matches_upstream_literal():
    """``time_alignment(90, 3)`` equals upstream's hardcoded 270x270 pattern."""
    ours = time_alignment(UPSTREAM_M, UPSTREAM_T)

    expected = torch.zeros(270, 270)
    for i in range(270 // 3):
        idx = list(range(i, 270, 270 // 3))
        for j in range(len(idx) - 1):
            expected[idx[j]][idx[j + 1]] = 1
    torch.testing.assert_close(ours, expected)


def test_temporal_pattern_matches_eye_literals():
    """The symmetrised pattern equals upstream's ``np.eye(270,270,±90)`` sum."""
    pattern = time_alignment(UPSTREAM_M, UPSTREAM_T)
    symmetrised = pattern + pattern.T

    expected = torch.tensor(np.eye(270, 270, 90) + np.eye(270, 270, -90), dtype=torch.float32)
    torch.testing.assert_close(symmetrised, expected)


@pytest.mark.parametrize("M,T", [(90, 3), (200, 2), (200, 3), (200, 6), (50, 4)])
def test_time_alignment_edge_count(M, T):
    """The pattern always carries exactly M*(T-1) directed temporal edges."""
    pattern = time_alignment(M, T)
    assert pattern.shape == (M * T, M * T)
    assert int((pattern != 0).sum()) == M * (T - 1)


def test_time_alignment_single_visit_is_empty():
    """T=1 yields no temporal edges (the path 1-scan DELCODE subjects would hit)."""
    pattern = time_alignment(200, 1)
    assert pattern.shape == (200, 200)
    assert int((pattern != 0).sum()) == 0


# --------------------------------------------------------------------------- #
# Generalisation actually runs at DELCODE's shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [2, 3, 4])
def test_forward_runs_at_schaefer200(T):
    """Forward works at M=200 with a variable visit count and returns one logit."""
    M = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTokenGT(
        in_channels=M,
        output_sizes=[32, 32],
        num_nodes=M,
        num_layers=1,
        static_edge_topk=180,
    ).to(device)
    model.eval()
    A_list, Nodes_list = _make_inputs(M, T, device)

    with torch.no_grad():
        out = model(A_list, Nodes_list, None)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_weighted_mode_changes_output():
    """``edge_weight_mode='weighted'`` is a real behavioural difference, not a no-op.

    Guards the two-row reporting story: if this ever became a no-op, the
    "edge weights restored" row would silently duplicate the faithful row.
    """
    M, T = 90, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(11)
    A_list = [
        (torch.rand(M, M, generator=gen) * (torch.rand(M, M, generator=gen) > 0.85)).to(device)
        for _ in range(T)
    ]
    Nodes_list = [torch.rand(M, M, generator=gen).to(device) for _ in range(T)]

    outs = {}
    for mode in ("binary", "weighted"):
        torch.manual_seed(3)
        model = BrainTokenGT(
            in_channels=M,
            output_sizes=[32],
            num_nodes=M,
            num_layers=1,
            static_edge_topk=180,
            edge_weight_mode=mode,
        ).to(device)
        model.eval()
        with torch.no_grad():
            outs[mode] = model(A_list, Nodes_list, None)

    assert not torch.allclose(outs["binary"], outs["weighted"])


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def test_rejects_layer_wider_than_roi_count():
    """out_feats > M cannot work (TopK selects out_feats of M nodes) — fail loudly."""
    with pytest.raises(ValueError, match="output_sizes entries must be <="):
        BrainTokenGT(in_channels=90, output_sizes=[128], num_nodes=90)


def test_rejects_unknown_modes():
    with pytest.raises(ValueError, match="edge_weight_mode"):
        BrainTokenGT(in_channels=90, output_sizes=[32], num_nodes=90, edge_weight_mode="nope")
    with pytest.raises(ValueError, match="readout"):
        BrainTokenGT(in_channels=90, output_sizes=[32], num_nodes=90, readout="nope")


def test_rejects_mismatched_sequence_lengths():
    model = BrainTokenGT(in_channels=90, output_sizes=[32], num_nodes=90)
    A_list, Nodes_list = _make_inputs(90, 3, torch.device("cpu"))
    with pytest.raises(ValueError, match="same length"):
        model(A_list, Nodes_list[:2], None)
