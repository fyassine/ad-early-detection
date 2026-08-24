"""
model/transformer.py — Brain-TokenGT: GIVE (INE + VEE) + BIGTR readout.

PORT of ``Brain-TokenGT/model_transformer.py::EvolveGCNH_Transformer`` (Dong et
al., MICCAI 2023). The upstream file is kept pristine at the repo root and is the
reference for ``tests/test_upstream_equivalence.py``, which asserts this class
reproduces it bit-for-bit at the authors' published configuration
(M=90 ROIs, T=3 visits, binary edge weights).

What changed, and why each change is a *port* and not a *modification*
======================================================================
An edit is a port when it provably yields the identical computation at the
authors' original settings; a modification changes the computation there.

PORTS (unconditional, always active)
  1. (M, T) generalisation. Upstream hardcodes ``max_num_nodes=270``,
     ``time_steps=3``, ``in_channels=90`` and — inside ``forward`` —
     ``np.eye(270,270,±90)``. Here M and T are constructor / call-time values and
     the temporal-edge pattern is derived from them. At (M=90, T=3) the generated
     tensors are elementwise equal to upstream's literals.
     T is read from ``len(A_list)`` per call, so subjects may have different
     visit counts without padding.
  2. Device hygiene. Upstream sprinkles ``.cuda()`` through ``__init__`` and
     ``forward`` and overrides ``parameters()`` to return a hand-rolled
     ``nn.ParameterList`` assigned over ``nn.Module``'s own ``_parameters``
     registry. Here submodules are registered normally and the module follows
     ``.to(device)``. Arithmetic is untouched.
  3. Edge-feature ordering. Upstream reads ``edge_attr`` row-major off
     ``adjs_all`` but builds ``hyperedge_index`` as ``[static block | temporal
     block]``, so feature i does not belong to edge i. Here features are gathered
     at the concatenated edge index. This is a NO-OP at the authors' settings
     (every upstream edge weight is exactly 1.0, so any permutation of an
     all-ones vector is the same vector) and only becomes observable under
     ``edge_weight_mode="weighted"``.

FLAGGED BEHAVIOURAL DIFFERENCES (default = reproduce upstream exactly)
  These upstream behaviours contradict the paper. Each is exposed as a flag whose
  DEFAULT reproduces the released code, so the faithful run is the default run and
  any deviation is an explicit, reportable choice.

  * ``force_single_head=True``  — upstream sets ``self.nhead = nhead`` then
    immediately ``self.nhead = 1``, discarding the configured head count.
  * ``readout="mean"``          — upstream averages ALL tokens; paper Fig. 1 reads
    out the ``[graph]`` token (``readout="graph_token"``).
  * ``edge_weight_mode="binary"`` — upstream's training script passes a binarised
    adjacency and never forwards ``edge_attr``, so VEE's hypergraph convolution is
    fed an all-ones vector and never sees FC weights at all. Paper §2.2 says edge
    features carry the connection weights (``"weighted"``).
  * ``train_give=False``        — upstream's ``Parameter(...).to(device)`` returns
    a plain tensor, so ``GRCU.parameters()`` is empty and the INE module is frozen
    at random init. ``False`` reproduces that optimiser parameter set; ``True``
    trains GIVE as the paper describes.

PRESERVED UPSTREAM QUIRKS (not flagged — kept unconditionally for fidelity)
  * Node identifiers are appended as M extra *tokens* rather than concatenated to
    each token's features (paper Eq. 4), and there are M of them for M*T node
    tokens.
  * The static/temporal split index is ``num_edge - M*(T-1)`` while the temporal
    block actually spans ``2*M*(T-1)`` columns, so half the temporal edges are
    pooled as "static". Reproduced exactly; see ``_split_static_temporal``.
  * ``self.projection`` is constructed but never used in ``forward``. Kept so
    parameter counts match upstream.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.pool import TopKPooling

from .grcu import GRCU, gaussian_orthogonal_random_matrix

_EDGE_WEIGHT_MODES = ("binary", "weighted")
_READOUTS = ("mean", "graph_token")


def time_alignment(
    num_nodes: int,
    time_steps: int,
    *,
    edge_weight: float = 1.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Upper-triangular temporal-edge pattern over the (M*T, M*T) giant graph.

    Connects node ``i`` at visit ``t`` to node ``i`` at visit ``t+1`` with weight
    ``edge_weight``, giving ``M*(T-1)`` directed entries. Generalisation of
    upstream ``time_alignment(edge_weight, max_num_nodes=270, time_steps=3)``:
    upstream derives ``M`` as ``max_num_nodes // time_steps``, here it is passed
    directly. Elementwise equal to upstream at (M=90, T=3).
    """
    total = num_nodes * time_steps
    adj = torch.zeros((total, total), device=device, dtype=dtype)
    if time_steps < 2:
        return adj
    idx = torch.arange(num_nodes, device=device)
    for t in range(time_steps - 1):
        adj[t * num_nodes + idx, (t + 1) * num_nodes + idx] = edge_weight
    return adj


def DHT(
    adjacency: torch.Tensor,
    temporal_edge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Dual Hypergraph Transformation: edges of the giant graph become dual nodes.

    Port of upstream ``DHT``. The ``batch`` / ``add_loops`` arguments and the
    ``edge_batch`` return value are dropped — upstream always calls this with a
    single graph (``batch = zeros``), ``add_loops=False``, and never consumes
    ``edge_batch``. Returns ``(hyperedge_index, edge_index, temporal_edge_num)``;
    ``edge_index`` is new (needed to gather edge features in the matching order).
    """
    temporal_edge = temporal_edge + temporal_edge.transpose(0, 1)

    static_edge_index = torch.vstack(torch.where(adjacency != 0)).contiguous()
    temporal_edge_index = torch.vstack(torch.where(temporal_edge != 0)).contiguous()
    temporal_edge_num = temporal_edge_index.shape[1] // 2

    edge_index = torch.hstack([static_edge_index, temporal_edge_index])
    num_edge = edge_index.size(1)

    edge_to_node_index = (
        torch.arange(0, num_edge, 1, device=adjacency.device).repeat_interleave(2).view(1, -1)
    )
    hyperedge_index = edge_index.T.reshape(1, -1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()

    return hyperedge_index, edge_index, temporal_edge_num


class BrainTokenGT(nn.Module):
    """Brain Tokenized Graph Transformer (Dong et al., MICCAI 2023).

    Parameters
    ----------
    in_channels : int
        Node-feature width. Equals the ROI count M for FC-row features
        (upstream: 90 for AAL-90; DELCODE: 200 for Schaefer-200).
    output_sizes : Sequence[int]
        Output width of each GRCU (EvolveGCN) layer. The last entry is the
        transformer ``d_model``. Each value must be <= ``num_nodes`` (the TopK
        inside ``mat_GRU_cell`` selects ``out_feats`` of the M nodes).
    num_nodes : int
        Number of ROIs M.
    nhead, num_layers
        Transformer encoder heads / depth. ``nhead`` is overridden to 1 unless
        ``force_single_head=False`` (see module docstring).
    static_edge_topk : int
        Number of spatial-edge tokens retained by ``TopKPooling``.
    edge_input_channels : int
        Input width of the VEE hypergraph convolution (upstream: 1).
    total_graph_size : int
        Number of distinct ``[graph]`` tokens (upstream: 1).

    Forward
    -------
    ``forward(A_list, Nodes_list)`` where both lists have length T:
      * ``A_list[t]``     : (M, M) dense adjacency for visit t
      * ``Nodes_list[t]`` : (M, in_channels) node features for visit t
    Returns a ``(1,)`` logit for the subject.
    """

    def __init__(
        self,
        in_channels: int,
        output_sizes: Sequence[int],
        *,
        num_nodes: int,
        activation=F.relu,
        nhead: int = 4,
        num_layers: int = 2,
        edge_input_channels: int = 1,
        total_graph_size: int = 1,
        static_edge_topk: int = 180,
        temporal_edge_weight: float = 1.0,
        edge_weight_mode: str = "binary",
        readout: str = "mean",
        force_single_head: bool = True,
        train_give: bool = False,
    ) -> None:
        super().__init__()

        if edge_weight_mode not in _EDGE_WEIGHT_MODES:
            raise ValueError(
                f"edge_weight_mode must be one of {_EDGE_WEIGHT_MODES}, got {edge_weight_mode!r}"
            )
        if readout not in _READOUTS:
            raise ValueError(f"readout must be one of {_READOUTS}, got {readout!r}")
        output_sizes = list(output_sizes)
        if not output_sizes:
            raise ValueError("output_sizes must contain at least one GRCU layer width")
        bad = [s for s in output_sizes if s > num_nodes]
        if bad:
            raise ValueError(
                f"output_sizes entries must be <= num_nodes ({num_nodes}); got {bad}. "
                "The TopK inside mat_GRU_cell selects out_feats of the M nodes, so a "
                "wider layer than the ROI count cannot be selected."
            )

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.output_sizes = output_sizes
        self.edge_weight_mode = edge_weight_mode
        self.readout = readout
        self.temporal_edge_weight = temporal_edge_weight
        self.train_give = train_give

        self.nhead = nhead
        if force_single_head:
            # Upstream: `self.nhead = nhead` immediately followed by `self.nhead = 1`.
            self.nhead = 1

        feats = [in_channels] + output_sizes
        # PORT: ModuleList (upstream used a plain Python list + a parameters() override).
        self.GRCU_layers = nn.ModuleList(
            GRCU(feats[i - 1], feats[i], activation) for i in range(1, len(feats))
        )
        if not train_give:
            # Reproduce upstream's effective optimiser parameter set: upstream's
            # Parameter(...).to(device) yields plain tensors, so GRCU contributes
            # nothing to parameters() and INE is never updated.
            for p in self.GRCU_layers.parameters():
                p.requires_grad_(False)

        last_size = output_sizes[-1]
        self.d_model = last_size

        self.linear = nn.Linear(last_size, last_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=last_size, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(last_size, 1)

        self.PoolingConvs = pyg_nn.HypergraphConv(edge_input_channels, last_size)
        self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=last_size)
        self.static_edge_topk = TopKPooling(in_channels=last_size, ratio=static_edge_topk)

        # Constructed by upstream but never used in forward. Kept for parity.
        self.projection = nn.Linear(in_channels, 256)

        # Non-trainable node identifiers Q. Upstream keeps this as a plain tensor
        # attribute; registered as a buffer here so it follows .to(device) and
        # round-trips through the checkpoint (values are identical).
        self.register_buffer(
            "orthogonal_matrix", gaussian_orthogonal_random_matrix(num_nodes, last_size)
        )
        self.graph_token = nn.Embedding(num_embeddings=total_graph_size, embedding_dim=last_size)

        # (T, device, dtype) -> temporal pattern; rebuilt only when T changes.
        self._temporal_cache: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}

    # ── helpers ──────────────────────────────────────────────────────────────
    def _temporal_pattern(self, time_steps: int, device, dtype) -> torch.Tensor:
        key = (time_steps, str(device), dtype)
        cached = self._temporal_cache.get(key)
        if cached is None:
            cached = time_alignment(
                self.num_nodes,
                time_steps,
                edge_weight=self.temporal_edge_weight,
                device=device,
                dtype=dtype,
            )
            self._temporal_cache[key] = cached
        return cached

    @staticmethod
    def _split_static_temporal(num_edge: int, temporal_edge_num: int) -> int:
        """Index at which upstream splits static from temporal edge embeddings.

        Upstream uses ``num_edge - temporal_edge_num`` where ``temporal_edge_num``
        is HALF the number of temporal columns (it is computed as
        ``temporal_edge_index.shape[1] // 2`` on a symmetrised pattern). The
        "static" block therefore also contains ``M*(T-1)`` temporal edges. This is
        an upstream off-by-half; reproduced verbatim so the port stays faithful.
        """
        return num_edge - temporal_edge_num

    def _edge_weights(self, A_list: List[torch.Tensor], temporal_sym: torch.Tensor):
        """Dense (M*T, M*T) weight matrix whose values become the VEE input.

        ``binary``  : every present edge weighs 1.0 — upstream's effective input,
                      since its training script passes ``to_dense_adj(edge_index)``
                      and never forwards ``edge_attr``.
        ``weighted``: spatial edges carry their FC weight (paper §2.2); temporal
                      edges carry ``temporal_edge_weight``.
        """
        adjs = torch.block_diag(*A_list)
        topology = (adjs != 0).to(adjs.dtype)
        if self.edge_weight_mode == "binary":
            return topology, topology + temporal_sym
        return topology, adjs + temporal_sym

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        A_list: Sequence[torch.Tensor],
        Nodes_list: Sequence[torch.Tensor],
        nodes_mask_list: Optional[Sequence] = None,
        graph_id: int = 0,
        use_node_identifier: bool = True,
        use_type_identifier: bool = True,
    ) -> torch.Tensor:
        A_list = list(A_list)
        Nodes_list = list(Nodes_list)
        T = len(A_list)
        if T != len(Nodes_list):
            raise ValueError(
                f"A_list and Nodes_list must have the same length; got {T} and {len(Nodes_list)}"
            )
        if T == 0:
            raise ValueError("A_list is empty: a subject must have at least one visit")

        device = A_list[0].device
        dtype = A_list[0].dtype

        # ── GIVE / INE: evolving graph convolution over the visit sequence ────
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list)
        node_embedding = torch.vstack(Nodes_list)  # (M*T, d)

        # ── GIVE / VEE: giant graph -> dual hypergraph -> edge embeddings ─────
        temporal_pattern = self._temporal_pattern(T, device, dtype)
        temporal_sym = temporal_pattern + temporal_pattern.transpose(0, 1)
        topology, weights = self._edge_weights(A_list, temporal_sym)

        hyperedge_index, edge_index, temporal_edge_num = DHT(topology, temporal_pattern)
        # PORT: gather features at the concatenated edge index so feature i belongs
        # to edge i. Upstream re-scans the dense matrix row-major, which permutes
        # them; a no-op while every weight is 1.0 (see module docstring).
        edge_attr = weights[edge_index[0], edge_index[1]]

        edge_embedding = F.mish(self.PoolingConvs(edge_attr.view(-1, 1), hyperedge_index))

        split = self._split_static_temporal(edge_embedding.shape[0], temporal_edge_num)
        static_edge_embedding = edge_embedding[0:split]
        static_edge_index = hyperedge_index[:, 0:split]
        static_edge_embedding = self.static_edge_topk(static_edge_embedding, static_edge_index)[0]
        temporal_edge_embedding = edge_embedding[split:]

        # ── BIGTR: type identifiers ──────────────────────────────────────────
        if use_type_identifier:
            node_embedding = node_embedding + self.type_embedding(
                torch.zeros(node_embedding.shape[0], dtype=torch.long, device=device)
            )
            static_edge_embedding = static_edge_embedding + self.type_embedding(
                torch.ones(static_edge_embedding.shape[0], dtype=torch.long, device=device)
            )
            temporal_edge_embedding = temporal_edge_embedding + self.type_embedding(
                2 * torch.ones(temporal_edge_embedding.shape[0], dtype=torch.long, device=device)
            )

        # ── BIGTR: assemble the token sequence ───────────────────────────────
        blocks = [node_embedding, static_edge_embedding, temporal_edge_embedding]
        if use_node_identifier:
            blocks.append(self.orthogonal_matrix)
        all_embeddings = torch.vstack(blocks)

        graph_embedding = self.graph_token(torch.tensor(graph_id, device=device)).squeeze(0)
        all_embeddings = torch.vstack([graph_embedding, all_embeddings])

        all_embeddings = self.linear(all_embeddings)
        encoded = self.transformer_encoder(all_embeddings)

        out = encoded[0] if self.readout == "graph_token" else encoded.mean(0)
        return self.classifier(out)

    # ── introspection ────────────────────────────────────────────────────────
    def token_count(self, time_steps: int, num_static_edges: int) -> int:
        """Token-sequence length for a T-visit subject (diagnostics / capacity)."""
        temporal = self.num_nodes * (time_steps - 1)
        return 1 + self.num_nodes * time_steps + num_static_edges + temporal + self.num_nodes

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
