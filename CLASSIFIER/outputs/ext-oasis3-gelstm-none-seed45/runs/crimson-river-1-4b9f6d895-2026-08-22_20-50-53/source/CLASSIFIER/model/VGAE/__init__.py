"""VGAE — Variational Graph Autoencoder feature extractor.

A reconstruction-only encoder trained like the GAAE (``model/GAAE``) but with the
GAT *feature* decoder replaced by a plain ``InnerProductDecoder`` over the latent
node embeddings (adjacency reconstruction) and a variational bottleneck (KL term).

Two encoder variants share one class via ``conv_type``:
    * ``"gcn"`` — canonical VGAE (``GCNConv`` encoder).
    * ``"gat"`` — attention encoder (``GATv2Conv``), closer to the GAAE.

Like the GAAE, the trained encoder is consumed downstream as a frozen feature
extractor: ``encode(x, edge_index, edge_attr).mean(0)`` yields the 64-d pooled
graph embedding the GEP / GEC / GELSTM classifiers sit on.
"""
