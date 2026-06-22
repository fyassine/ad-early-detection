"""
adapters/gep.py — GEPAdapter (Graph-Embedding Pooled classifier).

The simplest downstream model: each visit is encoded once through the frozen
feature extractor and mean-pooled to a single 64-d graph embedding; a subject's
visits are then averaged into ONE pooled vector that a small MLP classifies
(converter vs stable-MCI). No trajectory flattening, no Δt, no padding mask —
contrast with ``adapters/gec.py`` (which flattens the whole padded visit
sequence). It exists to test how much converter signal the pooled GAAE/VGAE
embedding carries on its own.

The encoder backbone is selected by ``encoder_arch``:
  * ``"gaae"`` (default) — the GAAE ``GraphAttentionAutoencoderConditioned``
    checkpoint at ``gaae_ckpt_path`` / ``gaae_hp`` (identical to GEC/GELSTM).
  * ``"vgae"``           — a ``model.VGAE`` checkpoint; its architecture
    (``conv_type``/``hidden_dim``/``latent_dim``/…) and the graph-construction
    knobs (``adjacency_k``/``file_variant``) are read from the training config so
    the shared LONGITUDINAL_COMMON notebook needs no per-encoder edits.

Per-subject pooled vectors are standardised with a per-fold ``StandardScaler``
(re-emitted as ``scaler.pkl`` by ``extra_artifacts``), matching the GEC convention.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from common.crossval import Bundle
from model.GELSTM.dataset import LongitudinalSubjectDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import (
    LongitudinalAdapter,
    binary_metrics,
    load_run_checkpoint,
    model_state_from_checkpoint,
)
from .gec import LongitudinalMLP

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]


class GEPAdapter(LongitudinalAdapter):
    """Frozen encoder → mean-pooled graph embedding → small MLP classifier."""

    model_tag = "gep"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        c = self.cfg
        self.mlp_hidden_layers = c.get("mlp_hidden_layers", [32])
        self.mlp_dropout = c.get("mlp_dropout", 0.3)
        self.learning_rate = c.get("learning_rate", 1e-3)
        self.weight_decay = c.get("weight_decay", 1e-4)
        self.epochs = c.get("epochs", 80)
        self.early_stopping_patience = c.get("early_stopping_patience", 20)
        self.batch_size = c.get("batch_size", 32)
        self.grad_clip = c.get("grad_clip", 1.0)
        self.graph_pool = c.get("graph_pool", "mean")

        # Encoder backbone. For VGAE the arch + graph-construction knobs come from
        # the training config (gaae_hp describes the GAAE encoder, not the VGAE).
        self.encoder_arch = str(c.get("encoder_arch", "gaae")).lower()
        if self.encoder_arch == "vgae":
            self.enc_hidden = c.get("hidden_dim", self.gaae_hidden)
            self.enc_latent = c.get("latent_dim", self.gaae_latent)
            self.enc_conv_type = c.get("conv_type", "gcn")
            self.enc_heads = c.get("num_heads", self.gaae_heads)
            self.enc_dropout = c.get("dropout", self.gaae_dropout)
            self.enc_feature_decoder = bool(c.get("feature_decoder", False))
            self.adjacency_k = c.get("adjacency_k", self.adjacency_k)
            self.file_variant = c.get("file_variant", self.file_variant)
            self.latent = self.enc_latent
        else:
            self.latent = self.gaae_latent

        self.feat_dim: Optional[int] = self.latent
        self._encoder_model: Optional[nn.Module] = None
        self._cached_state_id: Optional[int] = None
        self._cached_model: Optional[nn.Module] = None

    # ── frozen encoder ───────────────────────────────────────────────────────
    def _encoder(self) -> nn.Module:
        if self._encoder_model is None:
            if self.encoder_arch == "vgae":
                from model.VGAE.models import VariationalGraphAutoencoder

                enc = VariationalGraphAutoencoder(
                    in_features=self.in_features, hidden_dim=self.enc_hidden,
                    latent_dim=self.enc_latent, conv_type=self.enc_conv_type,
                    num_heads=self.enc_heads, dropout=self.enc_dropout,
                    feature_decoder=self.enc_feature_decoder,
                ).to(self.device)
            else:
                from model.GAAE.models import GraphAttentionAutoencoderConditioned

                enc = GraphAttentionAutoencoderConditioned(
                    in_features=self.in_features, hidden_dim=self.gaae_hidden,
                    out_features=self.gaae_latent, cond_dim=self.gaae_cond_dim,
                    num_heads=self.gaae_heads, dropout=self.gaae_dropout,
                ).to(self.device)
            obj = torch.load(self.gaae_ckpt_path, map_location=self.device, weights_only=False)
            enc.load_state_dict(obj if isinstance(obj, dict) else obj.state_dict())
            enc.eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            self._encoder_model = enc
        return self._encoder_model

    def _pool_graph(self, enc, g) -> np.ndarray:
        ea = g.edge_attr.to(self.device) if g.edge_attr is not None else None
        z = enc.encode(g.x.to(self.device), g.edge_index.to(self.device), ea)
        if self.graph_pool == "max":
            return z.max(0).values.cpu().numpy()
        if self.graph_pool == "sum":
            return z.sum(0).cpu().numpy()
        return z.mean(0).cpu().numpy()

    # ── data ─────────────────────────────────────────────────────────────────
    def prepare_data(self, df) -> Bundle:
        ds = LongitudinalSubjectDataset(
            self.data_root, df, self.cohorts_csv,
            adjacency_k=self.adjacency_k, file_variant=self.file_variant,
        )
        enc = self._encoder()
        records: List[Dict[str, Any]] = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                records.append({
                    "subject_id": item["subject_id"],
                    "label": item["label"],
                    "n_scans": item["n_scans"],
                    "visit_months": list(item["visit_months"]),
                    "zs": [self._pool_graph(enc, g) for g in item["graphs"]],
                })
        return Bundle([r["label"] for r in records], [r["subject_id"] for r in records], records)

    def _records_to_X(self, items, n_visits=None):
        """Mean-pool each subject's visit embeddings into one (latent,) vector."""
        n = len(items)
        X = np.zeros((n, self.latent), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        for i, it in enumerate(items):
            cap = it["n_scans"] if n_visits is None else min(n_visits, it["n_scans"])
            zs = np.stack(it["zs"][:cap])
            X[i] = zs.mean(0)
            y[i] = float(it["label"])
        return X, y

    # ── arch ─────────────────────────────────────────────────────────────────
    def _build_mlp(self, input_dim: int) -> LongitudinalMLP:
        return LongitudinalMLP(input_dim, self.mlp_hidden_layers, self.mlp_dropout).to(self.device)

    def build_model(self) -> LongitudinalMLP:
        m = self._build_mlp(self.latent)
        print(f"GEP MLP: input={self.latent}  encoder={self.encoder_arch}  "
              f"params={sum(p.numel() for p in m.parameters()):,}")
        return m

    # ── training ─────────────────────────────────────────────────────────────
    def train_fold(self, bundle_tr, bundle_va, cfg, *, rng, device) -> Dict[str, Any]:
        items_tr, items_va = bundle_tr.items, bundle_va.items
        X_tr_raw, y_tr = self._records_to_X(items_tr)
        X_va_raw, y_va = self._records_to_X(items_va)
        scaler = StandardScaler().fit(X_tr_raw)

        X_tr = torch.tensor(scaler.transform(X_tr_raw), dtype=torch.float32)
        X_va = torch.tensor(scaler.transform(X_va_raw), dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_w = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        model = self._build_mlp(self.latent)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=7)

        tr_ds = torch.utils.data.TensorDataset(X_tr, y_tr_t)
        drop_last = len(tr_ds) % self.batch_size == 1  # BatchNorm1d rejects a 1-sample batch
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        best_auc, best_state, no_improve = 0.0, None, 0
        for _epoch in range(self.epochs):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
            model.eval()
            with torch.no_grad():
                probs_va = torch.sigmoid(model(X_va.to(device))).cpu().numpy()
            va_auc = roc_auc_score(y_va, probs_va) if len(np.unique(y_va)) > 1 else 0.0
            sched.step(va_auc)
            if va_auc > best_auc:
                best_auc, best_state, no_improve = va_auc, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probs_va2 = torch.sigmoid(model(X_va.to(device))).cpu().numpy()
        from common.thresholds import youden_threshold

        best_thr = youden_threshold(y_va, probs_va2)
        vm = binary_metrics(y_va, probs_va2, best_thr)
        state = {"mlp_state": best_state, "scaler": scaler, "latent": self.latent}
        return {
            "state_dict": state,
            "val_metrics": vm,
            "best_threshold": best_thr,
            "oof_probs": probs_va2,
            "oof_targets": y_va.astype(int),
            "oof_sids": [it["subject_id"] for it in items_va],
        }

    # ── evaluation hooks ─────────────────────────────────────────────────────
    def _model_for_state(self, state) -> nn.Module:
        if self._cached_state_id != id(state):
            m = self._build_mlp(state["latent"])
            m.load_state_dict(state["mlp_state"])
            m.eval()
            self._cached_model, self._cached_state_id = m, id(state)
        return self._cached_model

    def eval_split(self, state, bundle, threshold, *, device) -> Dict[str, Any]:
        X, y = self._records_to_X(bundle.items)
        X_s = torch.tensor(state["scaler"].transform(X), dtype=torch.float32)
        model = self._model_for_state(state)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_s.to(device))).cpu().numpy()
        metrics = binary_metrics(y, probs, threshold)
        return {
            **metrics,
            "probs": probs,
            "targets": y.astype(int),
            "subject_ids": np.array([it["subject_id"] for it in bundle.items]),
            "n_scans": np.array([it["n_scans"] for it in bundle.items]),
        }

    def truncate_to_n_visits(self, bundle, n) -> Bundle:
        items = [
            {**it, "zs": it["zs"][:n], "visit_months": it["visit_months"][:n], "n_scans": n}
            for it in bundle.items if it["n_scans"] >= n
        ]
        return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)

    def per_visit_probs(self, state, item, *, device):
        model = self._model_for_state(state)
        out = []
        with torch.no_grad():
            for t in range(1, item["n_scans"] + 1):
                X, _ = self._records_to_X([item], n_visits=t)
                X_s = torch.tensor(state["scaler"].transform(X), dtype=torch.float32)
                prob = torch.sigmoid(model(X_s.to(device))).item()
                out.append((item["visit_months"][t - 1], prob))
        return out

    # ── descriptors / persistence ────────────────────────────────────────────
    def model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "LongitudinalMLP",
            "input_dim": int(self.latent),
            "mlp_hidden_layers": self.mlp_hidden_layers,
            "mlp_dropout": self.mlp_dropout,
            "graph_pool": self.graph_pool,
            "encoder_arch": self.encoder_arch,
            "latent": int(self.latent),
            "gaae_latent": self.gaae_latent,
        }

    def source_files(self):
        root = _CLASSIFIER_ROOT
        return [
            root / "model" / "GAAE" / "models.py",
            root / "model" / "VGAE" / "models.py",
            root / "model" / "GELSTM" / "dataset.py",
            root / "adapters" / "gep.py",
        ]

    def model_state_for_save(self, state) -> Dict[str, Any]:
        return state["mlp_state"]

    def extra_artifacts(self, run_dir, state) -> None:
        with open(Path(run_dir) / "scaler.pkl", "wb") as f:
            pickle.dump(state["scaler"], f)

    def load_state(self, run_dir) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        ckpt = load_run_checkpoint(run_dir, device=self.device)
        mlp_state = model_state_from_checkpoint(ckpt)
        mc = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

        sc_path = run_dir / "scaler.pkl"
        if not sc_path.is_file():
            raise FileNotFoundError(f"GEP reload needs {sc_path}; not found.")
        with open(sc_path, "rb") as f:
            scaler = pickle.load(f)

        latent = int(mc.get("latent") or mc.get("input_dim") or scaler.mean_.shape[0])
        self.latent = latent
        self.feat_dim = latent
        return {"mlp_state": mlp_state, "scaler": scaler, "latent": latent}
