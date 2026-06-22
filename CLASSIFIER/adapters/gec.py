"""
adapters/gec.py — GECAdapter (flattened-trajectory GEC-MLP).

Lifts ``notebooks/LONGITUDINAL/LONGITUDINAL_GEC_DELCODE.ipynb`` (full latent) and
its FDR sibling into the six-hook contract. Each visit is encoded once through the
frozen GAAE encoder; the per-subject visit sequence is flattened to a fixed-width
padded vector ``[z_1..z_Nmax, Δt_1..Δt_Nmax, mask_1..mask_Nmax]`` and classified by
an MLP. ``use_fdr=true`` keeps only the top-K Fisher latent dims.

Two stateful details the adapter carries:

  * ``MAX_VISITS`` is discovered while encoding the CV pool (the first
    ``prepare_data`` call) and reused for the test split, so the padded width — and
    hence the MLP input — is identical across splits.
  * The per-fold ``StandardScaler`` and ``dim_filter`` of the winning fold are
    bundled into the composite ``state`` (and re-emitted by ``extra_artifacts`` as
    the back-compat ``scaler.pkl`` / ``dim_filter.npy`` the comparison notebooks read).

Leakage note: unlike the original GEC-FDR notebook (which ranked FDR dims once on
the whole CV pool), the FDR dims here are selected **per fold from the training
subjects only** — the leakage-free convention already used by the GELSTM-FDR
notebook and ``common.fdr``.
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
from common.fdr import compute_fdr_filter
from model.GAAE.models import GraphAttentionAutoencoderConditioned
from model.GELSTM.dataset import LongitudinalSubjectDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import (
    LongitudinalAdapter,
    binary_metrics,
    load_run_checkpoint,
    model_state_from_checkpoint,
)

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]


class LongitudinalMLP(nn.Module):
    """MLP over the padded flat longitudinal vector (lifted from the GEC notebook)."""

    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GECAdapter(LongitudinalAdapter):
    """Frozen-GAAE → flattened latent trajectory → MLP classifier."""

    model_tag = "gec"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        c = self.cfg
        self.mlp_hidden_layers = c.get("mlp_hidden_layers", [256, 128, 64])
        self.mlp_dropout = c.get("mlp_dropout", 0.4)
        self.use_time_delta = c.get("use_time_delta", True)
        self.append_visit_mask = c.get("append_visit_mask", True)
        self.learning_rate = c.get("learning_rate", 1e-3)
        self.weight_decay = c.get("weight_decay", 1e-4)
        self.epochs = c.get("epochs", 80)
        self.early_stopping_patience = c.get("early_stopping_patience", 20)
        self.batch_size = c.get("batch_size", 32)
        self.grad_clip = c.get("grad_clip", 1.0)

        self._cfg_max_visits = c.get("max_visits")  # None -> auto-detect on CV pool
        self.max_visits: Optional[int] = None
        # number of latent dims fed to the MLP per visit (top_k under FDR, else full)
        self.k = self.top_k if self.use_fdr else self.gaae_latent
        self.feat_dim: Optional[int] = None

        self._encoder_model: Optional[GraphAttentionAutoencoderConditioned] = None
        self._cached_state_id: Optional[int] = None
        self._cached_model: Optional[nn.Module] = None

    # ── frozen GAAE encoder ─────────────────────────────────────────────────
    def _encoder(self) -> GraphAttentionAutoencoderConditioned:
        if self._encoder_model is None:
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

    def _encode_graph_full(self, enc, g) -> np.ndarray:
        ea = g.edge_attr.to(self.device) if g.edge_attr is not None else None
        z_nodes = enc.encode(g.x.to(self.device), g.edge_index.to(self.device), ea)
        return z_nodes.mean(0).cpu().numpy()  # full GAAE latent (dim selection is deferred)

    # ── data ────────────────────────────────────────────────────────────────
    def prepare_data(self, df) -> Bundle:
        ds = LongitudinalSubjectDataset(
            self.data_root, df, self.cohorts_csv,
            adjacency_k=self.adjacency_k, file_variant=self.file_variant,
        )
        enc = self._encoder()
        records: List[Dict[str, Any]] = []
        enc.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                records.append({
                    "subject_id": item["subject_id"],
                    "label": item["label"],
                    "n_scans": item["n_scans"],
                    "visit_months": list(item["visit_months"]),
                    "zs": [self._encode_graph_full(enc, g) for g in item["graphs"]],
                    "dts": list(item["delta_t"]),
                })
        # Lock the padded width on the first (CV-pool) call; reuse it for test.
        if self.max_visits is None:
            self.max_visits = (
                int(self._cfg_max_visits)
                if self._cfg_max_visits is not None
                else max((r["n_scans"] for r in records), default=1)
            )
            self.feat_dim = self._feature_dim()
            print(f"GEC: MAX_VISITS={self.max_visits}  k={self.k}  feat_dim={self.feat_dim}")
        return Bundle([r["label"] for r in records], [r["subject_id"] for r in records], records)

    def _feature_dim(self) -> int:
        d = self.k * self.max_visits
        if self.use_time_delta:
            d += self.max_visits
        if self.append_visit_mask:
            d += self.max_visits
        return d

    def _records_to_X(self, items, dim_filter, max_visits, n_visits=None):
        """Flatten subject records to padded (X, y) using ``dim_filter`` latent dims."""
        dim_filter = np.asarray(dim_filter)
        k = len(dim_filter)
        n = len(items)
        Xz = np.zeros((n, max_visits * k), dtype=np.float32)
        Xdt = np.zeros((n, max_visits), dtype=np.float32)
        Xm = np.zeros((n, max_visits), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        cap = max_visits if n_visits is None else min(n_visits, max_visits)
        for i, it in enumerate(items):
            T = min(it["n_scans"], cap)
            for t in range(T):
                Xz[i, t * k:(t + 1) * k] = it["zs"][t][dim_filter]
                Xdt[i, t] = it["dts"][t]
                Xm[i, t] = 1.0
            y[i] = float(it["label"])
        parts = [Xz]
        if self.use_time_delta:
            parts.append(Xdt)
        if self.append_visit_mask:
            parts.append(Xm)
        return np.concatenate(parts, axis=1).astype(np.float32), y.astype(np.float32)

    # ── arch ────────────────────────────────────────────────────────────────
    def _build_mlp(self, input_dim: int) -> LongitudinalMLP:
        return LongitudinalMLP(input_dim, self.mlp_hidden_layers, self.mlp_dropout).to(self.device)

    def build_model(self) -> LongitudinalMLP:
        if self.feat_dim is None:
            raise ValueError(
                "GECAdapter.build_model() called before prepare_data(); the MLP input "
                "width is only known after the CV pool is encoded. Call prepare_data "
                "on the CV pool first."
            )
        m = self._build_mlp(self.feat_dim)
        print(f"LongitudinalMLP: input={self.feat_dim}  params={sum(p.numel() for p in m.parameters()):,}")
        return m

    # ── training ────────────────────────────────────────────────────────────
    def train_fold(self, bundle_tr, bundle_va, cfg, *, rng, device) -> Dict[str, Any]:
        items_tr, items_va = bundle_tr.items, bundle_va.items

        if self.use_fdr:
            embs = np.stack([z for it in items_tr for z in it["zs"]])
            labs = np.array([it["label"] for it in items_tr for _ in it["zs"]], dtype=int)
            dim_filter, _ = compute_fdr_filter(embs, labs, self.top_k)
            print(f"  [FDR] top-{self.top_k} dims: {dim_filter.tolist()}")
        else:
            dim_filter = np.arange(self.gaae_latent)

        X_tr_raw, y_tr = self._records_to_X(items_tr, dim_filter, self.max_visits)
        X_va_raw, y_va = self._records_to_X(items_va, dim_filter, self.max_visits)
        scaler = StandardScaler().fit(X_tr_raw)
        feat_dim = X_tr_raw.shape[1]

        X_tr = torch.tensor(scaler.transform(X_tr_raw), dtype=torch.float32)
        X_va = torch.tensor(scaler.transform(X_va_raw), dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_w = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        model = self._build_mlp(feat_dim)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=7)

        tr_ds = torch.utils.data.TensorDataset(X_tr, y_tr_t)
        # drop_last only when the final batch would be a single sample (BatchNorm1d
        # rejects batch size 1 in train mode); otherwise keep every sample.
        drop_last = len(tr_ds) % self.batch_size == 1
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
        state = {
            "mlp_state": best_state,
            "scaler": scaler,
            "dim_filter": np.asarray(dim_filter),
            "feat_dim": feat_dim,
            "max_visits": self.max_visits,
        }
        return {
            "state_dict": state,
            "val_metrics": vm,
            "best_threshold": best_thr,
            "oof_probs": probs_va2,
            "oof_targets": y_va.astype(int),
            "oof_sids": [it["subject_id"] for it in items_va],
        }

    # ── evaluation hooks ────────────────────────────────────────────────────
    def _model_for_state(self, state) -> nn.Module:
        if self._cached_state_id != id(state):
            m = self._build_mlp(state["feat_dim"])
            m.load_state_dict(state["mlp_state"])
            m.eval()
            self._cached_model, self._cached_state_id = m, id(state)
        return self._cached_model

    def eval_split(self, state, bundle, threshold, *, device) -> Dict[str, Any]:
        X, y = self._records_to_X(bundle.items, state["dim_filter"], state["max_visits"])
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
            {**it,
             "zs": it["zs"][:n],
             "dts": it["dts"][:n],
             "visit_months": it["visit_months"][:n],
             "n_scans": n}
            for it in bundle.items if it["n_scans"] >= n
        ]
        return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)

    def per_visit_probs(self, state, item, *, device):
        model = self._model_for_state(state)
        out = []
        with torch.no_grad():
            for t in range(1, item["n_scans"] + 1):
                sub = [{**item, "zs": item["zs"][:t], "dts": item["dts"][:t], "n_scans": t}]
                X, _ = self._records_to_X(sub, state["dim_filter"], state["max_visits"])
                X_s = torch.tensor(state["scaler"].transform(X), dtype=torch.float32)
                prob = torch.sigmoid(model(X_s.to(device))).item()
                out.append((item["visit_months"][t - 1], prob))
        return out

    # ── descriptors / persistence ───────────────────────────────────────────
    def model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "LongitudinalMLP",
            "input_dim": int(self.feat_dim) if self.feat_dim is not None else None,
            "mlp_hidden_layers": self.mlp_hidden_layers,
            "mlp_dropout": self.mlp_dropout,
            "top_k": self.k,
            "max_visits": int(self.max_visits) if self.max_visits is not None else None,
            "use_time_delta": self.use_time_delta,
            "append_visit_mask": self.append_visit_mask,
            "use_fdr": self.use_fdr,
            "gaae_latent": self.gaae_latent, "gaae_heads": self.gaae_heads,
            "gaae_cond_dim": self.gaae_cond_dim, "gaae_dropout": self.gaae_dropout,
        }

    def source_files(self):
        root = _CLASSIFIER_ROOT
        return [
            root / "model" / "GAAE" / "models.py",
            root / "model" / "GAAE" / "dataset.py",
            root / "model" / "GAAE" / "utils.py",
            root / "model" / "GELSTM" / "dataset.py",
            root / "adapters" / "gec.py",
        ]

    def model_state_for_save(self, state) -> Dict[str, Any]:
        return state["mlp_state"]

    def extra_artifacts(self, run_dir, state) -> None:
        run_dir = Path(run_dir)
        np.save(run_dir / "dim_filter.npy", np.asarray(state["dim_filter"]))
        with open(run_dir / "scaler.pkl", "wb") as f:
            pickle.dump(state["scaler"], f)

    def load_state(self, run_dir) -> Dict[str, Any]:
        """Rebuild ``{mlp_state, scaler, dim_filter, feat_dim, max_visits}`` from a run dir."""
        run_dir = Path(run_dir)
        ckpt = load_run_checkpoint(run_dir, device=self.device)
        mlp_state = model_state_from_checkpoint(ckpt)
        mc = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

        df_path = run_dir / "dim_filter.npy"
        sc_path = run_dir / "scaler.pkl"
        if not df_path.is_file():
            raise FileNotFoundError(f"GEC reload needs {df_path}; not found.")
        if not sc_path.is_file():
            raise FileNotFoundError(f"GEC reload needs {sc_path}; not found.")
        dim_filter = np.load(df_path)
        with open(sc_path, "rb") as f:
            scaler = pickle.load(f)

        max_visits = mc.get("max_visits")
        if max_visits is None:
            raise ValueError(
                f"Checkpoint in {run_dir} has no model_config['max_visits']; cannot "
                "rebuild the GEC padded-vector width."
            )
        feat_dim = int(mc.get("input_dim") or scaler.mean_.shape[0])
        # Keep the adapter's own descriptors consistent with the reloaded model.
        self.max_visits = int(max_visits)
        self.feat_dim = feat_dim
        return {
            "mlp_state": mlp_state,
            "scaler": scaler,
            "dim_filter": dim_filter,
            "feat_dim": feat_dim,
            "max_visits": int(max_visits),
        }
