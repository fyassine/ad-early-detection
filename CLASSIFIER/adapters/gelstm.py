"""
adapters/gelstm.py — GELSTMAdapter (full-trajectory LSTM / GRU classifier).

Lifts the logic of ``notebooks/LONGITUDINAL/LONGITUDINAL_GELSTM_DELCODE.ipynb``
(and its FDR sibling) into the six-hook contract consumed by
``LONGITUDINAL_COMMON_DELCODE.ipynb``. One adapter covers four registry entries:

    * GELSTM (rnn_type=lstm) and GEGRU (rnn_type=gru) — cell type is a config flag.
    * FDR variants — ``use_fdr=true`` selects the top-K Fisher dims per fold and
      patches the recurrent core to a ``TOP_K (+Δt)`` input.

Per-fold ``StandardScaler`` standardisation of the pooled GAAE embeddings
(``standardize_features``, applied via ``model.set_feature_norm``) and the FDR
dim-filter both ride inside the returned composite ``state`` so the winning fold's
statistics survive into the test / early-detection / trajectory hooks.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from common.crossval import Bundle
from common.fdr import compute_fdr_filter
from configs.gelstm import EvalConfig
from model.GELSTM.dataset import LongitudinalSubjectDataset
from model.GELSTM.models import GELSTMClassifier
from model.GELSTM.train import evaluate, make_batches, train_epoch
from model.GELSTM.utils import compute_class_weights
from sklearn.preprocessing import StandardScaler

from . import LongitudinalAdapter, load_run_checkpoint, model_state_from_checkpoint

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]


class GELSTMAdapter(LongitudinalAdapter):
    """Full-trajectory recurrent classifier over per-visit GAAE embeddings."""

    model_tag = "gelstm"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        c = self.cfg
        self.rnn_type = c.get("rnn_type", "lstm")
        self.lstm_hidden = c.get("lstm_hidden", 32)
        self.lstm_layers = c.get("lstm_layers", 1)
        self.lstm_dropout = c.get("lstm_dropout", 0.3)
        self.classifier_hidden = c.get("classifier_hidden", 32)
        self.classifier_norm = c.get("classifier_norm", "none")
        self.use_time_delta = c.get("use_time_delta", True)
        self.graph_pool = c.get("graph_pool", "mean")
        self.freeze_encoder = c.get("freeze_encoder", True)
        self.standardize_features = c.get("standardize_features", True)
        self.learning_rate = c.get("learning_rate", 1e-3)
        self.weight_decay = c.get("weight_decay", 1e-4)
        self.epochs = c.get("epochs", 50)
        self.early_stopping_patience = c.get("early_stopping_patience", 15)
        self.batch_size = c.get("batch_size", 16)
        self.use_class_cost_weights = c.get("use_class_cost_weights", True)
        self.grad_clip = c.get("grad_clip", 1.0)
        # eval-model cache keyed by composite-state identity (avoids rebuilding +
        # reloading GAAE weights for every test subject in the trajectory hook).
        self._cached_state_id: Optional[int] = None
        self._cached_model: Optional[nn.Module] = None

    # ── arch ────────────────────────────────────────────────────────────────
    def _build_model(self) -> GELSTMClassifier:
        m = GELSTMClassifier(
            in_features=self.in_features, gaae_hidden=self.gaae_hidden,
            gaae_latent=self.gaae_latent, gaae_heads=self.gaae_heads,
            gaae_cond_dim=self.gaae_cond_dim, gaae_dropout=self.gaae_dropout,
            lstm_hidden=self.lstm_hidden, lstm_layers=self.lstm_layers,
            lstm_dropout=self.lstm_dropout, use_time_delta=self.use_time_delta,
            classifier_hidden=self.classifier_hidden, rnn_type=self.rnn_type,
            classifier_norm=self.classifier_norm,
        ).to(self.device)
        m.load_gaae_weights(self.gaae_ckpt_path, device=self.device)
        if not self.freeze_encoder:
            m.unfreeze_encoder()
        if self.use_fdr:
            self._patch_recurrent_core(m)
        return m

    def _patch_recurrent_core(self, m: GELSTMClassifier) -> None:
        """Resize the recurrent core to a TOP_K (+Δt) input for FDR-filtered runs.

        The encoder still emits ``gaae_latent`` dims; ``dim_filter`` selects the
        top-K *before* the recurrent core (see ``encode_batch_sequences``), so the
        LSTM/GRU and classifier must be rebuilt for the narrower input. Mirrors the
        patch in the FDR notebook.
        """
        lstm_in = self.top_k + (1 if self.use_time_delta else 0)
        rnn_cls = nn.GRU if str(self.rnn_type).lower() == "gru" else nn.LSTM
        m.lstm = rnn_cls(
            input_size=lstm_in, hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers, batch_first=True,
            dropout=self.lstm_dropout if self.lstm_layers > 1 else 0.0,
        ).to(self.device)
        from model.GELSTM.models import build_classifier_head

        m.classifier = build_classifier_head(
            self.lstm_hidden, self.classifier_hidden, self.lstm_dropout,
            self.classifier_norm,
        ).to(self.device)

    def build_model(self) -> GELSTMClassifier:
        m = self._build_model()
        trainable = sum(p.numel() for p in m.get_trainable_params())
        total = sum(p.numel() for p in m.parameters())
        print(
            f"Model built [{str(self.rnn_type).upper()} h{self.lstm_hidden} "
            f"L{self.lstm_layers}]: trainable={trainable:,}  total={total:,}  "
            f"use_fdr={self.use_fdr}"
        )
        return m

    # ── encoding helpers (FDR / standardisation) ────────────────────────────
    def _scan_embeddings(self, model, items):
        """Pooled per-visit GAAE embeddings (raw, identity-norm) + per-scan labels.

        Call before ``set_feature_norm`` so the returned embeddings are the raw
        pooled latents the FDR ratio and the StandardScaler are fit on.
        """
        embs: List[np.ndarray] = []
        labels: List[int] = []
        model.eval()
        with torch.no_grad():
            for item in items:
                for g in item["graphs"]:
                    ea = g.edge_attr.to(self.device) if g.edge_attr is not None else None
                    z = model.encode_visit(
                        g.x.to(self.device), g.edge_index.to(self.device), ea,
                        pool=self.graph_pool,
                    )
                    embs.append(z.cpu().numpy())
                    labels.append(int(item["label"]))
        return np.stack(embs), np.array(labels, dtype=int)

    def _eval_cfg(self, dim_filter, threshold: Optional[float] = None) -> EvalConfig:
        if threshold is None:
            return EvalConfig(
                use_time_delta=self.use_time_delta, graph_pool=self.graph_pool,
                dim_filter=dim_filter,
            )
        return EvalConfig(
            use_time_delta=self.use_time_delta, graph_pool=self.graph_pool,
            dim_filter=dim_filter, threshold_mode="fixed", fixed_threshold=threshold,
        )

    # ── data ────────────────────────────────────────────────────────────────
    def prepare_data(self, df) -> Bundle:
        ds = LongitudinalSubjectDataset(
            self.data_root, df, self.cohorts_csv,
            adjacency_k=self.adjacency_k, file_variant=self.file_variant,
        )
        items = [ds[i] for i in range(len(ds))]
        return Bundle(ds.get_labels(), ds.get_subject_ids(), items)

    # ── training ────────────────────────────────────────────────────────────
    def train_fold(self, bundle_tr, bundle_va, cfg, *, rng, device) -> Dict[str, Any]:
        tr_items, va_items = bundle_tr.items, bundle_va.items
        tr_labels = bundle_tr.labels

        model = self._build_model()

        dim_filter = None
        if self.use_fdr:
            embs, labs = self._scan_embeddings(model, tr_items)
            dim_filter, _ = compute_fdr_filter(embs, labs, self.top_k)
            print(f"  [FDR] top-{self.top_k} dims: {dim_filter.tolist()}")

        if self.standardize_features:
            scaler = self._fit_feature_scaler(model, tr_items)
            model.set_feature_norm(scaler.mean_, scaler.scale_)

        if self.use_class_cost_weights:
            criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(tr_labels, device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.get_trainable_params(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        eval_cfg = self._eval_cfg(dim_filter)

        best_auc, best_state, no_improve = 0.0, None, 0
        for epoch in range(self.epochs):
            tr_batches = make_batches(tr_items, self.batch_size, shuffle=True, rng=rng)
            va_batches = make_batches(va_items, self.batch_size, shuffle=False)
            train_epoch(model, tr_batches, optimizer, criterion, device,
                        grad_clip=self.grad_clip, eval_cfg=eval_cfg)
            va = evaluate(model, va_batches, device, eval_cfg=eval_cfg)
            scheduler.step(va["auc"])
            if va["auc"] > best_auc:
                best_auc, best_state, no_improve = va["auc"], copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    break

        model.load_state_dict(best_state)
        final_va = evaluate(
            model, make_batches(va_items, self.batch_size, shuffle=False),
            device, eval_cfg=eval_cfg,
        )
        state = {
            "model_state": best_state,
            "dim_filter": dim_filter.tolist() if dim_filter is not None else None,
        }
        return {
            "state_dict": state,
            "val_metrics": {k: final_va[k] for k in ("auc", "sensitivity", "specificity", "f1")},
            "best_threshold": final_va["best_threshold"],
            "oof_probs": final_va["probs"],
            "oof_targets": final_va["targets"],
            "oof_sids": list(final_va["subject_ids"]),
        }

    def _fit_feature_scaler(self, model, items) -> StandardScaler:
        embs, _ = self._scan_embeddings(model, items)
        return StandardScaler().fit(embs)

    # ── evaluation hooks ────────────────────────────────────────────────────
    def _model_for_state(self, state) -> nn.Module:
        if self._cached_state_id != id(state):
            m = self._build_model()
            m.load_state_dict(state["model_state"])
            m.eval()
            self._cached_model, self._cached_state_id = m, id(state)
        return self._cached_model

    def eval_split(self, state, bundle, threshold, *, device) -> Dict[str, Any]:
        model = self._model_for_state(state)
        batches = make_batches(bundle.items, self.batch_size, shuffle=False)
        return evaluate(model, batches, device,
                        eval_cfg=self._eval_cfg(state.get("dim_filter"), threshold))

    def truncate_to_n_visits(self, bundle, n) -> Bundle:
        items = [
            {**it,
             "graphs": it["graphs"][:n],
             "delta_t": it["delta_t"][:n],
             "visit_months": it["visit_months"][:n],
             "n_scans": n}
            for it in bundle.items if it["n_scans"] >= n
        ]
        return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)

    def per_visit_probs(self, state, item, *, device):
        from model.GELSTM.utils import encode_batch_sequences

        model = self._model_for_state(state)
        dim_filter = state.get("dim_filter")
        out = []
        with torch.no_grad():
            for t in range(1, item["n_scans"] + 1):
                sub = {**item, "graphs": item["graphs"][:t],
                       "delta_t": item["delta_t"][:t],
                       "visit_months": item["visit_months"][:t], "n_scans": t}
                packed, _, _ = encode_batch_sequences(
                    [sub], model, device,
                    use_time_delta=self.use_time_delta, graph_pool=self.graph_pool,
                    dim_filter=np.asarray(dim_filter) if dim_filter is not None else None,
                )
                prob = torch.sigmoid(model(packed)).item()
                out.append((item["visit_months"][t - 1], prob))
        return out

    # ── descriptors / persistence ───────────────────────────────────────────
    def model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "GELSTMClassifier",
            "in_features": self.in_features,
            "gaae_hidden": self.gaae_hidden, "gaae_latent": self.gaae_latent,
            "gaae_heads": self.gaae_heads, "gaae_cond_dim": self.gaae_cond_dim,
            "gaae_dropout": self.gaae_dropout,
            "rnn_type": self.rnn_type, "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers, "lstm_dropout": self.lstm_dropout,
            "classifier_hidden": self.classifier_hidden,
            "classifier_norm": self.classifier_norm,
            "use_time_delta": self.use_time_delta, "graph_pool": self.graph_pool,
            "freeze_encoder": self.freeze_encoder,
            "standardize_features": self.standardize_features,
            "use_fdr": self.use_fdr, "top_k": self.top_k if self.use_fdr else self.gaae_latent,
        }

    def source_files(self):
        root = _CLASSIFIER_ROOT
        return [
            root / "model" / "GELSTM" / "models.py",
            root / "model" / "GELSTM" / "dataset.py",
            root / "model" / "GELSTM" / "train.py",
            root / "model" / "GELSTM" / "utils.py",
            root / "model" / "GAAE" / "models.py",
            root / "adapters" / "gelstm.py",
        ]

    def model_state_for_save(self, state) -> Dict[str, Any]:
        return state["model_state"]

    def extra_artifacts(self, run_dir, state) -> None:
        if state.get("dim_filter") is not None:
            np.save(Path(run_dir) / "dim_filter.npy", np.asarray(state["dim_filter"]))

    def load_state(self, run_dir) -> Dict[str, Any]:
        """Rebuild ``{model_state, dim_filter}`` from a run dir.

        The per-fold StandardScaler is baked into ``model_state`` as the model's
        feature-norm buffers (see ``set_feature_norm``), so there is no scaler.pkl to
        reload — only the optional FDR ``dim_filter.npy``.
        """
        run_dir = Path(run_dir)
        ckpt = load_run_checkpoint(run_dir, device=self.device)
        model_state = model_state_from_checkpoint(ckpt)
        df_path = run_dir / "dim_filter.npy"
        dim_filter = np.load(df_path).tolist() if df_path.is_file() else None
        return {"model_state": model_state, "dim_filter": dim_filter}
