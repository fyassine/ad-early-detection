"""adapters/tfgn.py — TFGNAdapter (Temporal-First Graph Network classifier).

Implements the six-hook adapter contract consumed by
``LONGITUDINAL_COMMON_DELCODE.ipynb``, modelled directly on
``adapters/gelstm.py``. Per-fold ``StandardScaler`` on temporal embeddings,
``log Δt`` statistics, and centrality z-scoring all ride inside the composite
``state`` so the winning fold's statistics survive into test / early-detection /
trajectory hooks. ``patient_embeddings`` supports the mandatory cohort probe.
``extra_artifacts`` writes ``gate_scores.npy``, ``dual_scores.npy``, and
``cohort_tags.npy``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from common.crossval import Bundle
from configs.encoder import encoder_arm, resolve_encoder_init
from configs.tfgn import TFGNEvalConfig, TFGNTrainConfig
from model.GELSTM.dataset import LongitudinalSubjectDataset
from model.GELSTM.utils import compute_class_weights
from model.TFGN.dataset import (
    TFGNItem,
    prepare_tfgn_item,
)
from model.TFGN.models import TFGNClassifier
from model.TFGN.train import evaluate, make_batches, train_epoch
from sklearn.preprocessing import StandardScaler

from . import LongitudinalAdapter

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]


class TFGNAdapter(LongitudinalAdapter):
    """Temporal-First Graph Network classifier adapter."""

    model_tag = "tfgn"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        c = self.cfg
        # Model architecture
        self.n_rois = c.get("n_rois", 200)
        self.lstm_hidden = c.get("lstm_hidden", 64)
        self.lstm_layers = c.get("lstm_layers", 1)
        self.lstm_dropout = c.get("lstm_dropout", 0.3)
        self.use_time_delta = c.get("use_time_delta", True)
        self.gvae_hidden = c.get("gvae_hidden", 128)
        self.gvae_latent = c.get("gvae_latent", 64)
        self.gvae_heads = c.get("gvae_heads", 2)
        self.gvae_dropout = c.get("gvae_dropout", 0.3)
        self.cohort = str(c.get("cohort", "delcode")).lower()

        # TFGN ladder knobs
        self.node_lstm_init = str(c.get("node_lstm_init", "random")).lower()
        self.node_lstm_ckpt_path = c.get("node_lstm_ckpt_path")
        self.use_gate = c.get("use_gate", True)
        self.lambda_sparse = c.get("lambda_sparse", 0.1)
        self.lambda_drift = c.get("lambda_drift", 0.01)
        self.gate_rho = c.get("gate_rho", 0.15)
        self.recon_target = c.get("recon_target", "delta_a_topk")
        if self.recon_target in ("a_last", "delta_a_mse") and self.file_variant == "z_transformed":
            raise ValueError(
                f"recon_target={self.recon_target!r} assumes raw Pearson-range FC "
                f"(DOCS/temporal-first-ablation.md 'Reconstruction target' table: "
                f"a_last targets (A+1)/2 in [0,1], delta_a_mse targets ΔA/2 in [-1,1] — "
                f"both require A in [-1,1]), but file_variant='z_transformed' is not "
                f"range-bounded. Use recon_target='delta_a_topk' (the ladder default, "
                f"scale-free) or set file_variant='raw' if this arm genuinely needs "
                f"one of the two raw-FC-only targets."
            )
        self.lambda_recon = c.get("lambda_recon", 1.0)
        self.beta_kl = c.get("beta_kl", 1.0)
        self.free_bits = c.get("free_bits", 0.5)
        self.beta_warmup_epochs = c.get("beta_warmup_epochs", 5.0)
        self.change_mask_kappa = c.get("change_mask_kappa", 0.10)
        self.fusion = c.get("fusion", "concat_residual")
        self.readout = c.get("readout", "attention")
        self.dual_score = c.get("dual_score", True)
        self.lambda_cent = c.get("lambda_cent", 0.1)
        self.tau = c.get("tau", 0.05)
        self.cohort_conditioning = c.get("cohort_conditioning", "none")

        # Encoder arm (GVAE init)
        self.encoder_init = resolve_encoder_init(c.get("encoder_init"), c.get("freeze_encoder"))
        self.encoder_arm_info = encoder_arm(self.encoder_init)
        self.gvae_ckpt_path = c.get("gvae_ckpt_path") or self.gaae_ckpt_path

        # Training params
        self.learning_rate = c.get("learning_rate", 1e-3)
        self.weight_decay = c.get("weight_decay", 0.0)
        self.epochs = c.get("epochs", 100)
        self.early_stopping_patience = c.get("early_stopping_patience", 20)
        self.batch_size = c.get("batch_size", 16)
        self.use_class_cost_weights = c.get("use_class_cost_weights", True)
        self.grad_clip = c.get("grad_clip", 1.0)

        # Cohort window
        self.min_visits = c.get("min_visits")
        self.max_visits = c.get("max_visits")

        # Model cache
        self._cached_state_id: Optional[int] = None
        self._cached_model: Optional[nn.Module] = None

    # ── arch ────────────────────────────────────────────────────────────────
    def _build_model(self) -> TFGNClassifier:
        m = TFGNClassifier(
            n_rois=self.n_rois,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            lstm_dropout=self.lstm_dropout,
            use_time_delta=self.use_time_delta,
            gvae_hidden=self.gvae_hidden,
            gvae_latent=self.gvae_latent,
            gvae_heads=self.gvae_heads,
            gvae_dropout=self.gvae_dropout,
            use_gate=self.use_gate,
            recon_target=self.recon_target,
            fusion=self.fusion,
            readout=self.readout,
            dual_score=self.dual_score,
        ).to(self.device)

        # Handle node_lstm_init arm
        if (
            self.node_lstm_init in ("pretrained_frozen", "pretrained_finetuned")
            and self.node_lstm_ckpt_path
        ):
            m.load_node_lstm_weights(self.node_lstm_ckpt_path, device=self.device)
            if self.node_lstm_init == "pretrained_frozen":
                m.freeze_node_lstm()

        # Handle GVAE encoder_init arm
        if self.encoder_arm_info.loads_pretrained and self.gvae_ckpt_path and m.gvae is not None:
            m.load_gvae_weights(self.gvae_ckpt_path, device=self.device)
            if not self.encoder_arm_info.trains_encoder:
                m.freeze_gvae()

        return m

    def build_model(self) -> TFGNClassifier:
        m = self._build_model()
        trainable = sum(p.numel() for p in m.get_trainable_params())
        total = sum(p.numel() for p in m.parameters())
        print(
            f"TFGN model built: trainable={trainable:,}  total={total:,}  "
            f"node_lstm_init={self.node_lstm_init}  use_gate={self.use_gate}  "
            f"recon_target={self.recon_target}  fusion={self.fusion}  "
            f"readout={self.readout}"
        )
        return m

    # ── data ────────────────────────────────────────────────────────────────
    def prepare_data(self, df) -> Bundle:
        if "cohort" in getattr(df, "columns", []):
            from common.pooled_data import build_multicohort_bundle

            return build_multicohort_bundle(
                df,
                adjacency_k=self.adjacency_k,
                file_variant=self.file_variant,
                min_visits=self.min_visits,
                max_visits=self.max_visits,
            )
        ds = LongitudinalSubjectDataset(
            self.data_root,
            df,
            self.cohorts_csv,
            adjacency_k=self.adjacency_k,
            file_variant=self.file_variant,
            min_visits=self.min_visits,
            max_visits=self.max_visits,
            cohort=self.cohort,
        )
        items = [ds[i] for i in range(len(ds))]
        return Bundle(ds.get_labels(), ds.get_subject_ids(), items)

    def _prepare_tfgn_items(self, raw_items: List[Dict]) -> List[TFGNItem]:
        """Convert raw LongitudinalSubjectDataset items to TFGNItems (fails loud on invalid data)."""
        tfgn_items = []
        for it in raw_items:
            tfgn_item = prepare_tfgn_item(
                it,
                kappa=self.change_mask_kappa,
                gate_rho=self.gate_rho,
                tau_d=self.tau,
            )
            tfgn_items.append(tfgn_item)
        return tfgn_items

    # ── training ────────────────────────────────────────────────────────────
    def train_fold(
        self,
        bundle_tr,
        bundle_va,
        cfg,
        *,
        rng,
        device,
        epoch_log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        tr_items_raw, va_items_raw = bundle_tr.items, bundle_va.items

        # Convert to TFGNItems (each subject self-contained with per-node drift anchor)
        tr_items = self._prepare_tfgn_items(tr_items_raw)
        va_items = self._prepare_tfgn_items(va_items_raw)

        # Fit log_dt StandardScaler on training items
        all_log_dt = np.concatenate([it.log_dt.numpy() for it in tr_items], axis=0)
        log_dt_scaler = StandardScaler()
        log_dt_scaler.fit(all_log_dt.reshape(-1, 1))

        # Fit centrality z-scoring on training items
        all_cent = np.concatenate([it.strength_centrality.numpy() for it in tr_items], axis=0)
        cent_mean = float(all_cent.mean())
        cent_std = float(max(all_cent.std(), 1e-8))

        # Apply standardization to all items
        for it in tr_items + va_items:
            it.log_dt = torch.tensor(
                log_dt_scaler.transform(it.log_dt.numpy().reshape(-1, 1)).ravel(),
                dtype=torch.float32,
            )
            it.strength_centrality = (it.strength_centrality - cent_mean) / cent_std

        model = self._build_model()

        tr_labels = [it.label for it in tr_items]
        if self.use_class_cost_weights:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=compute_class_weights(tr_labels, device=device)
            )
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.get_trainable_params(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # Build train config for loss weights
        train_cfg = TFGNTrainConfig(
            lambda_sparse=self.lambda_sparse,
            lambda_drift=self.lambda_drift,
            gate_rho=self.gate_rho,
            recon_target=self.recon_target,
            lambda_recon=self.lambda_recon,
            beta_kl=self.beta_kl,
            free_bits=self.free_bits,
            beta_warmup_epochs=self.beta_warmup_epochs,
            change_mask_kappa=self.change_mask_kappa,
            lambda_cent=self.lambda_cent,
            use_gate=self.use_gate,
        )
        eval_cfg = TFGNEvalConfig()

        best_auc, best_state, no_improve = 0.0, None, 0
        for epoch in range(self.epochs):
            tr_batches = make_batches(tr_items, self.batch_size, shuffle=True, rng=rng)
            va_batches = make_batches(va_items, self.batch_size, shuffle=False)
            train_loss, train_loss_components = train_epoch(
                model,
                tr_batches,
                optimizer,
                criterion,
                device,
                cfg=train_cfg,
                grad_clip=self.grad_clip,
                epoch=epoch,
            )
            va = evaluate(model, va_batches, device, eval_cfg=eval_cfg)
            scheduler.step(va["auc"])
            if epoch_log_fn is not None:
                epoch_log_fn(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_auc": va["auc"],
                        "val_f1": va["f1"],
                        **{f"train_loss_{k}": v for k, v in train_loss_components.items()},
                    }
                )
            if va["auc"] > best_auc:
                best_auc = va["auc"]
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    break

        model.load_state_dict(best_state)
        final_va = evaluate(
            model,
            make_batches(va_items, self.batch_size, shuffle=False),
            device,
            eval_cfg=eval_cfg,
        )

        state = {
            "model_state": best_state,
            "log_dt_scaler_mean": log_dt_scaler.mean_.tolist(),
            "log_dt_scaler_scale": log_dt_scaler.scale_.tolist(),
            "cent_mean": cent_mean,
            "cent_std": cent_std,
        }
        return {
            "state_dict": state,
            "val_metrics": {k: final_va[k] for k in ("auc", "sensitivity", "specificity", "f1")},
            "best_threshold": final_va["best_threshold"],
            "oof_probs": final_va["probs"],
            "oof_targets": final_va["targets"],
            "oof_sids": list(final_va["subject_ids"]),
        }

    # ── evaluation hooks ────────────────────────────────────────────────────
    def _model_for_state(self, state) -> nn.Module:
        if self._cached_state_id != id(state):
            m = self._build_model()
            m.load_state_dict(state["model_state"])
            m.eval()
            self._cached_model, self._cached_state_id = m, id(state)
        return self._cached_model

    def _apply_state_normalization(self, items: List[TFGNItem], state: Dict) -> None:
        """Apply the winning fold's standardization to items in-place."""
        log_dt_mean = np.array(state["log_dt_scaler_mean"])
        log_dt_scale = np.array(state["log_dt_scaler_scale"])
        cent_mean = state["cent_mean"]
        cent_std = state["cent_std"]
        for it in items:
            raw = it.log_dt.numpy().reshape(-1, 1)
            normed = (raw - log_dt_mean) / log_dt_scale
            it.log_dt = torch.tensor(normed.ravel(), dtype=torch.float32)
            it.strength_centrality = (it.strength_centrality - cent_mean) / cent_std

    def patient_embeddings(
        self, state: Dict[str, Any], bundle: Bundle, *, device: Any
    ) -> np.ndarray:
        """Encode each subject into a pooled patient embedding vector for the cohort probe."""
        model = self._model_for_state(state)
        tfgn_items = self._prepare_tfgn_items(bundle.items)
        self._apply_state_normalization(tfgn_items, state)
        embs = []
        model.eval()
        with torch.no_grad():
            for it in tfgn_items:
                cond = torch.tensor([[it.age, float(it.sex)]], dtype=torch.float32, device=device)
                A0_ea = it.A0_edge_attr.to(device) if it.A0_edge_attr is not None else None
                emb = model.encode_patient(
                    it.X.to(device),
                    it.log_dt.to(device),
                    it.A0_edge_index.to(device),
                    A0_ea,
                    cond,
                )
                embs.append(emb.cpu().numpy().ravel())
        return np.stack(embs, axis=0)

    def eval_split(self, state, bundle, threshold, *, device) -> Dict[str, Any]:
        model = self._model_for_state(state)
        tfgn_items = self._prepare_tfgn_items(bundle.items)
        self._apply_state_normalization(tfgn_items, state)
        batches = make_batches(tfgn_items, self.batch_size, shuffle=False)
        eval_cfg = TFGNEvalConfig(threshold_mode="fixed", fixed_threshold=threshold)
        res = evaluate(model, batches, device, eval_cfg=eval_cfg)

        # Collect gate scores, dual scores, and cohort tags for extra_artifacts
        gate_scores_list = []
        dual_scores_list = []
        cohort_tags_list = []
        model.eval()
        with torch.no_grad():
            for it in tfgn_items:
                cond = torch.tensor([[it.age, float(it.sex)]], dtype=torch.float32, device=device)
                A0_ea = it.A0_edge_attr.to(device) if it.A0_edge_attr is not None else None
                out = model(
                    it.X.to(device),
                    it.log_dt.to(device),
                    it.A0_edge_index.to(device),
                    A0_ea,
                    cond,
                )
                if out["gate_scores"] is not None:
                    gate_scores_list.append(out["gate_scores"].cpu().numpy().ravel())
                if out["s_topo"] is not None:
                    dual_scores_list.append(out["s_topo"].cpu().numpy().ravel())
                cohort_tags_list.append(it.cohort)

        if gate_scores_list:
            state["gate_scores"] = np.stack(gate_scores_list, axis=0)
        if dual_scores_list:
            state["dual_scores"] = np.stack(dual_scores_list, axis=0)
        if cohort_tags_list:
            state["cohort_tags"] = np.array(cohort_tags_list)

        return res

    def truncate_to_n_visits(self, bundle, n) -> Bundle:
        items = [
            {
                **it,
                "graphs": it["graphs"][:n],
                "delta_t": it["delta_t"][:n],
                "visit_months": it["visit_months"][:n],
                "n_scans": n,
            }
            for it in bundle.items
            if it["n_scans"] >= n
        ]
        return Bundle(
            [it["label"] for it in items],
            [it["subject_id"] for it in items],
            items,
        )

    def per_visit_probs(self, state, item, *, device):
        model = self._model_for_state(state)
        out = []
        with torch.no_grad():
            for t in range(1, item["n_scans"] + 1):
                sub = {
                    **item,
                    "graphs": item["graphs"][:t],
                    "delta_t": item["delta_t"][:t],
                    "visit_months": item["visit_months"][:t],
                    "n_scans": t,
                }
                tfgn_item = prepare_tfgn_item(
                    sub,
                    kappa=self.change_mask_kappa,
                    gate_rho=self.gate_rho,
                    tau_d=self.tau,
                )
                self._apply_state_normalization([tfgn_item], state)
                tfgn_item.X = tfgn_item.X.to(device)
                tfgn_item.log_dt = tfgn_item.log_dt.to(device)
                tfgn_item.A0_edge_index = tfgn_item.A0_edge_index.to(device)
                if tfgn_item.A0_edge_attr is not None:
                    tfgn_item.A0_edge_attr = tfgn_item.A0_edge_attr.to(device)
                cond = torch.tensor(
                    [[tfgn_item.age, float(tfgn_item.sex)]],
                    dtype=torch.float32,
                    device=device,
                )
                result = model(
                    tfgn_item.X,
                    tfgn_item.log_dt,
                    tfgn_item.A0_edge_index,
                    tfgn_item.A0_edge_attr,
                    cond,
                )
                prob = torch.sigmoid(result["logits"]).item()
                out.append((item["visit_months"][t - 1], prob))
        return out

    # ── descriptors / persistence ───────────────────────────────────────────
    def model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "TFGNClassifier",
            "n_rois": self.n_rois,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "use_time_delta": self.use_time_delta,
            "gvae_hidden": self.gvae_hidden,
            "gvae_latent": self.gvae_latent,
            "gvae_heads": self.gvae_heads,
            "gvae_dropout": self.gvae_dropout,
            "node_lstm_init": self.node_lstm_init,
            "use_gate": self.use_gate,
            "recon_target": self.recon_target,
            "fusion": self.fusion,
            "readout": self.readout,
            "dual_score": self.dual_score,
            "lambda_sparse": self.lambda_sparse,
            "lambda_drift": self.lambda_drift,
            "gate_rho": self.gate_rho,
            "lambda_recon": self.lambda_recon,
            "beta_kl": self.beta_kl,
            "free_bits": self.free_bits,
            "change_mask_kappa": self.change_mask_kappa,
            "lambda_cent": self.lambda_cent,
            "tau": self.tau,
            "encoder_init": self.encoder_init,
            "cohort_conditioning": self.cohort_conditioning,
            "min_visits": self.min_visits,
            "max_visits": self.max_visits,
        }

    def source_files(self):
        root = _CLASSIFIER_ROOT
        return [
            root / "model" / "TFGN" / "models.py",
            root / "model" / "TFGN" / "dataset.py",
            root / "model" / "TFGN" / "layers.py",
            root / "model" / "TFGN" / "losses.py",
            root / "model" / "TFGN" / "train.py",
            root / "configs" / "tfgn.py",
            root / "adapters" / "tfgn.py",
        ]

    #: Non-weight entries of the composite ``state`` that ``load_state`` needs back.
    #: Single source of truth for the save/load round trip — see
    #: ``checkpoint_extras`` and ``load_state`` below, which both read it.
    STATE_NORMALIZATION_KEYS = (
        "log_dt_scaler_mean",
        "log_dt_scaler_scale",
        "cent_mean",
        "cent_std",
    )

    def model_state_for_save(self, state) -> Dict[str, Any]:
        return state["model_state"]

    def checkpoint_extras(self, state) -> Dict[str, Any]:
        """Persist the winning fold's normalisation statistics into the checkpoint.

        ``_apply_state_normalization`` re-applies the *training* fold's ``log Δt``
        scaler and centrality z-scoring to any split scored later, so a frozen read
        is only valid if these four numbers survive the save. They are not model
        weights, so ``model_state_for_save`` cannot carry them.
        """
        missing = [k for k in self.STATE_NORMALIZATION_KEYS if k not in state]
        if missing:
            raise ValueError(
                f"TFGN state is missing normalization statistics {missing}. "
                "train_fold must return them alongside 'model_state' — without "
                "them the checkpoint cannot be re-scored on a held-out split."
            )
        return {k: state[k] for k in self.STATE_NORMALIZATION_KEYS}

    def extra_artifacts(self, run_dir, state) -> None:
        run_dir = Path(run_dir)
        if "gate_scores" in state:
            np.save(run_dir / "gate_scores.npy", np.asarray(state["gate_scores"]))
        if "dual_scores" in state:
            np.save(run_dir / "dual_scores.npy", np.asarray(state["dual_scores"]))
        if "cohort_tags" in state:
            np.save(run_dir / "cohort_tags.npy", np.asarray(state["cohort_tags"]))

    def load_state(self, run_dir) -> Dict[str, Any]:
        """Rebuild the composite eval state from a saved run dir."""
        from . import load_run_checkpoint, model_state_from_checkpoint

        run_dir = Path(run_dir)
        ckpt = load_run_checkpoint(run_dir, device=self.device)
        model_state = model_state_from_checkpoint(ckpt)
        state = {"model_state": model_state}
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"{run_dir}: checkpoint is a bare state dict, not a full-state "
                "checkpoint — it carries no normalization statistics and cannot be "
                "re-scored on a held-out split."
            )
        missing = [k for k in self.STATE_NORMALIZATION_KEYS if k not in ckpt]
        if missing:
            raise KeyError(
                f"{run_dir}: checkpoint is missing normalization statistics "
                f"{missing}. Runs saved before `checkpoint_extras` was wired do not "
                "carry them; scoring without them would silently apply the wrong "
                "feature scaling. Backfill the checkpoint from the winning fold's "
                "training split first — see `scripts/backfill_tfgn_norm_stats.py` "
                "and DOCS/flipped/PLAN.md section G."
            )
        for key in self.STATE_NORMALIZATION_KEYS:
            state[key] = ckpt[key]
        return state
