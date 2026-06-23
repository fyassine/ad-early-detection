"""
adapters/explain.py — per-model explain adapters for ``EXPLAIN_COMMON_DELCODE.ipynb``.

Parallel to the training adapters (``get_adapter``): ``get_explain_adapter(key)``
resolves an *explain* adapter whose contract the single, model-agnostic EXPLAIN
notebook calls. The shared notebook cells touch only this contract; genuinely
model-specific visuals run in capability-guarded "extra" cells that call
``adapter.extra(name, ctx)``.

The GEC / GELSTM explain adapters **reuse** their training adapter for the reload path
(``get_adapter(...).prepare_data / eval_split / per_visit_probs / load_state`` +
``read_run_threshold``) — identical to ``SANITY_VISIT_COUNT_CONFOUND.ipynb`` — so the
reloaded model reproduces the saved run exactly. GAAE loads its encoder checkpoint
directly (it is the feature extractor, not a downstream run).

Contract (see ``ExplainAdapter``):
    capabilities, load, prepare_bundles, latent_embeddings, diagnostics,
    pick_walkthrough_subject, baseline_visit, trace_forward, region_importance,
    predict_one, extra.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from common.crossval import Bundle
from model.GAAE.models import GraphAttentionAutoencoderConditioned

from . import get_adapter, read_run_threshold

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]

_REGISTRY: Dict[str, str] = {
    "gaae": "GAAEExplainAdapter",
    "vgae": "VGAEExplainAdapter",
    "gec": "GECExplainAdapter",
    "gep": "GEPExplainAdapter",
    "gelstm": "GELSTMExplainAdapter",
    "gegru": "GELSTMExplainAdapter",  # GRU is a config flag on the GELSTM run
}


def get_explain_adapter(name: str) -> type:
    """Resolve an explain-adapter key (``gaae|gec|gelstm|gegru``) to its class."""
    if not name:
        raise ValueError(
            f"get_explain_adapter() requires a non-empty name; known keys: {sorted(_REGISTRY)}"
        )
    key = str(name).strip().lower()
    cls_name = _REGISTRY.get(key)
    if cls_name is None:
        raise ValueError(
            f"Unknown explain adapter {name!r}. Known keys: {sorted(_REGISTRY)}."
        )
    return globals()[cls_name]


def resolve_source_run(source_experiment: str, *, classifier_root: Path = _CLASSIFIER_ROOT) -> Path:
    """Resolve ``outputs/<source_experiment>/latest`` (or ``latest.txt``) to a run dir."""
    src_root = classifier_root / "outputs" / source_experiment
    latest = src_root / "latest"
    if latest.exists():
        return latest.resolve()
    if (src_root / "latest.txt").is_file():
        return src_root / "runs" / (src_root / "latest.txt").read_text().strip()
    raise FileNotFoundError(
        f"No latest run for source_experiment={source_experiment!r} under {src_root}. "
        "Run the base experiment first (e.g. run_experiment.py --id <id>)."
    )


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class ExplainAdapter:
    """Base contract. Subclasses set ``capabilities`` and implement the hooks."""

    capabilities: set = set()
    model_tag: str = "explain"

    def __init__(
        self,
        *,
        gaae_ckpt_path: Optional[str] = None,
        gaae_hp: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        data_root: str,
        cohorts_csv: str,
        device: Any,
        rng: Any,
        source_experiment: Optional[str] = None,
        classifier_root: Path = _CLASSIFIER_ROOT,
    ) -> None:
        self.gaae_ckpt_path = str(gaae_ckpt_path) if gaae_ckpt_path else None
        self.gaae_hp = gaae_hp or {}
        self.cfg = dict(train_config or {})
        self.data_root = data_root
        self.cohorts_csv = cohorts_csv
        self.device = device
        self.rng = rng
        self.source_experiment = source_experiment
        self.classifier_root = Path(classifier_root)
        # GAAE arch knobs (shared by all adapters for latent viz / region maps)
        hp = self.gaae_hp
        self.in_features = 200
        self.gaae_hidden = hp.get("hidden_dim", 128)
        self.gaae_latent = hp.get("latent_dim", 64)
        self.gaae_heads = hp.get("num_heads", 2)
        self.gaae_cond_dim = hp.get("cond_dim", 2)
        self.gaae_dropout = hp.get("dropout", 0.3)
        self.adjacency_k = hp.get("adjacency_k", self.cfg.get("adjacency_k", 8))
        self.file_variant = hp.get("file_variant", self.cfg.get("file_variant", "z_transformed"))
        self.graph_pool = self.cfg.get("graph_pool", "mean")

    # ── hooks (overridden) ──────────────────────────────────────────────────
    def load(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def prepare_bundles(self, cv_pool_df, test_df) -> Tuple[Bundle, Bundle]:  # pragma: no cover
        raise NotImplementedError

    def latent_embeddings(self, bundle: Bundle) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        raise NotImplementedError

    def diagnostics(self, bundle: Bundle) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def pick_walkthrough_subject(self, bundle: Bundle) -> Dict[str, Any]:
        """Subject with the most visits (ties → first)."""
        items = list(bundle.items)
        if not items:
            raise ValueError("Cannot pick a walkthrough subject from an empty bundle.")
        n_scans = [int(it.get("n_scans", len(it.get("graphs", [it])))) for it in items]
        return items[int(np.argmax(n_scans))]

    def baseline_visit(self, subject: Dict[str, Any]) -> int:
        """Earliest visit month of a subject (the scan shown in the data journey)."""
        months = subject.get("visit_months") or [0]
        return int(months[0])

    def baseline_graph(self, subject: Dict[str, Any]):
        """The PyG ``Data`` for the subject's earliest visit (rebuilt if needed)."""
        graphs = subject.get("graphs")
        if graphs:
            return graphs[0]
        # GEC records store latents not graphs — rebuild the baseline graph.
        from common.pipeline_trace import nii_to_fc_to_graph
        from torch_geometric.data import Data

        tr = nii_to_fc_to_graph(
            subject["subject_id"], self.baseline_visit(subject),
            file_variant=self.file_variant, adjacency_k=self.adjacency_k,
        )
        ei = torch.tensor(tr["edge_index"], dtype=torch.long)
        x = torch.tensor(tr["x"], dtype=torch.float)
        return Data(x=x, edge_index=ei, edge_attr=torch.ones(ei.shape[1]))

    def trace_forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def region_importance(self, subject: Dict[str, Any]) -> np.ndarray:
        """Per-ROI (200,) importance from GAAE attention on the subject's baseline graph.

        Captum/GNNExplainer-free so it always runs; the ``extra`` cells refine it.
        """
        from model.GAAE.explain import aggregate_gat_attention

        enc = self._encoder()
        g = self.baseline_graph(subject)
        x = g.x.to(self.device)
        ei = g.edge_index.to(self.device)
        ea = g.edge_attr.to(self.device) if getattr(g, "edge_attr", None) is not None else None
        enc.eval()
        with torch.no_grad():
            _z, attn = enc.encode(x, ei, ea, return_attention=True)
        return aggregate_gat_attention(attn, x.shape[0])

    def predict_one(self, subject: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def extra(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    # ── shared helpers ──────────────────────────────────────────────────────
    def _encoder(self) -> GraphAttentionAutoencoderConditioned:  # overridden where a trained encoder exists
        raise NotImplementedError

    def _build_gaae_encoder(self) -> GraphAttentionAutoencoderConditioned:
        if not self.gaae_ckpt_path:
            raise ValueError("No GAAE checkpoint path available to build the encoder.")
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
        return enc

    def _subject_dataset(self, df):
        from model.GELSTM.dataset import LongitudinalSubjectDataset

        return LongitudinalSubjectDataset(
            self.data_root, df, self.cohorts_csv,
            adjacency_k=self.adjacency_k, file_variant=self.file_variant,
        )

    @torch.no_grad()
    def _pool_graph(self, enc, g) -> np.ndarray:
        x = g.x.to(self.device)
        ei = g.edge_index.to(self.device)
        ea = g.edge_attr.to(self.device) if getattr(g, "edge_attr", None) is not None else None
        z = enc.encode(x, ei, ea)
        if self.graph_pool == "max":
            return z.max(0).values.cpu().numpy()
        if self.graph_pool == "sum":
            return z.sum(0).cpu().numpy()
        return z.mean(0).cpu().numpy()


# --------------------------------------------------------------------------- #
# GAAE
# --------------------------------------------------------------------------- #
class GAAEExplainAdapter(ExplainAdapter):
    """Explain the GAAE encoder itself: latent space, attention, reconstruction, IG."""

    capabilities = {"reconstruction", "latent_ig", "gnn_explainer", "attention"}
    model_tag = "gaae"

    def load(self) -> Dict[str, Any]:
        self.encoder = self._build_gaae_encoder()
        return {"encoder": self.encoder, "gaae_checkpoint": self.gaae_ckpt_path}

    def _encoder(self):
        return self.encoder

    def prepare_bundles(self, cv_pool_df, test_df) -> Tuple[Bundle, Bundle]:
        def _bundle(df):
            ds = self._subject_dataset(df)
            items = [ds[i] for i in range(len(ds))]
            return Bundle(ds.get_labels(), ds.get_subject_ids(), items)

        return _bundle(cv_pool_df), _bundle(test_df)

    def latent_embeddings(self, bundle: Bundle):
        enc = self._encoder()
        X, labels, sids = [], [], []
        for it in bundle.items:
            X.append(self._pool_graph(enc, it["graphs"][0]))  # baseline visit
            labels.append(int(it["label"]))
            sids.append(it["subject_id"])
        return np.stack(X), np.array(labels, dtype=int), sids

    def diagnostics(self, bundle: Bundle) -> Dict[str, Any]:
        from model.GAAE.explain import reconstruct_features, reconstruction_quality
        from sklearn.metrics import roc_auc_score

        enc = self._encoder()
        per_subj, fidelity_r, labels = [], [], []
        for it in bundle.items:
            x, x_rec = reconstruct_features(enc, it["graphs"][0], device=self.device)
            q = reconstruction_quality(x, x_rec)  # one forward pass → error + fidelity
            per_subj.append(q["mse"])             # mean per-ROI reconstruction error (MSE)
            fidelity_r.append(q["pearson_r"])     # input↔reconstruction agreement
            labels.append(int(it["label"]))
        per_subj = np.array(per_subj)
        fidelity_r = np.array(fidelity_r)
        labels = np.array(labels)
        auc = (float(roc_auc_score(labels, per_subj))
               if len(np.unique(labels)) > 1 else float("nan"))
        return {
            "kind": "reconstruction",
            "recon_error": per_subj, "labels": labels,
            "recon_error_auc": auc,
            "mean_converter": float(per_subj[labels == 1].mean()) if (labels == 1).any() else float("nan"),
            "mean_stable": float(per_subj[labels == 0].mean()) if (labels == 0).any() else float("nan"),
            # Cohort reconstruction fidelity (Pearson r) — calibrates the single-subject
            # Stage 5b score against what is typical for this trained encoder.
            "recon_fidelity_r": fidelity_r,
            "fidelity_median": float(np.nanmedian(fidelity_r)) if fidelity_r.size else float("nan"),
            "fidelity_iqr": ([float(np.nanpercentile(fidelity_r, 25)),
                              float(np.nanpercentile(fidelity_r, 75))]
                             if fidelity_r.size else [float("nan"), float("nan")]),
        }

    def trace_forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        from model.GAAE.explain import trace_forward as gaae_trace

        return gaae_trace(self._encoder(), subject["graphs"][0], device=self.device)

    def extra(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        subject = ctx["subject"]
        g = subject["graphs"][0]
        enc = self._encoder()
        if name == "reconstruction":
            from model.GAAE.explain import per_node_reconstruction_error

            err = per_node_reconstruction_error(enc, g, device=self.device)
            return {"per_node_error": err}
        if name == "attention":
            return {"region_importance": self.region_importance(subject)}
        if name == "latent_ig":
            from model.GAAE.explain import latent_dim_integrated_gradients

            dim = int(ctx.get("latent_dim", 0))
            return {"node_importance": latent_dim_integrated_gradients(enc, g, dim, device=self.device),
                    "latent_dim": dim}
        if name == "gnn_explainer":
            from model.GAAE.explain import gnn_explain_latent_dim

            dim = int(ctx.get("latent_dim", 0))
            return {**gnn_explain_latent_dim(enc, g, dim, device=self.device), "latent_dim": dim}
        raise ValueError(f"GAAEExplainAdapter has no extra {name!r}.")


# --------------------------------------------------------------------------- #
# VGAE (variational encoder; adjacency reconstruction)
# --------------------------------------------------------------------------- #
class VGAEExplainAdapter(ExplainAdapter):
    """Explain the VGAE encoder: latent space, (GAT) attention, adjacency reconstruction.

    Parallels ``GAAEExplainAdapter`` but the reconstruction is over the *adjacency*
    (``sigmoid(z zᵀ)``), so ``diagnostics`` / the ``reconstruction`` extra report a
    per-ROI adjacency BCE rather than feature MSE. The encoder arch (``conv_type`` /
    ``hidden_dim`` / ``latent_dim`` / …) is read from the training config, falling
    back to ``gaae_hp``.
    """

    capabilities = {"reconstruction", "attention"}
    model_tag = "vgae"

    def _vgae_kw(self) -> Dict[str, Any]:
        c = self.cfg
        return dict(
            in_features=self.in_features,
            hidden_dim=c.get("hidden_dim", self.gaae_hidden),
            latent_dim=c.get("latent_dim", self.gaae_latent),
            conv_type=c.get("conv_type", "gcn"),
            num_heads=c.get("num_heads", self.gaae_heads),
            dropout=c.get("dropout", self.gaae_dropout),
            feature_decoder=bool(c.get("feature_decoder", False)),
        )

    def load(self) -> Dict[str, Any]:
        from model.VGAE.models import VariationalGraphAutoencoder

        if not self.gaae_ckpt_path:
            raise ValueError("VGAEExplainAdapter requires checkpoint_path to the VGAE encoder.")
        enc = VariationalGraphAutoencoder(**self._vgae_kw()).to(self.device)
        obj = torch.load(self.gaae_ckpt_path, map_location=self.device, weights_only=False)
        enc.load_state_dict(obj if isinstance(obj, dict) else obj.state_dict())
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        self.encoder = enc
        return {"encoder": type(enc).__name__, "conv_type": enc.conv_type,
                "vgae_checkpoint": self.gaae_ckpt_path}

    def _encoder(self):
        return self.encoder

    def prepare_bundles(self, cv_pool_df, test_df) -> Tuple[Bundle, Bundle]:
        def _bundle(df):
            ds = self._subject_dataset(df)
            items = [ds[i] for i in range(len(ds))]
            return Bundle(ds.get_labels(), ds.get_subject_ids(), items)

        return _bundle(cv_pool_df), _bundle(test_df)

    def latent_embeddings(self, bundle: Bundle):
        enc = self._encoder()
        X, labels, sids = [], [], []
        for it in bundle.items:
            X.append(self._pool_graph(enc, it["graphs"][0]))  # baseline visit
            labels.append(int(it["label"]))
            sids.append(it["subject_id"])
        return np.stack(X), np.array(labels, dtype=int), sids

    def diagnostics(self, bundle: Bundle) -> Dict[str, Any]:
        from model.GAAE.explain import reconstruction_quality
        from model.VGAE.explain import reconstruct_adjacency
        from sklearn.metrics import roc_auc_score

        enc = self._encoder()
        per_subj, fidelity_r, labels = [], [], []
        for it in bundle.items:
            adj_true, adj_hat = reconstruct_adjacency(enc, it["graphs"][0], device=self.device)
            q = reconstruction_quality(adj_true, adj_hat)  # adjacency MSE + input↔recon r
            per_subj.append(q["mse"])
            fidelity_r.append(q["pearson_r"])
            labels.append(int(it["label"]))
        per_subj = np.array(per_subj)
        fidelity_r = np.array(fidelity_r)
        labels = np.array(labels)
        auc = (float(roc_auc_score(labels, per_subj))
               if len(np.unique(labels)) > 1 else float("nan"))
        return {
            "kind": "reconstruction",
            "recon_error": per_subj, "labels": labels,
            "recon_error_auc": auc,
            "mean_converter": float(per_subj[labels == 1].mean()) if (labels == 1).any() else float("nan"),
            "mean_stable": float(per_subj[labels == 0].mean()) if (labels == 0).any() else float("nan"),
            # Adjacency reconstruction fidelity (Pearson r) — calibrates the single-subject
            # score against what is typical for this trained VGAE encoder.
            "recon_fidelity_r": fidelity_r,
            "fidelity_median": float(np.nanmedian(fidelity_r)) if fidelity_r.size else float("nan"),
            "fidelity_iqr": ([float(np.nanpercentile(fidelity_r, 25)),
                              float(np.nanpercentile(fidelity_r, 75))]
                             if fidelity_r.size else [float("nan"), float("nan")]),
        }

    def trace_forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        from model.VGAE.explain import trace_forward as vgae_trace

        return vgae_trace(self._encoder(), subject["graphs"][0], device=self.device)

    def extra(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        subject = ctx["subject"]
        g = subject["graphs"][0]
        enc = self._encoder()
        if name == "reconstruction":
            from model.VGAE.explain import per_node_adjacency_error

            return {"per_node_error": per_node_adjacency_error(enc, g, device=self.device)}
        if name == "attention":
            return {"region_importance": self.region_importance(subject)}
        raise ValueError(f"VGAEExplainAdapter has no extra {name!r}.")


# --------------------------------------------------------------------------- #
# Classifier base (reload via training adapter)
# --------------------------------------------------------------------------- #
class _ClassifierExplainAdapter(ExplainAdapter):
    """Shared reload + diagnostics for the GEC / GELSTM explain adapters."""

    _train_key: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tr = get_adapter(self._train_key)(
            gaae_ckpt_path=self.gaae_ckpt_path, gaae_hp=self.gaae_hp,
            train_config=self.cfg, data_root=self.data_root,
            cohorts_csv=self.cohorts_csv, device=self.device, rng=self.rng,
        )
        self.state: Optional[Dict[str, Any]] = None
        self.threshold: Optional[float] = None
        self.run_dir: Optional[Path] = None
        self.saved_auc: Optional[float] = None

    def load(self) -> Dict[str, Any]:
        if not self.source_experiment:
            raise ValueError(
                f"{type(self).__name__} requires source_experiment; set "
                "'source_experiment:' on the entry in the experiments/ directory."
            )
        self.run_dir = resolve_source_run(self.source_experiment, classifier_root=self.classifier_root)
        summary_path = self.run_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        gaae_ckpt = summary.get("gaae_checkpoint")
        if gaae_ckpt and Path(gaae_ckpt).exists():
            self.gaae_ckpt_path = gaae_ckpt
            self._tr.gaae_ckpt_path = gaae_ckpt
        self.state = self._tr.load_state(self.run_dir)
        self.threshold = read_run_threshold(self.run_dir)
        metrics = summary.get("metrics") or {}
        self.saved_auc = metrics.get("test_auc", summary.get("test_auc"))
        return {
            "run_dir": str(self.run_dir),
            "threshold": float(self.threshold),
            "saved_test_auc": self.saved_auc,
            "gaae_checkpoint": self.gaae_ckpt_path,
        }

    def prepare_bundles(self, cv_pool_df, test_df) -> Tuple[Bundle, Bundle]:
        cv = self._tr.prepare_data(cv_pool_df)
        test = self._tr.prepare_data(test_df)
        return cv, test

    def diagnostics(self, bundle: Bundle) -> Dict[str, Any]:
        from common.calibration import expected_calibration_error

        res = self._tr.eval_split(self.state, bundle, self.threshold, device=self.device)
        probs = np.asarray(res["probs"], dtype=float)
        targets = np.asarray(res["targets"], dtype=int)
        ece = expected_calibration_error(probs, targets) if probs.size else float("nan")
        return {
            "kind": "classification",
            "auc": float(res["auc"]),
            "sensitivity": float(res.get("sensitivity", float("nan"))),
            "specificity": float(res.get("specificity", float("nan"))),
            "f1": float(res.get("f1", float("nan"))),
            "probs": probs, "targets": targets,
            "threshold": float(self.threshold), "ece": float(ece),
            "reloaded_auc": float(res["auc"]), "saved_auc": self.saved_auc,
        }


# --------------------------------------------------------------------------- #
# GEC (flattened-trajectory MLP)
# --------------------------------------------------------------------------- #
class GECExplainAdapter(_ClassifierExplainAdapter):
    capabilities = {"probability", "trajectories", "early_detection", "latent_ig"}
    model_tag = "gec"
    _train_key = "gec"

    def _encoder(self):
        return self._tr._encoder()

    def latent_embeddings(self, bundle: Bundle):
        X, labels, sids = [], [], []
        for it in bundle.items:
            zs = np.stack(it["zs"]) if it.get("zs") else None
            X.append(zs.mean(0) if zs is not None else np.zeros(self.gaae_latent))
            labels.append(int(it["label"]))
            sids.append(it["subject_id"])
        return np.stack(X), np.array(labels, dtype=int), sids

    def trace_forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        pv = self._tr.per_visit_probs(self.state, subject, device=self.device)
        zs = np.stack(subject["zs"]) if subject.get("zs") else np.zeros((1, self.gaae_latent))
        return {
            "visit_embeddings": zs,
            "per_visit_probs": pv,
            "prob": float(pv[-1][1]) if pv else float("nan"),
            "visit_months": list(subject.get("visit_months", [])),
            "stages": [
                ("per-visit pooled latent", zs.shape),
                ("flattened + scaled MLP input", (self._tr.feat_dim,)),
                ("MLP logit → P", (1,)),
            ],
        }

    def predict_one(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        pv = self._tr.per_visit_probs(self.state, subject, device=self.device)
        prob = float(pv[-1][1]) if pv else float("nan")
        return {"prob": prob, "pred": int(prob >= self.threshold), "true": int(subject["label"]),
                "threshold": float(self.threshold)}

    def extra(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        subject = ctx["subject"]
        if name == "trajectories":
            return {"per_visit_probs": self._tr.per_visit_probs(self.state, subject, device=self.device)}
        if name == "early_detection":
            from common.early_detection import early_detection_table

            rows = early_detection_table(
                ctx["test_bundle"], self._tr.eval_split, self._tr.truncate_to_n_visits,
                self.state, self.threshold, device=self.device,
            )
            return {"early_detection": rows}
        if name == "latent_ig":
            from model.GEC.explain import mlp_input_attribution, unpack_flat_importance

            model = self._tr._model_for_state(self.state)
            X, _y = self._tr._records_to_X([subject], self.state["dim_filter"], self.state["max_visits"])
            X_s = self.state["scaler"].transform(X)[0]
            imp = mlp_input_attribution(model, X_s, device=self.device)
            unpacked = unpack_flat_importance(
                imp, k=len(self.state["dim_filter"]), max_visits=int(self.state["max_visits"]),
                use_time_delta=self._tr.use_time_delta, append_visit_mask=self._tr.append_visit_mask,
            )
            return unpacked
        raise ValueError(f"GECExplainAdapter has no extra {name!r}.")


# --------------------------------------------------------------------------- #
# GEP (pooled-embedding MLP)
# --------------------------------------------------------------------------- #
class GEPExplainAdapter(_ClassifierExplainAdapter):
    capabilities = {"probability", "trajectories", "early_detection"}
    model_tag = "gep"
    _train_key = "gep"

    def _encoder(self):
        return self._tr._encoder()

    def latent_embeddings(self, bundle: Bundle):
        X, labels, sids = [], [], []
        for it in bundle.items:
            zs = np.stack(it["zs"]) if it.get("zs") else None
            X.append(zs.mean(0) if zs is not None else np.zeros(self._tr.latent))
            labels.append(int(it["label"]))
            sids.append(it["subject_id"])
        return np.stack(X), np.array(labels, dtype=int), sids

    def trace_forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        pv = self._tr.per_visit_probs(self.state, subject, device=self.device)
        zs = np.stack(subject["zs"]) if subject.get("zs") else np.zeros((1, self._tr.latent))
        return {
            "visit_embeddings": zs,
            "per_visit_probs": pv,
            "prob": float(pv[-1][1]) if pv else float("nan"),
            "visit_months": list(subject.get("visit_months", [])),
            "stages": [
                ("per-visit pooled latent", zs.shape),
                ("mean-pooled subject embedding", (self._tr.latent,)),
                ("MLP logit → P", (1,)),
            ],
        }

    def predict_one(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        pv = self._tr.per_visit_probs(self.state, subject, device=self.device)
        prob = float(pv[-1][1]) if pv else float("nan")
        return {"prob": prob, "pred": int(prob >= self.threshold), "true": int(subject["label"]),
                "threshold": float(self.threshold)}

    def extra(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        subject = ctx["subject"]
        if name == "trajectories":
            return {"per_visit_probs": self._tr.per_visit_probs(self.state, subject, device=self.device)}
        if name == "early_detection":
            from common.early_detection import early_detection_table

            rows = early_detection_table(
                ctx["test_bundle"], self._tr.eval_split, self._tr.truncate_to_n_visits,
                self.state, self.threshold, device=self.device,
            )
            return {"early_detection": rows}
        raise ValueError(f"GEPExplainAdapter has no extra {name!r}.")


# --------------------------------------------------------------------------- #
# GELSTM / GEGRU (recurrent)
# --------------------------------------------------------------------------- #
class GELSTMExplainAdapter(_ClassifierExplainAdapter):
    capabilities = {
        "probability", "trajectories", "early_detection",
        "visit_occlusion", "hidden_state", "temporal_ablation", "latent_ig",
    }
    model_tag = "gelstm"
    _train_key = "gelstm"

    def _classifier_model(self):
        return self._tr._model_for_state(self.state)

    def _encoder(self):
        return self._classifier_model().encoder

    def _dim_filter(self):
        df = self.state.get("dim_filter")
        return np.asarray(df) if df is not None else None

    def latent_embeddings(self, bundle: Bundle):
        enc = self._encoder()
        X, labels, sids = [], [], []
        for it in bundle.items:
            X.append(self._pool_graph(enc, it["graphs"][0]))
            labels.append(int(it["label"]))
            sids.append(it["subject_id"])
        return np.stack(X), np.array(labels, dtype=int), sids

    def trace_forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        from model.GELSTM.explain import trace_forward as rnn_trace

        return rnn_trace(
            self._classifier_model(), subject, device=self.device,
            use_time_delta=self._tr.use_time_delta, graph_pool=self.graph_pool,
            dim_filter=self._dim_filter(),
        )

    def predict_one(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        tr = self.trace_forward(subject)
        prob = float(tr["prob"])
        return {"prob": prob, "pred": int(prob >= self.threshold), "true": int(subject["label"]),
                "threshold": float(self.threshold)}

    def extra(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        subject = ctx["subject"]
        if name == "trajectories":
            return {"per_visit_probs": self._tr.per_visit_probs(self.state, subject, device=self.device)}
        if name == "early_detection":
            from common.early_detection import early_detection_table

            rows = early_detection_table(
                ctx["test_bundle"], self._tr.eval_split, self._tr.truncate_to_n_visits,
                self.state, self.threshold, device=self.device,
            )
            return {"early_detection": rows}
        if name == "hidden_state":
            from model.GELSTM.explain import hidden_state_trajectory

            h = hidden_state_trajectory(
                self._classifier_model(), subject, device=self.device,
                use_time_delta=self._tr.use_time_delta, graph_pool=self.graph_pool,
                dim_filter=self._dim_filter(),
            )
            return {"hidden_states": h, "visit_months": list(subject.get("visit_months", []))}
        if name == "visit_occlusion":
            return {"occlusion": self._visit_occlusion(subject)}
        if name == "temporal_ablation":
            return {"ablation": self._temporal_ablation(ctx["test_bundle"])}
        if name == "latent_ig":
            from model.GELSTM.explain import sequence_integrated_gradients

            return sequence_integrated_gradients(
                self._classifier_model(), subject, device=self.device,
                use_time_delta=self._tr.use_time_delta, graph_pool=self.graph_pool,
                dim_filter=self._dim_filter(),
            )
        raise ValueError(f"GELSTMExplainAdapter has no extra {name!r}.")

    def _visit_occlusion(self, subject) -> List[Dict[str, Any]]:
        """Drop each visit in turn; record the change in P(converter)."""
        base = self.predict_one(subject)["prob"]
        months = list(subject["visit_months"])
        out = []
        for t in range(subject["n_scans"]):
            keep = [i for i in range(subject["n_scans"]) if i != t]
            if len(keep) < 1:
                continue
            sub = {
                **subject,
                "graphs": [subject["graphs"][i] for i in keep],
                "delta_t": [subject["delta_t"][i] for i in keep],
                "visit_months": [months[i] for i in keep],
                "n_scans": len(keep),
            }
            p = self.predict_one(sub)["prob"]
            out.append({"dropped_visit_month": months[t], "prob_without": p,
                        "delta": float(base - p)})
        return out

    def _temporal_ablation(self, bundle: Bundle) -> Dict[str, float]:
        """Test-set AUC under Δt-zeroed and visit-order-shuffled evaluation."""
        from configs.gelstm import EvalConfig
        from model.GELSTM.train import evaluate, make_batches

        model = self._classifier_model()
        df = self._dim_filter()
        batches = make_batches(bundle.items, self._tr.batch_size, shuffle=False)

        def _auc(eval_cfg):
            return float(evaluate(model, batches, self.device, eval_cfg=eval_cfg)["auc"])

        base = EvalConfig(use_time_delta=self._tr.use_time_delta, graph_pool=self.graph_pool,
                          dim_filter=df, threshold_mode="fixed", fixed_threshold=self.threshold)
        zero_dt = EvalConfig(use_time_delta=self._tr.use_time_delta, zero_time_delta=True,
                             graph_pool=self.graph_pool, dim_filter=df,
                             threshold_mode="fixed", fixed_threshold=self.threshold)
        shuffled = EvalConfig(use_time_delta=self._tr.use_time_delta, graph_pool=self.graph_pool,
                              dim_filter=df, shuffle_order=True, shuffle_rng=self.rng,
                              threshold_mode="fixed", fixed_threshold=self.threshold)
        return {"auc_full": _auc(base), "auc_zero_dt": _auc(zero_dt), "auc_shuffled": _auc(shuffled)}


__all__ = ["get_explain_adapter", "resolve_source_run", "ExplainAdapter",
           "GAAEExplainAdapter", "VGAEExplainAdapter", "GECExplainAdapter",
           "GEPExplainAdapter", "GELSTMExplainAdapter"]
