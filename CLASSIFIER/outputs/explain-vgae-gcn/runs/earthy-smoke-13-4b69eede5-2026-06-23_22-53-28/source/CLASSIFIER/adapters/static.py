"""
adapters/static.py — per-model implementations of the STATIC pretraining contract.

`STATIC_COMMON_DELCODE.ipynb` is a single, model-agnostic notebook for pretraining
a whole-brain reconstruction autoencoder (GAAE, VGAE). Every model-specific
operation is funnelled through a *static adapter* — a stateful object implementing:

    build_model, run_training, compute_sample_error, latest_checkpoint_tag

plus the `source_files` descriptor. The shared notebook cells call ONLY these, so
they stay identical across encoder architectures.

This is a separate registry/base class from `adapters/__init__.py`'s
`LongitudinalAdapter`/`get_adapter`: that contract is for *downstream* models
consuming a frozen, already-pretrained encoder (GELSTM/GEC/GEP); this one is for
*pretraining* the encoder itself. The two are different enough (constructor
inputs, hook signatures, save-state shape) that conflating them under one
registry would blur which contract a given adapter actually satisfies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

# Make `model.*` / `common.*` importable the same way the notebooks do (CLASSIFIER
# root on sys.path), and `CLASSIFIER.*` importable the way the tests do (repo root).
_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CLASSIFIER_ROOT.parent
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Registry name -> "module:ClassName". Imported lazily so that merely importing
# this module does not drag in torch / the model code.
_REGISTRY: Dict[str, str] = {
    "gaae": "CLASSIFIER.adapters.static:GAAEStaticAdapter",
    "vgae": "CLASSIFIER.adapters.static:VGAEStaticAdapter",
}


def get_static_adapter(name: str) -> type:
    """Resolve a registry key (e.g. ``MODEL`` / ``ADAPTER``) to its adapter class.

    Case-insensitive. Raises ``ValueError`` listing the known keys when the name is
    unknown — never silently falls back to a default adapter (see
    ``.claude/rules/errors.md``).
    """
    if not name:
        raise ValueError(
            f"get_static_adapter() requires a non-empty adapter name; known keys: "
            f"{sorted(_REGISTRY)}"
        )
    key = str(name).strip().lower()
    target = _REGISTRY.get(key)
    if target is None:
        raise ValueError(
            f"Unknown static adapter {name!r}. Known adapter keys: {sorted(_REGISTRY)}. "
            "Set 'adapter:' on the experiment in the experiments/ directory (defaults to MODEL)."
        )
    module_path, _, cls_name = target.partition(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


class StaticAdapter:
    """Base class: holds the merged training config and declares the hook contract.

    Concrete adapters (``GAAEStaticAdapter``, ``VGAEStaticAdapter``) fill in the
    hooks. Unlike ``LongitudinalAdapter`` (which loads a frozen, already-pretrained
    encoder), a static adapter builds and trains the encoder itself, so its
    constructor needs only the merged hyperparameter config, device, and rng.
    """

    model_tag: str = "static"

    def __init__(self, *, cfg: Dict[str, Any], device: Any, rng: Any) -> None:
        self.cfg: Dict[str, Any] = dict(cfg or {})
        self.device = device
        self.rng = rng

    # ── contract hooks (overridden) ─────────────────────────────────────────
    def build_model(self, in_features: int) -> Tuple[Any, Dict[str, Any]]:  # pragma: no cover
        """Instantiate the model and return (model, model_config)."""
        raise NotImplementedError

    def run_training(
        self, model, optimizer, train_loader, val_loader, wandb_run
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # pragma: no cover
        """Train one model to convergence. Returns (best_state_dict, history)."""
        raise NotImplementedError

    def compute_sample_error(
        self, sample, model
    ) -> Union[float, Dict[str, float]]:  # pragma: no cover
        """Per-sample reconstruction error: float (Total Error) or a dict with 'total_error'."""
        raise NotImplementedError

    def latest_checkpoint_tag(self) -> str:  # pragma: no cover - overridden
        """Tag passed to common.checkpoints.update_latest_checkpoint."""
        raise NotImplementedError

    # ── descriptors (overridden) ────────────────────────────────────────────
    def source_files(self):  # pragma: no cover - overridden
        raise NotImplementedError


class GAAEStaticAdapter(StaticAdapter):
    """Pretrains the GAAE (Graph Attention Autoencoder, FiLM-conditioned)."""

    model_tag = "gaae"

    def build_model(self, in_features: int):
        from model.GAAE.models import GraphAttentionAutoencoderConditioned

        c = self.cfg
        model = GraphAttentionAutoencoderConditioned(
            in_features=in_features,
            hidden_dim=c.get("hidden_dim", 128),
            out_features=c.get("latent_dim", 64),
            cond_dim=c.get("cond_dim", 2),
            num_heads=c.get("num_heads", 2),
            dropout=c.get("dropout", 0.3),
        ).to(self.device)
        model_config = {
            "model_type": model.__class__.__name__,
            "in_features": in_features,
            "hidden_dim": c.get("hidden_dim", 128),
            "latent_dim": c.get("latent_dim", 64),
            "cond_dim": c.get("cond_dim", 2),
            "num_heads": c.get("num_heads", 2),
            "dropout": c.get("dropout", 0.3),
        }
        return model, model_config

    def run_training(self, model, optimizer, train_loader, val_loader, wandb_run):
        from model.GAAE.train import train_model_with_val_notebook_train_loss

        c = self.cfg
        return train_model_with_val_notebook_train_loss(
            model,
            train_loader,
            val_loader,
            optimizer,
            self.device,
            batch_size=c.get("batch_size", 64),
            learning_rate=c.get("learning_rate", 1e-3),
            model_config={},
            adj_loss_weight=c.get("adj_loss_weight", 1.0),
            epochs=c.get("epochs", 500),
            early_stopping_patience=c.get("early_stopping_patience", 25),
            wandb_run=wandb_run,
        )

    def compute_sample_error(self, sample, model):
        from model.GAAE.losses import compute_sample_reconstruction_error

        x_err, adj_err, total_err = compute_sample_reconstruction_error(
            sample, model, self.device, self.cfg.get("adj_loss_weight", 1.0)
        )
        return {"x_error": x_err, "adj_error": adj_err, "total_error": total_err}

    def latest_checkpoint_tag(self) -> str:
        return "GAAE"

    def source_files(self):
        root = _CLASSIFIER_ROOT
        return [
            root / "model" / "GAAE" / "models.py",
            root / "model" / "GAAE" / "train.py",
            root / "model" / "GAAE" / "losses.py",
        ]


class VGAEStaticAdapter(StaticAdapter):
    """Pretrains the VGAE (Variational Graph Autoencoder, GCN or GAT backbone)."""

    model_tag = "vgae"

    def build_model(self, in_features: int):
        from model.VGAE.models import VariationalGraphAutoencoder

        c = self.cfg
        model = VariationalGraphAutoencoder(
            in_features=in_features,
            hidden_dim=c.get("hidden_dim", 128),
            latent_dim=c.get("latent_dim", 64),
            conv_type=c.get("conv_type", "gcn"),
            num_heads=c.get("num_heads", 2),
            dropout=c.get("dropout", 0.3),
            feature_decoder=bool(c.get("feature_decoder", False)),
        ).to(self.device)
        model_config = {
            "model_type": model.__class__.__name__,
            "in_features": in_features,
            "hidden_dim": c.get("hidden_dim", 128),
            "latent_dim": c.get("latent_dim", 64),
            "conv_type": c.get("conv_type", "gcn"),
            "num_heads": c.get("num_heads", 2),
            "dropout": c.get("dropout", 0.3),
            "beta": c.get("beta", 1.0),
            "beta_warmup_epochs": c.get("beta_warmup_epochs", 0),
            "free_bits": c.get("free_bits", 0.0),
            "feature_decoder": bool(c.get("feature_decoder", False)),
            "feature_loss_weight": c.get("feature_loss_weight", 0.0),
        }
        return model, model_config

    def run_training(self, model, optimizer, train_loader, val_loader, wandb_run):
        from model.VGAE.train import train_vgae_with_val

        c = self.cfg
        return train_vgae_with_val(
            model,
            train_loader,
            val_loader,
            optimizer,
            self.device,
            beta=c.get("beta", 1.0),
            beta_warmup_epochs=c.get("beta_warmup_epochs", 0),
            free_bits=c.get("free_bits", 0.0),
            feature_loss_weight=c.get("feature_loss_weight", 0.0),
            epochs=c.get("epochs", 500),
            early_stopping_patience=c.get("early_stopping_patience", 25),
            wandb_run=wandb_run,
        )

    def compute_sample_error(self, sample, model):
        from model.VGAE.losses import compute_sample_reconstruction_error

        c = self.cfg
        recon_err, kl_err, feat_err, total_err = compute_sample_reconstruction_error(
            sample,
            model,
            self.device,
            c.get("beta", 1.0),
            free_bits=c.get("free_bits", 0.0),
            feature_loss_weight=c.get("feature_loss_weight", 0.0),
        )
        return {
            "recon_error": recon_err,
            "kl_error": kl_err,
            "feat_error": feat_err,
            "total_error": total_err,
        }

    def latest_checkpoint_tag(self) -> str:
        conv_type = str(self.cfg.get("conv_type", "gcn")).upper()
        tag = f"VGAE_{conv_type}"
        checkpoint_tag = str(self.cfg.get("checkpoint_tag", "") or "").strip().upper()
        return f"{tag}_{checkpoint_tag}" if checkpoint_tag else tag

    def source_files(self):
        root = _CLASSIFIER_ROOT
        return [
            root / "model" / "VGAE" / "models.py",
            root / "model" / "VGAE" / "train.py",
            root / "model" / "VGAE" / "losses.py",
        ]


__all__ = [
    "get_static_adapter",
    "StaticAdapter",
    "GAAEStaticAdapter",
    "VGAEStaticAdapter",
]
