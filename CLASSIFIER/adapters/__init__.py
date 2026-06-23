"""
adapters/ — per-model implementations of the longitudinal adapter contract.

`LONGITUDINAL_COMMON_DELCODE.ipynb` is a single, model-agnostic notebook. Every
model-specific operation is funnelled through an *adapter*: a stateful object that
implements the six contract hooks the SHARED notebook cells call —

    build_model, prepare_data, train_fold, eval_split,
    truncate_to_n_visits, per_visit_probs

— plus a few descriptors the save cell needs (`model_config`, `source_files`,
`model_tag`, `model_state_for_save`, `extra_artifacts`). The shared cells call
ONLY these, so they stay identical across models; adding a new longitudinal model
means writing one adapter class, not a new notebook.

Adapters are stateful by necessity: the GEC-MLP discovers `MAX_VISITS` (and hence
its input width) while encoding the CV pool, and the per-fold `StandardScaler` /
FDR dim-filter of the winning fold must survive into the test-set hooks. The hooks
are bound methods, which the `common.crossval` / `common.early_detection` callables
accept transparently.

Selection: `get_adapter(name)` resolves the registry key injected as the `ADAPTER`
papermill param (which defaults to the experiment's `MODEL`). FDR / GRU variants
are config flags (`use_fdr`, `rnn_type`) on the GEC / GELSTM adapters, not separate
adapters.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

# Make `model.*` / `common.*` importable the same way the notebooks do (CLASSIFIER
# root on sys.path), and `CLASSIFIER.*` importable the way the tests do (repo root).
_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CLASSIFIER_ROOT.parent
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Registry name -> "module:ClassName". Imported lazily so that merely importing
# this package (e.g. from the lightweight experiment runner) does not drag in
# torch / the model code.
_REGISTRY: Dict[str, str] = {
    "gelstm": "CLASSIFIER.adapters.gelstm:GELSTMAdapter",
    "gegru": "CLASSIFIER.adapters.gelstm:GELSTMAdapter",  # alias; rnn_type=gru via config
    "gec": "CLASSIFIER.adapters.gec:GECAdapter",
    "gep": "CLASSIFIER.adapters.gep:GEPAdapter",
}


def get_adapter(name: str) -> type:
    """Resolve a registry key (e.g. ``MODEL`` / ``ADAPTER``) to its adapter class.

    Case-insensitive. Raises ``ValueError`` listing the known keys when the name is
    unknown — never silently falls back to a default adapter (see
    ``.claude/rules/errors.md``).
    """
    if not name:
        raise ValueError(
            f"get_adapter() requires a non-empty adapter name; known keys: "
            f"{sorted(_REGISTRY)}"
        )
    key = str(name).strip().lower()
    target = _REGISTRY.get(key)
    if target is None:
        raise ValueError(
            f"Unknown adapter {name!r}. Known adapter keys: {sorted(_REGISTRY)}. "
            "Set 'adapter:' on the experiment in the experiments/ directory (defaults to MODEL)."
        )
    module_path, _, cls_name = target.partition(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _find_checkpoint_file(run_dir: Path) -> Path:
    """Locate the saved weights in a run dir: ``checkpoint_*.pth`` then ``model_*.pth``."""
    run_dir = Path(run_dir)
    cands = sorted(run_dir.glob("checkpoint_*.pth")) or sorted(run_dir.glob("model_*.pth"))
    if not cands:
        raise FileNotFoundError(
            f"No checkpoint_*.pth / model_*.pth found in {run_dir}. Cannot reload a "
            "trained model from this run."
        )
    return cands[0]


def load_run_checkpoint(run_dir, *, device: Any = "cpu"):
    """Load a run's saved checkpoint (full-state dict or a bare state_dict).

    ``weights_only=False`` is used deliberately — we only ever load our own runs (see
    ``.claude/rules/checkpoints.md``). Use ``model_state_from_checkpoint`` to extract
    the plain weights regardless of which schema was written.
    """
    import torch

    path = _find_checkpoint_file(Path(run_dir))
    return torch.load(path, map_location=device, weights_only=False)


def model_state_from_checkpoint(ckpt):
    """Plain ``state_dict`` from either checkpoint schema (full-state or bare)."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def read_run_threshold(run_dir, *, ckpt=None) -> float:
    """Validation-derived threshold for a saved run, never re-optimised on test.

    Prefers the checkpoint's ``best_threshold``; falls back to
    ``run_summary.json``'s ``active_threshold``. Raises if neither is present (a
    silent default would leak — see ``.claude/rules/evaluation.md``).
    """
    run_dir = Path(run_dir)
    if ckpt is None:
        ckpt = load_run_checkpoint(run_dir)
    if isinstance(ckpt, dict) and ckpt.get("best_threshold") is not None:
        return float(ckpt["best_threshold"])
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        import json

        thr = json.loads(summary_path.read_text()).get("active_threshold")
        if thr is not None:
            return float(thr)
    raise ValueError(
        f"No threshold for run {run_dir}: checkpoint lacks 'best_threshold' and "
        "run_summary.json lacks 'active_threshold'."
    )


def binary_metrics(targets, probs, threshold: float) -> Dict[str, float]:
    """AUC / sensitivity / specificity / F1 of ``probs >= threshold`` vs ``targets``.

    Shared by adapters whose eval path is plain numpy (the GEC-MLP). The GELSTM
    adapter reuses ``model.GELSTM.train.evaluate`` instead, which returns the same
    keys. ``threshold`` is always supplied by the caller — never derived here — so
    test-set leakage is impossible by construction (see ``.claude/rules/evaluation.md``).
    """
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    both = len(np.unique(targets)) > 1
    auc = float(roc_auc_score(targets, probs)) if both else 0.0
    if both:
        tn, fp, fn, tp = confusion_matrix(targets, pred).ravel()
    else:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": auc,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(targets, pred, zero_division=0)),
    }


class LongitudinalAdapter:
    """Base class: parses the GAAE/training config and declares the hook contract.

    Concrete adapters (``GELSTMAdapter``, ``GECAdapter``) fill in the six hooks plus
    the descriptors. Common fields parsed here: GAAE architecture (from ``gaae_hp``),
    the ``use_fdr`` / ``top_k`` / ``n_folds`` knobs, and the data paths needed by
    ``prepare_data``.
    """

    model_tag: str = "longitudinal"

    def __init__(
        self,
        *,
        gaae_ckpt_path: str,
        gaae_hp: Dict[str, Any],
        train_config: Dict[str, Any],
        data_root: str,
        cohorts_csv: str,
        device: Any,
        rng: Any,
    ) -> None:
        self.gaae_ckpt_path = str(gaae_ckpt_path)
        self.cfg: Dict[str, Any] = dict(train_config or {})
        self.data_root = data_root
        self.cohorts_csv = cohorts_csv
        self.device = device
        self.rng = rng

        gaae_hp = gaae_hp or {}
        self.in_features = 200  # Schaefer-200 ROIs
        self.gaae_hidden = gaae_hp.get("hidden_dim", 128)
        self.gaae_latent = gaae_hp.get("latent_dim", 64)
        self.gaae_heads = gaae_hp.get("num_heads", 2)
        self.gaae_cond_dim = gaae_hp.get("cond_dim", 2)
        self.gaae_dropout = gaae_hp.get("dropout", 0.3)
        self.adjacency_k = gaae_hp.get("adjacency_k", self.cfg.get("adjacency_k", 8))
        self.file_variant = gaae_hp.get("file_variant", self.cfg.get("file_variant", "z_transformed"))

        self.use_fdr = bool(self.cfg.get("use_fdr", False))
        self.top_k = int(self.cfg.get("top_k", self.gaae_latent))
        self.n_folds = int(self.cfg.get("n_folds", 5))

    # ── contract hooks (overridden) ─────────────────────────────────────────
    def build_model(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def prepare_data(self, df):  # pragma: no cover - overridden
        raise NotImplementedError

    def train_fold(self, bundle_tr, bundle_va, cfg, *, rng, device):  # pragma: no cover
        raise NotImplementedError

    def eval_split(self, state, bundle, threshold, *, device):  # pragma: no cover
        raise NotImplementedError

    def truncate_to_n_visits(self, bundle, n):  # pragma: no cover - overridden
        raise NotImplementedError

    def per_visit_probs(self, state, item, *, device):  # pragma: no cover - overridden
        raise NotImplementedError

    # ── descriptors / persistence (overridden) ──────────────────────────────
    def model_config(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def source_files(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def model_state_for_save(self, state) -> Dict[str, Any]:
        """Plain ``nn.Module`` state dict to write as the back-compat ``model_<run>.pth``.

        ``state`` is the composite the adapter's ``train_fold`` returned (it may bundle
        a scaler / dim-filter for the eval hooks). Stripping it back to the raw state
        dict keeps the artifact loadable by the dashboard and comparison notebooks.
        """
        raise NotImplementedError

    def extra_artifacts(self, run_dir, state) -> None:
        """Write any model-specific side artifacts (dim_filter.npy, scaler.pkl, …).

        Default: nothing. Called once after ``save_run`` with the resolved run dir
        and the winning fold's composite ``state``.
        """
        return None

    def load_state(self, run_dir) -> Dict[str, Any]:
        """Rebuild the composite eval ``state`` from a saved run dir (no retraining).

        The inverse of ``model_state_for_save`` + ``extra_artifacts``: reads the
        ``checkpoint_*.pth`` weights plus any side artifacts (``scaler.pkl`` /
        ``dim_filter.npy``) back into the dict the eval hooks (``eval_split``,
        ``per_visit_probs``) expect. Lets analyses re-score a previously trained run
        (e.g. the visit-count confound diagnostics). Read the threshold separately
        with ``read_run_threshold``.
        """
        raise NotImplementedError


__all__ = [
    "get_adapter",
    "binary_metrics",
    "LongitudinalAdapter",
    "load_run_checkpoint",
    "model_state_from_checkpoint",
    "read_run_threshold",
]
