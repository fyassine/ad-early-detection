"""
gelstm.py — GELSTM inference service.

Loads the CLASSIFIER_v2 GELSTM 5-fold ensemble (if checkpoints are
deployed) and provides cohort-wide + per-subject conversion-risk
predictions. The service degrades gracefully when checkpoints are
missing: every method returns a payload with ``available=False`` so the
frontend renders a "model not yet deployed" placeholder instead of
erroring.

Public surface:

    class GELSTMService:
        is_available() -> bool                          fast no-IO check
        load_ensemble() -> bool                         loads the ensemble; cached
        predict_subject(sid, visit_matrices, sex, age)  -> {prob, ci_lo, ci_hi, ...}
        predict_cohort(stats, df)                       -> {sid: prediction}
        model_card_metrics()                            -> {roc, pr, calibration, ...}

Caching strategy:
    - Ensemble + weights loaded once at first use; held in module-level
      globals so repeated requests don't pay reload cost.
    - Per-subject predictions cached to disk at
      ``$DASHBOARD_CACHE_ROOT/gelstm/predictions_<model_version>.pkl``.
      ``model_version`` is the SHA1 of the concatenated first-4kB of each
      checkpoint file — mismatch triggers automatic recompute.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from ..config import CLASSIFIER_ROOT, DASHBOARD_CACHE_ROOT, GELSTM_CHECKPOINT_DIR


# --------------------------------------------------------------------------- #
# Checkpoint discovery                                                        #
# --------------------------------------------------------------------------- #

_FOLD_CHECKPOINT_GLOB = "best_model_fold*.pth"
_GAAE_CHECKPOINT_HINTS = ("gaae_encoder.pth", "gaae.pth", "model_gaae.pth")
_NORM_JSON = "gelstm_norm.json"
_MODEL_CARD_JSON = "model_card.json"


def _discover_fold_checkpoints() -> list[Path]:
    if not GELSTM_CHECKPOINT_DIR.exists():
        return []
    return sorted(GELSTM_CHECKPOINT_DIR.glob(_FOLD_CHECKPOINT_GLOB))


def _discover_gaae_checkpoint() -> Optional[Path]:
    if not GELSTM_CHECKPOINT_DIR.exists():
        return None
    for name in _GAAE_CHECKPOINT_HINTS:
        path = GELSTM_CHECKPOINT_DIR / name
        if path.exists():
            return path
    # Fall back to first .pth that isn't a fold checkpoint.
    for path in GELSTM_CHECKPOINT_DIR.glob("*.pth"):
        if not path.name.startswith("best_model_fold"):
            return path
    return None


def _compute_model_version(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        try:
            with p.open("rb") as f:
                h.update(p.name.encode("utf-8"))
                h.update(b"||")
                h.update(f.read(4096))
                h.update(b"##")
        except OSError:
            continue
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Service                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class _LoadedEnsemble:
    folds: list = field(default_factory=list)      # list[GELSTMClassifier]
    model_version: str = ""
    gaae_path: Optional[Path] = None
    fold_paths: list[Path] = field(default_factory=list)
    norm: dict = field(default_factory=dict)
    device: str = "cpu"


class GELSTMService:
    """
    Singleton-style service. Use ``GELSTMService.instance()`` instead of
    instantiating directly.
    """

    _instance: Optional["GELSTMService"] = None
    _lock = Lock()

    def __init__(self):
        self._ensemble: Optional[_LoadedEnsemble] = None
        self._load_failed = False
        self._load_error: Optional[str] = None

    @classmethod
    def instance(cls) -> "GELSTMService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ─── Availability + loading ──────────────────────────────────────────── #

    def is_available(self) -> bool:
        """Cheap no-IO check: are checkpoints present and loading hasn't failed?"""
        if self._load_failed:
            return False
        if self._ensemble is not None:
            return True
        return bool(_discover_fold_checkpoints())

    def load_ensemble(self) -> bool:
        """
        Load the GELSTM ensemble lazily. Returns True on success, False on
        any failure (checkpoint missing, torch import error, etc.). Failure
        is sticky — subsequent calls short-circuit until the process is
        restarted.
        """
        if self._ensemble is not None:
            return True
        if self._load_failed:
            return False

        fold_paths = _discover_fold_checkpoints()
        if not fold_paths:
            self._load_failed = True
            self._load_error = (
                f"No GELSTM fold checkpoints found at {GELSTM_CHECKPOINT_DIR}. "
                f"Drop best_model_fold[1-5].pth into that directory."
            )
            return False

        gaae_path = _discover_gaae_checkpoint()
        if gaae_path is None:
            self._load_failed = True
            self._load_error = (
                f"No GAAE encoder checkpoint found in {GELSTM_CHECKPOINT_DIR}. "
                f"Expected one of: {', '.join(_GAAE_CHECKPOINT_HINTS)}."
            )
            return False

        # Make CLASSIFIER_v2 importable. The GELSTM models.py already does
        # sys.path.insert internally, but we add the root here so other modules
        # (model.GAAE.models) resolve cleanly even when the dashboard process
        # was started from a different cwd.
        if str(CLASSIFIER_ROOT) not in sys.path:
            sys.path.insert(0, str(CLASSIFIER_ROOT))

        try:
            import torch  # noqa: F401
            from model.GELSTM.models import GELSTMClassifier  # type: ignore
        except Exception as e:
            self._load_failed = True
            self._load_error = f"Failed to import GELSTM model: {e!r}"
            return False

        # Conditioning normalisation constants. If absent we fall back to
        # the identity transform with a warning surfaced via model_card.
        norm = {}
        norm_file = GELSTM_CHECKPOINT_DIR / _NORM_JSON
        if norm_file.exists():
            try:
                norm = json.loads(norm_file.read_text())
            except Exception:
                norm = {}

        # Best-effort default architecture matching CLASSIFIER_v2/train.py.
        # Real deployments should ship hyperparameters in model_card.json
        # alongside the checkpoints; we read them below if present.
        card_file = GELSTM_CHECKPOINT_DIR / _MODEL_CARD_JSON
        arch = {
            "in_features": 200,
            "gaae_hidden": 200,
            "gaae_latent": 64,
            "gaae_heads": 4,
            "gaae_cond_dim": 2,
            "gaae_dropout": 0.2,
            "lstm_hidden": 128,
            "lstm_layers": 2,
            "lstm_dropout": 0.2,
            "use_time_delta": True,
            "classifier_hidden": 64,
        }
        if card_file.exists():
            try:
                card = json.loads(card_file.read_text())
                arch.update(card.get("arch", {}) or {})
                norm = norm or (card.get("norm") or {})
            except Exception:
                pass

        device = "cpu"  # dashboard has no GPU
        try:
            import torch
            folds = []
            for ckpt in fold_paths:
                model = GELSTMClassifier(**arch)
                # GAAE weights must load first; load_gaae_weights freezes
                # the encoder so subsequent state_dict reads don't unfreeze.
                model.load_gaae_weights(str(gaae_path), device=device)
                state = torch.load(str(ckpt), map_location=device, weights_only=False)
                sd = state.state_dict() if hasattr(state, "state_dict") else state
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    print(f"[gelstm] {ckpt.name} missing keys: {missing[:5]}…")
                if unexpected:
                    print(f"[gelstm] {ckpt.name} unexpected keys: {unexpected[:5]}…")
                model.eval()
                folds.append(model)
        except Exception as e:
            self._load_failed = True
            self._load_error = f"Failed to instantiate GELSTM ensemble: {e!r}"
            return False

        self._ensemble = _LoadedEnsemble(
            folds=folds,
            model_version=_compute_model_version([gaae_path] + fold_paths),
            gaae_path=gaae_path,
            fold_paths=fold_paths,
            norm=norm,
            device=device,
        )
        print(f"[gelstm] loaded ensemble ({len(folds)} folds) version={self._ensemble.model_version}")
        return True

    # ─── Inference ──────────────────────────────────────────────────────── #

    def predict_subject(
        self,
        subject_id: str,
        visit_matrices: list[np.ndarray],
        delta_t: list[float],
        sex: Optional[float],
        age: Optional[float],
    ) -> dict:
        """
        Predict conversion probability for a single subject.

        Parameters
        ----------
        subject_id : str
        visit_matrices : list of (n_rois, n_rois) symmetric correlation matrices
            One per visit, ordered chronologically.
        delta_t : list of float
            Normalised inter-visit intervals (months / 108). First entry is
            0.0. Must be the same length as visit_matrices.
        sex : float | None
            0 = F, 1 = M. None if unknown.
        age : float | None
            Subject's age in years at baseline.

        Returns
        -------
        ``{
             available: bool,
             prob: float | None,
             ci_lo: float | None,
             ci_hi: float | None,
             fold_probs: list[float],
             model_version: str,
             note: str | None
        }``
        """
        if not self.load_ensemble() or self._ensemble is None:
            return {
                "available": False,
                "prob": None, "ci_lo": None, "ci_hi": None,
                "fold_probs": [],
                "model_version": "",
                "note": self._load_error or "GELSTM ensemble unavailable",
            }

        # Check prediction disk cache before running inference.
        cached = self.get_cached_prediction(subject_id)
        if cached is not None:
            return cached

        import torch
        from torch_geometric.data import Data, Batch
        from torch_geometric.utils import dense_to_sparse

        if not visit_matrices:
            return {
                "available": True,
                "prob": None, "ci_lo": None, "ci_hi": None,
                "fold_probs": [],
                "model_version": self._ensemble.model_version,
                "note": "no visits",
            }

        cond_vec = self._build_cond_vector(sex, age)

        # Build per-visit graphs (kNN k=8 on absolute correlations, matching
        # CLASSIFIER_v2/model/GELSTM/dataset.py).
        graphs = [self._matrix_to_graph(m) for m in visit_matrices]

        fold_probs: list[float] = []
        with torch.inference_mode():
            for model in self._ensemble.folds:
                try:
                    z_seq = []
                    for g in graphs:
                        batch = Batch.from_data_list([g])
                        z_nodes = model.encoder.encode(
                            batch.x, batch.edge_index,
                            edge_attr=getattr(batch, "edge_attr", None),
                        )
                        # FiLM-condition the latents with sex/age (matches GAAE forward()).
                        batch_mask = batch.batch if hasattr(batch, "batch") else torch.zeros(
                            z_nodes.shape[0], dtype=torch.long, device=z_nodes.device,
                        )
                        z_nodes = model.encoder.condition_latent(z_nodes, cond_vec, batch_mask)
                        z_pooled = z_nodes.mean(dim=0, keepdim=True)
                        z_seq.append(z_pooled)
                    z = torch.cat(z_seq, dim=0).unsqueeze(0)  # (1, T, d)
                    if model.use_time_delta:
                        dt = torch.tensor(delta_t, dtype=z.dtype).view(1, -1, 1)
                        z = torch.cat([z, dt], dim=-1)
                    lengths = torch.tensor([z.shape[1]], dtype=torch.long)
                    packed = torch.nn.utils.rnn.pack_padded_sequence(
                        z, lengths, batch_first=True, enforce_sorted=False,
                    )
                    logits = model(packed)
                    fold_probs.append(float(torch.sigmoid(logits.view(-1)[0]).item()))
                except Exception as e:
                    print(f"[gelstm] fold inference failed for {subject_id}: {e!r}")
                    continue

        if not fold_probs:
            return {
                "available": True,
                "prob": None, "ci_lo": None, "ci_hi": None,
                "fold_probs": [],
                "model_version": self._ensemble.model_version,
                "note": "all folds failed",
            }

        probs = np.asarray(fold_probs, dtype=np.float64)
        result = {
            "available": True,
            "prob": float(probs.mean()),
            "ci_lo": float(np.quantile(probs, 0.025)) if probs.size > 1 else float(probs.min()),
            "ci_hi": float(np.quantile(probs, 0.975)) if probs.size > 1 else float(probs.max()),
            "fold_probs": fold_probs,
            "model_version": self._ensemble.model_version,
            "note": None,
        }
        # Persist prediction to disk so subsequent requests + server restarts
        # don't need to run inference again.
        try:
            cache = self._load_predictions_cache()
            cache[subject_id] = result
            self._save_predictions_cache(cache)
        except Exception:
            pass
        return result

    # ─── Helpers ────────────────────────────────────────────────────────── #

    def _build_cond_vector(self, sex, age):
        import torch
        norm = self._ensemble.norm if self._ensemble else {}
        sex_v = float(sex) if sex is not None else float(norm.get("sex_default", 0.5))
        if age is None:
            age_v = 0.0
        else:
            mu = float(norm.get("age_mean", 65.0))
            sd = float(norm.get("age_std", 10.0))
            age_v = (float(age) - mu) / max(sd, 1e-6)
        return torch.tensor([sex_v, age_v], dtype=torch.float32).view(1, -1)

    def _matrix_to_graph(self, corr: np.ndarray, k: int = 8):
        """Replicates LongitudinalSubjectDataset._load_graph: kNN(k=8) on |corr|."""
        import torch
        from torch_geometric.data import Data
        from torch_geometric.utils import dense_to_sparse

        arr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        feat = torch.tensor(arr, dtype=torch.float)
        abs_feat = torch.abs(feat)
        n = abs_feat.shape[0]
        eye = torch.eye(n, dtype=torch.bool)
        abs_feat = abs_feat.masked_fill(eye, -1.0)
        k_eff = min(k, n - 1)
        idx = abs_feat.topk(k_eff, dim=-1).indices  # (n, k)
        adj = torch.zeros((n, n), dtype=torch.float32)
        adj.scatter_(1, idx, 1.0)
        adj = ((adj + adj.t()) > 0).float()
        ei, ew = dense_to_sparse(adj)
        return Data(x=feat, edge_index=ei, edge_attr=ew)

    # ─── Prediction disk cache ──────────────────────────────────────────── #

    def _predictions_cache_path(self) -> Optional[Path]:
        if self._ensemble is None:
            return None
        return DASHBOARD_CACHE_ROOT / "gelstm" / f"predictions_{self._ensemble.model_version}.pkl"

    def _load_predictions_cache(self) -> dict:
        path = self._predictions_cache_path()
        if path is None or not path.exists():
            return {}
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            return {}

    def _save_predictions_cache(self, predictions: dict) -> None:
        path = self._predictions_cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(predictions))
            tmp.replace(path)
        except Exception:
            pass

    def get_cached_prediction(self, subject_id: str) -> Optional[dict]:
        """Return the cached prediction for this subject, or None if not cached."""
        if self._ensemble is None:
            return None
        cache = self._load_predictions_cache()
        entry = cache.get(subject_id)
        if entry and entry.get("model_version") == self._ensemble.model_version:
            return entry
        return None

    # ─── Model card ─────────────────────────────────────────────────────── #

    def model_card_metrics(self) -> dict:
        """
        Surface the GELSTM model's published evaluation metrics. We do NOT
        rerun training-time evaluation here — the card is read from
        ``<checkpoint_dir>/model_card.json`` written by the training
        pipeline. Missing fields are returned as None so the frontend can
        render a partial card.
        """
        if not self.load_ensemble() or self._ensemble is None:
            return {
                "available": False,
                "note": self._load_error or "GELSTM ensemble unavailable",
            }
        card_file = GELSTM_CHECKPOINT_DIR / _MODEL_CARD_JSON
        if not card_file.exists():
            return {
                "available": True,
                "model_version": self._ensemble.model_version,
                "n_folds": len(self._ensemble.folds),
                "note": (
                    f"Ensemble loaded ({len(self._ensemble.folds)} folds) but no "
                    f"model_card.json — drop a JSON file alongside checkpoints "
                    f"with roc/pr/calibration/cm payloads."
                ),
                "roc": None, "pr": None, "calibration": None, "cm": None,
            }
        try:
            card = json.loads(card_file.read_text())
        except Exception as e:
            return {
                "available": True,
                "model_version": self._ensemble.model_version,
                "note": f"model_card.json present but unreadable: {e!r}",
            }
        return {
            "available": True,
            "model_version": self._ensemble.model_version,
            "n_folds": len(self._ensemble.folds),
            **card,
        }


# Convenience function used by routes to avoid leaking the singleton pattern.
def get_gelstm_service() -> GELSTMService:
    return GELSTMService.instance()
