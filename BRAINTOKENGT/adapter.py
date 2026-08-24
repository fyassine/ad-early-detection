"""
adapter.py — BrainTokenGTAdapter.

Implements the same six-hook contract as ``CLASSIFIER/adapters/*`` so the
competitor baseline runs through the identical protocol as GELSTM:

    build_model, prepare_data, train_fold, eval_split,
    truncate_to_n_visits, per_visit_probs

Why reuse the CLASSIFIER machinery instead of porting upstream's ``main_optuna.py``
==================================================================================
Upstream has no usable harness to port. ``main_optuna.py:60`` reads

    val_index = list(range(9)); train_index = [9]

— it trains on ONE subject and validates on nine, with ``StratifiedKFold``
commented out on the line above and no test split anywhere. The value it returns
to Optuna is ``best_auc``, a max-over-epochs on that same validation set.

So the harness has to be written from scratch regardless, and writing it on top of
``CLASSIFIER.common`` is what makes the comparison fair *by construction*: the CV
splitter, the threshold policy, the metric definitions, the artifact schema and
the subject/visit selection are then literally the same code objects GELSTM runs
through, not a re-implementation that might differ subtly.

In particular ``prepare_data`` builds its Bundle from the very same
``LongitudinalSubjectDataset``, so both models see identical FC matrices for an
identical subject set.

Cohort window
-------------
``min_visits`` / ``max_visits`` (default 2 / 3) restrict subjects to the regime
the authors evaluated ("all subjects have 2-3 time points"). DELCODE spans 1-6
visits, and 47 of 167 subjects have a single scan — a T=1 sequence has no temporal
edges at all, a path upstream never executes. Run GELSTM with the same window when
reporting the head-to-head; see README.md §"Fair-comparison contract".
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_CLASSIFIER_ROOT = _REPO_ROOT / "CLASSIFIER"
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from CLASSIFIER.adapters import (  # noqa: E402
    LongitudinalAdapter,
    binary_metrics,
    load_run_checkpoint,
    model_state_from_checkpoint,
)
from CLASSIFIER.common.crossval import Bundle  # noqa: E402
from CLASSIFIER.common.thresholds import best_f1_threshold  # noqa: E402
from CLASSIFIER.model.GELSTM.dataset import LongitudinalSubjectDataset  # noqa: E402
from CLASSIFIER.model.GELSTM.utils import compute_class_weights  # noqa: E402

from .model import UPSTREAM_EDGE_DENSITY, BrainTokenGT, item_to_sequence, window_item  # noqa: E402


class BrainTokenGTAdapter(LongitudinalAdapter):
    """Brain-TokenGT (Dong et al., MICCAI 2023) as a longitudinal-contract adapter.

    Unlike the GELSTM/GEC/GEP adapters this model is trained END-TO-END: it has no
    pretrained GAAE encoder. ``gaae_ckpt_path`` is accepted (the base constructor
    and the shared runner both pass it) but never loaded; it is recorded in the run
    provenance as ``none (end-to-end)`` so the artifact schema stays uniform.
    """

    model_tag = "braintokengt"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        c = self.cfg

        # ── architecture ────────────────────────────────────────────────────
        self.in_features = int(c.get("in_features", 200))  # Schaefer-200 ROIs
        self.num_nodes = int(c.get("num_nodes", self.in_features))
        self.output_sizes = list(c.get("output_sizes", [32, 32]))
        self.nhead = int(c.get("nhead", 2))
        self.num_layers = int(c.get("num_layers", 2))
        self.static_edge_topk = int(c.get("static_edge_topk", 180))

        # ── upstream-fidelity flags (defaults reproduce the released code) ──
        self.edge_weight_mode = c.get("edge_weight_mode", "binary")
        self.readout = c.get("readout", "mean")
        self.force_single_head = bool(c.get("force_single_head", True))
        self.train_give = bool(c.get("train_give", False))
        # Stability knobs for the newly-unfrozen GIVE/GRCU parameters when
        # train_give=True (see "fix-stabilized" experiment, README.md). Default to
        # the main optimizer settings so train_give=False (or an unset config) is
        # byte-for-byte the pre-existing single-param-group behaviour.
        self.give_weight_decay = c.get("give_weight_decay")
        self.give_lr_scale = float(c.get("give_lr_scale", 1.0))

        # ── data / cohort ───────────────────────────────────────────────────
        self.edge_density = float(c.get("edge_density", UPSTREAM_EDGE_DENSITY))
        self.adjacency_metric = c.get("adjacency_metric", "raw")
        self.min_visits = c.get("min_visits", 2)
        self.max_visits = c.get("max_visits", 3)
        self.cohort = str(c.get("cohort", "delcode")).lower()

        # ── optimisation ────────────────────────────────────────────────────
        self.optimizer_name = c.get("optimizer", "Adam")
        self.learning_rate = float(c.get("learning_rate", 1e-3))
        self.weight_decay = float(c.get("weight_decay", 0.0))
        self.epochs = int(c.get("epochs", 50))
        self.early_stopping_patience = int(c.get("early_stopping_patience", 15))
        self.batch_size = int(c.get("batch_size", 8))
        self.use_class_cost_weights = bool(c.get("use_class_cost_weights", True))
        self.grad_clip = c.get("grad_clip", 1.0)
        self.max_subjects = c.get("max_subjects")  # smoke-run cap; None = full cohort

        self._cached_state_id: Optional[int] = None
        self._cached_model: Optional[nn.Module] = None

    # ── arch ────────────────────────────────────────────────────────────────
    def _build_model(self) -> BrainTokenGT:
        return BrainTokenGT(
            in_channels=self.in_features,
            output_sizes=self.output_sizes,
            num_nodes=self.num_nodes,
            nhead=self.nhead,
            num_layers=self.num_layers,
            static_edge_topk=self.static_edge_topk,
            edge_weight_mode=self.edge_weight_mode,
            readout=self.readout,
            force_single_head=self.force_single_head,
            train_give=self.train_give,
        ).to(self.device)

    def build_model(self) -> BrainTokenGT:
        m = self._build_model()
        trainable = sum(p.numel() for p in m.get_trainable_params())
        total = sum(p.numel() for p in m.parameters())
        print(
            f"Model built [Brain-TokenGT d_model={m.d_model} heads={m.nhead} "
            f"L{self.num_layers}]: trainable={trainable:,}  total={total:,}"
        )
        print(
            f"  fidelity: edge_weight_mode={self.edge_weight_mode}  readout={self.readout}  "
            f"force_single_head={self.force_single_head}  train_give={self.train_give}"
        )
        return m

    # ── data ────────────────────────────────────────────────────────────────
    def prepare_data(self, df) -> Bundle:
        """Build the Bundle from the SAME dataset object the GELSTM adapter uses."""
        if "cohort" in getattr(df, "columns", []):
            # Pooled ADNI+DELCODE frame (temporal-first ladder S0d) — same dispatch
            # as GELSTMAdapter / TFGNAdapter. min_visits/max_visits are applied
            # below via the same post-hoc filter+window_item path as the
            # single-cohort branch, not passed into build_multicohort_bundle, so
            # the two branches windowed identically either way.
            from CLASSIFIER.common.pooled_data import build_multicohort_bundle

            bundle = build_multicohort_bundle(
                df,
                adjacency_k=self.adjacency_k,
                file_variant=self.file_variant,
            )
            items = bundle.items
        else:
            ds = LongitudinalSubjectDataset(
                self.data_root,
                df,
                self.cohorts_csv,
                adjacency_k=self.adjacency_k,
                file_variant=self.file_variant,
                cohort=self.cohort,
            )
            items = [ds[i] for i in range(len(ds))]

        kept = [it for it in items if it["n_scans"] >= int(self.min_visits or 1)]
        n_dropped = len(items) - len(kept)
        kept = [window_item(it, max_visits=self.max_visits) for it in kept]

        if self.max_subjects:
            # Smoke runs only: deterministic, class-stratified subsample.
            kept = _stratified_head(kept, int(self.max_subjects))

        n_pos = sum(int(it["label"]) for it in kept)
        print(
            f"Brain-TokenGT cohort: {len(kept)} subjects "
            f"({n_pos} converter, {len(kept) - n_pos} stable MCI); "
            f"window=[min_visits={self.min_visits}, max_visits={self.max_visits}]; "
            f"dropped (too few visits)={n_dropped}"
        )
        if kept:
            ns = [it["n_scans"] for it in kept]
            print(f"  Visits per subject: min={min(ns)}  max={max(ns)}  mean={np.mean(ns):.2f}")

        return Bundle([int(it["label"]) for it in kept], [it["subject_id"] for it in kept], kept)

    def _sequence(self, item):
        return item_to_sequence(
            item,
            edge_density=self.edge_density,
            metric=self.adjacency_metric,
            device=self.device,
        )

    # ── training ────────────────────────────────────────────────────────────
    def _forward_logit(self, model, item) -> torch.Tensor:
        A_list, Nodes_list = self._sequence(item)
        return model(A_list, Nodes_list, None)

    def _predict(self, model, items) -> np.ndarray:
        model.eval()
        probs: List[float] = []
        with torch.no_grad():
            for item in items:
                logit = self._forward_logit(model, item)
                probs.append(float(torch.sigmoid(logit).item()))
        return np.nan_to_num(np.asarray(probs, dtype=float), nan=0.5)

    def _evaluate(self, model, items, threshold: Optional[float]) -> Dict[str, Any]:
        probs = self._predict(model, items)
        targets = np.asarray([int(it["label"]) for it in items], dtype=int)
        if threshold is None:
            # Validation-side only. Never called with a test split — the notebook
            # passes the OOF-derived threshold explicitly (rules/evaluation.md).
            threshold = best_f1_threshold(targets, probs) if len(np.unique(targets)) > 1 else 0.5
        metrics = binary_metrics(targets, probs, threshold)
        return {
            **metrics,
            "best_threshold": float(threshold),
            "probs": probs,
            "targets": targets,
            "subject_ids": [it["subject_id"] for it in items],
        }

    def _build_optimizer(self, model: BrainTokenGT, params: List[torch.nn.Parameter]):
        """Adam over ``params``, splitting off a GIVE/GRCU param group when
        ``train_give=True`` and ``give_weight_decay`` is explicitly configured
        (stabilization for the previously-frozen GIVE parameters — see
        BRAINTOKENGT/README.md "fix-stabilized" experiment). Unset -> a single
        param group identical to the pre-existing behaviour.
        """
        if not self.train_give or self.give_weight_decay is None:
            return getattr(torch.optim, self.optimizer_name)(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )

        give_ids = {id(p) for p in model.GRCU_layers.parameters()}
        give_params = [p for p in params if id(p) in give_ids]
        other_params = [p for p in params if id(p) not in give_ids]
        return getattr(torch.optim, self.optimizer_name)(
            [
                {"params": other_params, "weight_decay": self.weight_decay},
                {
                    "params": give_params,
                    "weight_decay": float(self.give_weight_decay),
                    "lr": self.learning_rate * self.give_lr_scale,
                },
            ],
            lr=self.learning_rate,
        )

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
        tr_items, va_items = list(bundle_tr.items), list(bundle_va.items)
        model = self._build_model()

        if self.use_class_cost_weights:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=compute_class_weights(bundle_tr.labels, device=device)
            )
        else:
            criterion = nn.BCEWithLogitsLoss()

        params = model.get_trainable_params()
        optimizer = self._build_optimizer(model, params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_auc, best_state, no_improve = 0.0, copy.deepcopy(model.state_dict()), 0
        order = np.arange(len(tr_items))

        for epoch in range(self.epochs):
            model.train()
            # Subject-level shuffling through the injected Generator only
            # (.claude/rules/seeding.md) — never global numpy state.
            if rng is not None:
                rng.shuffle(order)

            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss_sum = None
                for j in chunk:
                    item = tr_items[int(j)]
                    logit = self._forward_logit(model, item)
                    target = torch.tensor(
                        [float(item["label"])], dtype=logit.dtype, device=logit.device
                    )
                    loss = criterion(logit, target)
                    loss_sum = loss if loss_sum is None else loss_sum + loss
                if loss_sum is None:
                    continue
                batch_loss = loss_sum / len(chunk)
                batch_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, float(self.grad_clip))
                optimizer.step()
                total_loss += float(batch_loss.item())
                n_batches += 1

            va = self._evaluate(model, va_items, None)
            scheduler.step(va["auc"])
            if epoch_log_fn is not None:
                epoch_log_fn(
                    {
                        "epoch": epoch,
                        "train_loss": total_loss / max(n_batches, 1),
                        "val_auc": va["auc"],
                        "val_f1": va["f1"],
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
        final_va = self._evaluate(model, va_items, None)
        state = {"model_state": best_state}
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

    def eval_split(self, state, bundle, threshold, *, device) -> Dict[str, Any]:
        model = self._model_for_state(state)
        return self._evaluate(model, list(bundle.items), threshold)

    def truncate_to_n_visits(self, bundle, n) -> Bundle:
        items = [window_item(it, max_visits=n) for it in bundle.items if it["n_scans"] >= n]
        return Bundle([int(it["label"]) for it in items], [it["subject_id"] for it in items], items)

    def per_visit_probs(self, state, item, *, device):
        model = self._model_for_state(state)
        out = []
        with torch.no_grad():
            for t in range(1, item["n_scans"] + 1):
                sub = window_item(item, max_visits=t)
                prob = float(torch.sigmoid(self._forward_logit(model, sub)).item())
                out.append((item["visit_months"][t - 1], prob))
        return out

    # ── descriptors / persistence ───────────────────────────────────────────
    def model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "BrainTokenGT",
            "paper": "Dong et al., MICCAI 2023 (arXiv:2307.00858)",
            "upstream_ref": "Brain-TokenGT/ (pristine checkout)",
            "pretrained_encoder": None,
            "in_features": self.in_features,
            "num_nodes": self.num_nodes,
            "output_sizes": self.output_sizes,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "static_edge_topk": self.static_edge_topk,
            "edge_density": self.edge_density,
            "adjacency_metric": self.adjacency_metric,
            "min_visits": self.min_visits,
            "max_visits": self.max_visits,
            # Fidelity flags — the row label in the results table.
            "edge_weight_mode": self.edge_weight_mode,
            "readout": self.readout,
            "force_single_head": self.force_single_head,
            "train_give": self.train_give,
            "give_weight_decay": self.give_weight_decay,
            "give_lr_scale": self.give_lr_scale,
            "upstream_faithful": self.is_upstream_faithful(),
        }

    def is_upstream_faithful(self) -> bool:
        """True when every fidelity flag reproduces the released code."""
        return (
            self.edge_weight_mode == "binary"
            and self.readout == "mean"
            and self.force_single_head
            and not self.train_give
        )

    def source_files(self):
        return [
            _PACKAGE_ROOT / "model" / "transformer.py",
            _PACKAGE_ROOT / "model" / "grcu.py",
            _PACKAGE_ROOT / "model" / "sequences.py",
            _PACKAGE_ROOT / "adapter.py",
            _REPO_ROOT / "Brain-TokenGT" / "model_transformer.py",
            _REPO_ROOT / "Brain-TokenGT" / "model_grcu.py",
        ]

    def model_state_for_save(self, state) -> Dict[str, Any]:
        return state["model_state"]

    def load_state(self, run_dir) -> Dict[str, Any]:
        ckpt = load_run_checkpoint(Path(run_dir), device=self.device)
        return {"model_state": model_state_from_checkpoint(ckpt)}


def _stratified_head(items: List[Dict], n: int) -> List[Dict]:
    """First ``n`` subjects, class-balanced and deterministic (smoke runs only).

    Interleaves converters and stable MCI so a tiny subsample still contains both
    classes — otherwise AUC is undefined and the smoke run tells you nothing.
    """
    if n >= len(items):
        return items
    pos = [it for it in items if int(it["label"]) == 1]
    neg = [it for it in items if int(it["label"]) == 0]
    out: List[Dict] = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos):
            out.append(pos[i])
        if i < len(neg):
            out.append(neg[i])
        if len(out) >= n:
            break
    return out[:n]
