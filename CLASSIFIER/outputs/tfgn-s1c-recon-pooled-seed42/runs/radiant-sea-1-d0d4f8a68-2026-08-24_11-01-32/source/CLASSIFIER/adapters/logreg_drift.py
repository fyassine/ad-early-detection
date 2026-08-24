"""adapters/logreg_drift.py — LogRegDriftAdapter: logistic regression on FC-change features.

Stage 0 baseline for the temporal-first ablation ladder: PCA-compressed
vectorised FC change (ΔA), plus visit count, total follow-up, age and sex.
Routes through ``model/classification/logreg_cv.py::train_logreg_cv`` so it
shares the identical CV, threshold and external-test path as every other arm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from common.crossval import Bundle
from model.classification.logreg_cv import train_logreg_cv
from sklearn.decomposition import PCA

from . import LongitudinalAdapter, binary_metrics, load_run_checkpoint


class LogRegDriftAdapter(LongitudinalAdapter):
    """Logistic regression on PCA-compressed FC change features + metadata."""

    model_tag = "logregdrift"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_visits = self.cfg.get("min_visits")
        self.max_visits = self.cfg.get("max_visits")

    def build_model(self) -> None:
        """No torch model: train_fold fits sklearn PCA + LogisticRegression directly.

        Returns None — the shared notebook cell only calls this as an
        architecture smoke test (`_ = build_model()`) and never uses the
        return value for this adapter.
        """
        return None

    def _extract_features(self, items: List[Dict[str, Any]], pca: Optional[PCA] = None):
        """Extract [PCA_32(vec(ΔA)), n_visits, total_months, age, sex] for each item."""
        if not items:
            raise ValueError("Empty items list provided to _extract_features")

        vec_diffs = []
        metadata = []

        for item in items:
            A_first = item["graphs"][0].x.cpu().numpy()
            A_last = item["graphs"][-1].x.cpu().numpy()
            delta_A = A_last - A_first

            # Upper triangle flattened (k=1 excludes the diagonal)
            vec_delta = delta_A[np.triu_indices_from(delta_A, k=1)]
            vec_diffs.append(vec_delta)

            n_visits = float(item["n_scans"])
            total_months = float(np.sum(item["delta_t"]) * 108.0)
            age = float(item["age"])
            sex = float(item["sex"])
            metadata.append([n_visits, total_months, age, sex])

        vec_diffs_arr = np.stack(vec_diffs)
        metadata_arr = np.stack(metadata)

        if pca is None:
            n_samples, n_features = vec_diffs_arr.shape
            n_components = min(n_samples, n_features, 32)
            pca = PCA(n_components=n_components, random_state=42)
            pca_feats = pca.fit_transform(vec_diffs_arr)
        else:
            pca_feats = pca.transform(vec_diffs_arr)

        X = np.concatenate([pca_feats, metadata_arr], axis=1)
        return X, pca

    def prepare_data(self, df) -> Bundle:
        # Route through multicohort bundle if 'cohort' column is present
        if "cohort" in getattr(df, "columns", []):
            from common.pooled_data import build_multicohort_bundle

            return build_multicohort_bundle(
                df,
                adjacency_k=self.adjacency_k,
                file_variant=self.file_variant,
                min_visits=self.min_visits,
                max_visits=self.max_visits,
            )

        from model.GELSTM.dataset import LongitudinalSubjectDataset

        ds = LongitudinalSubjectDataset(
            self.data_root,
            df,
            self.cohorts_csv,
            adjacency_k=self.adjacency_k,
            file_variant=self.file_variant,
            min_visits=self.min_visits,
            max_visits=self.max_visits,
            cohort=self.cfg.get("cohort", "delcode"),
        )
        items = [ds[i] for i in range(len(ds))]
        return Bundle(ds.get_labels(), ds.get_subject_ids(), items)

    def train_fold(
        self,
        bundle_tr: Bundle,
        bundle_va: Bundle,
        cfg: Dict[str, Any],
        *,
        rng: Any,
        device: Any,
        epoch_log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        X_tr, pca = self._extract_features(bundle_tr.items, pca=None)
        X_va, _ = self._extract_features(bundle_va.items, pca=pca)

        y_tr = np.array(bundle_tr.labels, dtype=int)
        y_va = np.array(bundle_va.labels, dtype=int)
        groups_tr = np.array(bundle_tr.groups)

        seed = 42
        if rng is not None:
            seed = int(rng.integers(0, 1_000_000))

        # Train with internal CV to find the best configuration and scaler
        cv_result = train_logreg_cv(
            X=X_tr,
            y=y_tr,
            groups=groups_tr,
            n_folds=self.n_folds,
            seed=seed,
            lr_max_iter=2000,
        )

        scaler = cv_result.best_scaler
        clf = cv_result.best_clf

        if scaler is None or clf is None:
            raise ValueError("train_logreg_cv failed to return a valid scaler or classifier")

        X_va_scaled = scaler.transform(X_va)
        va_probs = clf.predict_proba(X_va_scaled)[:, 1]

        best_threshold = float(cv_result.youden_threshold)
        vm = binary_metrics(y_va, va_probs, best_threshold)

        state_dict = {
            "pca": pca,
            "scaler": scaler,
            "clf": clf,
            "threshold": best_threshold,
        }

        return {
            "state_dict": state_dict,
            "val_metrics": vm,
            "best_threshold": best_threshold,
            "oof_probs": va_probs,
            "oof_targets": y_va,
            "oof_sids": list(bundle_va.groups),
        }

    def eval_split(self, state: Dict[str, Any], bundle: Bundle, threshold: float, *, device: Any) -> Dict[str, Any]:
        pca = state["pca"]
        scaler = state["scaler"]
        clf = state["clf"]

        X, _ = self._extract_features(bundle.items, pca=pca)
        X_scaled = scaler.transform(X)
        probs = clf.predict_proba(X_scaled)[:, 1]

        res = binary_metrics(bundle.labels, probs, threshold)
        res["probs"] = probs
        res["targets"] = np.array(bundle.labels, dtype=int)
        res["subject_ids"] = np.array(bundle.groups)
        return res

    def truncate_to_n_visits(self, bundle: Bundle, n: int) -> Bundle:
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
        return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)

    def per_visit_probs(self, state: Dict[str, Any], item: Dict[str, Any], *, device: Any) -> List[tuple]:
        pca = state["pca"]
        scaler = state["scaler"]
        clf = state["clf"]

        out = []
        for t in range(1, item["n_scans"] + 1):
            sub = {
                **item,
                "graphs": item["graphs"][:t],
                "delta_t": item["delta_t"][:t],
                "visit_months": item["visit_months"][:t],
                "n_scans": t,
            }
            X, _ = self._extract_features([sub], pca=pca)
            X_scaled = scaler.transform(X)
            prob = float(clf.predict_proba(X_scaled)[0, 1])
            out.append((item["visit_months"][t - 1], prob))

        return out

    def model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "LogRegDriftAdapter",
            "pca_components": 32,
            "n_folds": self.n_folds,
            "min_visits": self.min_visits,
            "max_visits": self.max_visits,
        }

    def source_files(self) -> List[Path]:
        classifier_root = Path(__file__).resolve().parents[1]
        return [
            classifier_root / "model" / "classification" / "logreg_cv.py",
            classifier_root / "adapters" / "logreg_drift.py",
        ]

    def model_state_for_save(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def extra_artifacts(self, run_dir: str, state: Dict[str, Any]) -> None:
        pass

    def load_state(self, run_dir: str) -> Dict[str, Any]:
        return load_run_checkpoint(run_dir, device="cpu")
