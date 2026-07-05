"""
common/visit_confound.py — visit-count confound diagnostics over the test bundle.

Converters are labelled by *ever* having a "converter" visit, and they tend to drop
out of follow-up sooner, so they carry fewer visits on average. The shared
``early_detection_table`` (see ``common/early_detection.py``) re-scores a *different,
shrinking* cohort at each N (only subjects with ``>= N`` visits), so its AUC-vs-N
curve conflates three things: real information per visit, a shifting class balance,
and small-sample noise. These routines quantify that confound. They are
model-agnostic — like ``early_detection.py`` they take the trained model's
``state`` plus the adapter hooks (``truncate_to_n_visits``, ``eval_split``,
``per_visit_probs``) and never retrain or derive a new threshold.

All functions are deterministic (no sampling), so none takes an ``rng``.

Label convention (see ``DATA/src/splitting/create_downstream_data_splits.py`` and
``model/GELSTM/dataset.py``): ``1 = converter``, ``0 = non-converter`` (stable MCI).
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from .crossval import Bundle
from .early_detection import early_detection_table

CONVERTER = 1
NON_CONVERTER = 0


def _label_name(label: int) -> str:
    return "converter" if int(label) == CONVERTER else "non_converter"


def visit_counts_by_label(bundle) -> pd.DataFrame:
    """One row per subject: ``subject_id``, ``label``, ``group``, ``n_scans``."""
    return pd.DataFrame(
        [
            {
                "subject_id": item["subject_id"],
                "label": int(item["label"]),
                "group": _label_name(item["label"]),
                "n_scans": int(item["n_scans"]),
            }
            for item in bundle.items
        ],
        columns=["subject_id", "label", "group", "n_scans"],
    )


def summarize_visit_counts(bundle) -> pd.DataFrame:
    """Per-group visit-count summary + a converter-vs-non-converter Mann-Whitney U.

    Returns one row per group (``converter`` / ``non_converter``) plus an ``overall``
    row, with ``n``, ``mean``, ``median``, ``std``, ``min``, ``max`` of ``n_scans``.
    A two-sided Mann-Whitney U p-value (converter vs non-converter visit counts) is
    attached to every row in the ``mwu_pvalue`` column; it is ``NaN`` when either
    group is empty (the test is undefined). This directly answers "do converters
    have fewer visits?".
    """
    df = visit_counts_by_label(bundle)

    def _row(name: str, ns: np.ndarray) -> dict:
        return {
            "group": name,
            "n": int(ns.size),
            "mean": float(np.mean(ns)) if ns.size else float("nan"),
            "median": float(np.median(ns)) if ns.size else float("nan"),
            "std": float(np.std(ns, ddof=1)) if ns.size > 1 else 0.0,
            "min": int(np.min(ns)) if ns.size else 0,
            "max": int(np.max(ns)) if ns.size else 0,
        }

    conv = df.loc[df["label"] == CONVERTER, "n_scans"].to_numpy()
    nonconv = df.loc[df["label"] == NON_CONVERTER, "n_scans"].to_numpy()
    rows = [
        _row("converter", conv),
        _row("non_converter", nonconv),
        _row("overall", df["n_scans"].to_numpy()),
    ]

    pvalue = float("nan")
    if conv.size and nonconv.size:
        pvalue = float(mannwhitneyu(conv, nonconv, alternative="two-sided").pvalue)
    out = pd.DataFrame(rows)
    out["mwu_pvalue"] = pvalue
    return out


def cohort_composition_table(
    test_bundle,
    truncate_to_n_visits: Callable[[Any, int], Any],
    *,
    max_visits: int | None = None,
) -> List[dict]:
    """Per-N class balance of the ``early_detection_table`` evaluation cohort.

    For ``N = 1 .. max_visits`` restrict to subjects with ``>= N`` visits (via the
    same ``truncate_to_n_visits`` hook the AUC table uses) and count converters vs
    non-converters. Rows: ``n_visits, n_subjects, n_converters, n_nonconverters,
    frac_converter``. This is the companion that explains why the AUC table's
    per-N rows are not comparable (the cohort and its class balance change with N).
    Needs the data only — no trained model.
    """
    if not test_bundle.items:
        return []
    if max_visits is None:
        max_visits = max(int(item["n_scans"]) for item in test_bundle.items)

    rows: List[dict] = []
    for n_vis in range(1, max_visits + 1):
        sub = truncate_to_n_visits(test_bundle, n_vis)
        labels = np.asarray(sub.labels, dtype=int)
        n_sub = len(sub.items)
        if n_sub == 0:
            continue
        n_conv = int((labels == CONVERTER).sum())
        rows.append(
            {
                "n_visits": n_vis,
                "n_subjects": n_sub,
                "n_converters": n_conv,
                "n_nonconverters": n_sub - n_conv,
                "frac_converter": n_conv / n_sub,
            }
        )
    return rows


def fixed_cohort_bundle(test_bundle, *, min_n_scans: int | None = None) -> Bundle:
    """Subset to the deepest-followed subjects (``n_scans >= min_n_scans``).

    Defaults ``min_n_scans`` to the maximum visit count in the bundle, i.e. only
    subjects observed at every step of the early-detection curve. Used to hold the
    cohort *fixed* so an AUC-vs-N trend reflects information per visit, not a
    changing population.
    """
    if not test_bundle.items:
        return Bundle([], [], [])
    if min_n_scans is None:
        min_n_scans = max(int(item["n_scans"]) for item in test_bundle.items)
    items = [it for it in test_bundle.items if int(it["n_scans"]) >= min_n_scans]
    return Bundle(
        [it["label"] for it in items],
        [it["subject_id"] for it in items],
        items,
    )


def early_detection_fixed_cohort(
    test_bundle,
    eval_split: Callable[..., dict],
    truncate_to_n_visits: Callable[[Any, int], Any],
    state_dict: Any,
    threshold: float,
    *,
    device: Any,
    min_n_scans: int | None = None,
    min_subjects: int = 4,
) -> List[dict]:
    """``early_detection_table`` on a cohort held fixed across N.

    Restrict to subjects with ``n_scans >= min_n_scans``, then reuse
    ``early_detection_table``. Because the cohort no longer changes between rows, the
    AUC trend isolates added-visit information from the cohort/class-balance shift of
    the standard variable-cohort table. Note the fixed cohort can be small —
    interpret with its ``n_subjects``.

    When ``min_n_scans`` is None it defaults to the **deepest viable depth**: the
    largest N whose ``>= N`` cohort still has at least ``min_subjects`` subjects and
    both classes. Anchoring to the raw maximum visit count instead would often select
    a tiny, single-class tail (converters tend to be followed longest) and yield an
    empty table.
    """
    if min_n_scans is None:
        comp = cohort_composition_table(test_bundle, truncate_to_n_visits)
        viable = [
            r["n_visits"]
            for r in comp
            if r["n_subjects"] >= min_subjects
            and r["n_converters"] > 0
            and r["n_nonconverters"] > 0
        ]
        if not viable:
            return []
        min_n_scans = max(viable)
    fixed = fixed_cohort_bundle(test_bundle, min_n_scans=min_n_scans)
    return early_detection_table(
        fixed,
        eval_split,
        truncate_to_n_visits,
        state_dict,
        threshold,
        device=device,
        min_subjects=min_subjects,
    )


def prob_vs_visit_count(
    test_bundle,
    per_visit_probs: Callable[..., list],
    state_dict: Any,
    *,
    device: Any,
) -> Tuple[pd.DataFrame, dict]:
    """Full-trajectory P(converter) vs number of visits, with Spearman correlations.

    For each subject takes the *final* prediction from the ``per_visit_probs`` hook
    (the model's P(converter) given the full trajectory) and pairs it with
    ``n_scans`` and ``label``. Returns the tidy frame (``subject_id, label, group,
    n_scans, prob``) and a stats dict with Spearman r/p of ``prob`` vs ``n_scans``
    overall and within each label group. A correlation that survives *within* a
    label is the shortcut signal: the model reading visit count rather than biology.
    ``r`` is ``NaN`` when a group has < 2 subjects or constant values.
    """
    records: List[dict] = []
    for item in test_bundle.items:
        traj = per_visit_probs(state_dict, item, device=device)
        if not traj:
            continue
        _, final_prob = traj[-1]
        records.append(
            {
                "subject_id": item["subject_id"],
                "label": int(item["label"]),
                "group": _label_name(item["label"]),
                "n_scans": int(item["n_scans"]),
                "prob": float(final_prob),
            }
        )
    df = pd.DataFrame(records, columns=["subject_id", "label", "group", "n_scans", "prob"])

    def _spearman(sub: pd.DataFrame) -> dict:
        if len(sub) < 2 or sub["n_scans"].nunique() < 2 or sub["prob"].nunique() < 2:
            return {"r": float("nan"), "p": float("nan"), "n": int(len(sub))}
        r, p = spearmanr(sub["n_scans"], sub["prob"])
        return {"r": float(r), "p": float(p), "n": int(len(sub))}

    stats = {
        "overall": _spearman(df),
        "converter": _spearman(df[df["label"] == CONVERTER]),
        "non_converter": _spearman(df[df["label"] == NON_CONVERTER]),
    }
    return df, stats


def within_subject_prob_slopes(
    test_bundle,
    per_visit_probs: Callable[..., list],
    state_dict: Any,
    *,
    device: Any,
    min_visits: int = 2,
) -> Tuple[pd.DataFrame, dict]:
    """Per-subject trend of P(converter) as that subject's own visits accumulate.

    For each subject with >= ``min_visits`` visits, take the prefix-probability
    sequence from ``per_visit_probs`` (P at 1, 2, … visits of the *same* subject) and
    fit a least-squares slope of ``prob`` vs visit index. Returns the per-subject
    frame (``subject_id, label, group, n_scans, slope``) and per-group aggregates
    (``median_slope``, ``frac_negative``, ``n``).

    This separates the two explanations of the between-subject prob~``n_scans``
    correlation (``prob_vs_visit_count``): a **count shortcut** would leave each
    subject's own prediction roughly flat as visits are added (slope ~0), whereas
    **evidence accumulation** shows a consistent within-subject drift — e.g.
    predominantly *negative* slopes for non-converters (each additional stable visit
    lowers P(converter)).
    """
    records: List[dict] = []
    for item in test_bundle.items:
        if int(item["n_scans"]) < min_visits:
            continue
        traj = per_visit_probs(state_dict, item, device=device)
        probs = [p for _, p in traj]
        if len(probs) < 2:
            continue
        xs = np.arange(len(probs), dtype=float)
        slope = float(np.polyfit(xs, np.asarray(probs, dtype=float), 1)[0])
        records.append(
            {
                "subject_id": item["subject_id"],
                "label": int(item["label"]),
                "group": _label_name(item["label"]),
                "n_scans": int(item["n_scans"]),
                "slope": slope,
            }
        )
    df = pd.DataFrame(records, columns=["subject_id", "label", "group", "n_scans", "slope"])

    def _agg(sub: pd.DataFrame) -> dict:
        s = sub["slope"].to_numpy(dtype=float)
        if s.size == 0:
            return {"n": 0, "median_slope": float("nan"), "frac_negative": float("nan")}
        return {
            "n": int(s.size),
            "median_slope": float(np.median(s)),
            "frac_negative": float(np.mean(s < 0)),
        }

    stats = {
        "overall": _agg(df),
        "converter": _agg(df[df["label"] == CONVERTER]),
        "non_converter": _agg(df[df["label"] == NON_CONVERTER]),
    }
    return df, stats


def prob_spread_summary(prob_df: pd.DataFrame) -> dict:
    """Spread / separation of predicted probabilities, by label.

    Accepts any frame with ``label`` and ``prob`` columns (the ``prob_vs_visit_count``
    frame, or a per-visit ``trajectory_frame``). Returns per-group mean/std/IQR of
    ``prob`` plus ``separation`` = mean(prob | converter) − mean(prob | non-converter).
    A small spread / separation is the "too narrow" pattern the RNNs show relative to
    the GEC-MLP. ``NaN`` for an empty group.
    """
    def _stats(sub: pd.DataFrame) -> dict:
        p = sub["prob"].to_numpy(dtype=float)
        if p.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "iqr": float("nan")}
        q75, q25 = np.percentile(p, [75, 25])
        return {
            "n": int(p.size),
            "mean": float(np.mean(p)),
            "std": float(np.std(p, ddof=1)) if p.size > 1 else 0.0,
            "iqr": float(q75 - q25),
        }

    conv = _stats(prob_df[prob_df["label"] == CONVERTER])
    nonconv = _stats(prob_df[prob_df["label"] == NON_CONVERTER])
    separation = float("nan")
    if not (np.isnan(conv["mean"]) or np.isnan(nonconv["mean"])):
        separation = conv["mean"] - nonconv["mean"]
    return {
        "converter": conv,
        "non_converter": nonconv,
        "overall": _stats(prob_df),
        "separation": separation,
    }


__all__ = [
    "visit_counts_by_label",
    "summarize_visit_counts",
    "cohort_composition_table",
    "fixed_cohort_bundle",
    "early_detection_fixed_cohort",
    "prob_vs_visit_count",
    "within_subject_prob_slopes",
    "prob_spread_summary",
]
