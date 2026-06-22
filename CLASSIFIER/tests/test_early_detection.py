"""Tests for CLASSIFIER.common.early_detection."""
from __future__ import annotations

import numpy as np

from CLASSIFIER.common.crossval import Bundle
from CLASSIFIER.common.early_detection import early_detection_table, trajectory_frame


def _bundle_with_visits(n_scans_per_subject, labels):
    items = [
        {"subject_id": f"s{i}", "label": lab, "n_scans": ns}
        for i, (ns, lab) in enumerate(zip(n_scans_per_subject, labels))
    ]
    return Bundle(list(labels), [it["subject_id"] for it in items], items)


def _truncate(bundle, n):
    """Keep subjects with >= n visits, cap their n_scans at n."""
    items = [
        {**it, "n_scans": min(it["n_scans"], n)}
        for it in bundle.items
        if it["n_scans"] >= n
    ]
    return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)


def _eval_split(state, bundle, threshold, *, device):
    labels = np.array(bundle.labels)
    return {
        "auc": 0.8,
        "sensitivity": 0.7,
        "specificity": 0.6,
        "f1": 0.65,
        "probs": np.full(len(labels), 0.5),
        "targets": labels,
    }


def test_early_detection_table_respects_min_subjects_guard():
    # 6 subjects: 3 have 3 visits, 3 have 1 visit; balanced classes.
    bundle = _bundle_with_visits([3, 3, 3, 1, 1, 1], [1, 0, 1, 0, 1, 0])
    rows = early_detection_table(
        bundle, _eval_split, _truncate, state_dict={}, threshold=0.5,
        device="cpu", min_subjects=4,
    )
    by_n = {r["n_visits"]: r for r in rows}
    # N=1 keeps all 6 -> included; N=2/N=3 keep only 3 -> skipped (min_subjects=4).
    assert 1 in by_n
    assert by_n[1]["n_subjects"] == 6
    assert 2 not in by_n and 3 not in by_n


def test_early_detection_table_skips_single_class():
    # All eligible subjects share one class at every N -> always skipped.
    bundle = _bundle_with_visits([2, 2, 2, 2], [1, 1, 1, 1])
    rows = early_detection_table(
        bundle, _eval_split, _truncate, state_dict={}, threshold=0.5,
        device="cpu", min_subjects=2,
    )
    assert rows == []


def test_early_detection_table_empty_bundle():
    rows = early_detection_table(
        Bundle([], [], []), _eval_split, _truncate, state_dict={}, threshold=0.5, device="cpu"
    )
    assert rows == []


def test_trajectory_frame_builds_expected_rows():
    bundle = _bundle_with_visits([3, 1, 2], [1, 0, 0])

    def per_visit_probs(state, item, *, device):
        # one (month, prob) per visit
        return [(m * 6, 0.5) for m in range(item["n_scans"])]

    df = trajectory_frame(bundle, per_visit_probs, state_dict={}, device="cpu", min_visits=2)
    # only the 3-visit and 2-visit subjects qualify -> 3 + 2 = 5 rows
    assert list(df.columns) == ["pid", "label", "month", "prob"]
    assert len(df) == 5
    assert set(df["pid"]) == {"s0", "s2"}
