import numpy as np
import pytest

from CLASSIFIER.common.splits import make_splits


def test_deterministic_with_same_seed():
    sids = list(range(100))
    labels = [i % 2 for i in sids]
    a = make_splits(sids, labels, seed=42)
    b = make_splits(sids, labels, seed=42)
    assert np.array_equal(a["train"], b["train"])
    assert np.array_equal(a["val"],   b["val"])
    assert np.array_equal(a["test"],  b["test"])


def test_no_overlap_between_partitions():
    sids = list(range(100))
    labels = [i % 2 for i in sids]
    splits = make_splits(sids, labels, seed=42)
    t, v, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert t.isdisjoint(v)
    assert t.isdisjoint(te)
    assert v.isdisjoint(te)
    assert t | v | te == set(range(100))


def test_split_fractions_respected():
    sids = list(range(100))
    splits = make_splits(sids, labels=None, seed=42,
                         val_frac=0.2, test_frac=0.1, stratify=False)
    # val_frac and test_frac are fractions of the total dataset, achieved by
    # first carving out test_frac then val_frac/(1-test_frac) of the remainder.
    assert len(splits["test"]) == 10
    assert len(splits["val"])  == 20
    assert len(splits["train"]) == 70


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        make_splits(list(range(10)), labels=None, seed=42,
                    val_frac=0.6, test_frac=0.6)
