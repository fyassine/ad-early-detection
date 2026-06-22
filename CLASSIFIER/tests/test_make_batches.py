import numpy as np

from CLASSIFIER.common.seeding import make_rng
from CLASSIFIER.model.GELSTM.train import make_batches


def _items(n):
    return [{"id": i} for i in range(n)]


def test_no_shuffle_preserves_order():
    items = _items(10)
    batches = make_batches(items, batch_size=3, shuffle=False)
    flat = [b["id"] for batch in batches for b in batch]
    assert flat == list(range(10))


def test_batch_sizes_and_total_count():
    items = _items(10)
    batches = make_batches(items, batch_size=3, shuffle=False)
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    assert sum(len(b) for b in batches) == 10


def test_shuffle_deterministic_with_same_seed():
    items = _items(20)
    a = make_batches(items, batch_size=4, shuffle=True, rng=make_rng(42))
    b = make_batches(items, batch_size=4, shuffle=True, rng=make_rng(42))
    a_flat = [x["id"] for batch in a for x in batch]
    b_flat = [x["id"] for batch in b for x in batch]
    assert a_flat == b_flat


def test_shuffle_actually_permutes():
    items = _items(20)
    shuffled = make_batches(items, batch_size=4, shuffle=True, rng=make_rng(42))
    flat = [x["id"] for batch in shuffled for x in batch]
    assert sorted(flat) == list(range(20))
    assert flat != list(range(20))
