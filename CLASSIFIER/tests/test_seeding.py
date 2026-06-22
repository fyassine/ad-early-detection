import numpy as np
import torch

from CLASSIFIER.common.seeding import (
    make_rng,
    make_torch_generator,
    set_seed,
)


def test_set_seed_makes_torch_deterministic():
    set_seed(42)
    a = torch.randn(8)
    set_seed(42)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_make_rng_deterministic():
    assert np.array_equal(make_rng(42).random(8), make_rng(42).random(8))


def test_make_torch_generator_deterministic():
    g1 = make_torch_generator(42)
    g2 = make_torch_generator(42)
    a = torch.randn(8, generator=g1)
    b = torch.randn(8, generator=g2)
    assert torch.equal(a, b)
