---
paths:
  - "CLASSIFIER/model/**/*.py"
  - "CLASSIFIER/common/**/*.py"
  - "CLASSIFIER/notebooks/**/*.ipynb"
---

# Seeding & reproducibility

## Principles

- Every function that introduces non-determinism (sampling, shuffling, dropout scheduling, augmentations) MUST accept an explicit `rng: np.random.Generator` argument. Never read from global `np.random` or `random` state.
- Every training or evaluation entry point MUST accept an explicit `seed: int` (typically as a field on the relevant config dataclass) and derive its own RNG from it rather than relying on ambient state.

These rules apply to *new code you write*, not just to existing call sites.

## Canonical helpers

From `CLASSIFIER.common.seeding`:

```python
from CLASSIFIER.common.seeding import (
    set_seed, make_rng, make_torch_generator, seed_worker,
)
```

- `set_seed(SEED)` — process-wide seeding (`random`, `numpy`, `torch`, `PYTHONHASHSEED`, deterministic cuDNN).
- `make_rng(SEED) -> np.random.Generator` — pass as `rng=` into any function that shuffles items.
- `make_torch_generator(SEED) -> torch.Generator` — pass as `generator=` into `DataLoader`.
- `seed_worker` — pass as `worker_init_fn=seed_worker` into `DataLoader` when `num_workers > 0`.

## Notebook first-cell template

Every training notebook starts with:

```python
from CLASSIFIER.common.seeding import set_seed, make_rng, make_torch_generator, seed_worker
SEED = 42
set_seed(SEED)
rng       = make_rng(SEED)
torch_gen = make_torch_generator(SEED)
```

Then thread these into call sites:

```python
batches = make_batches(items, batch_size=16, shuffle=True, rng=rng)
loader  = DataLoader(ds, batch_size=32, shuffle=True,
                     generator=torch_gen, worker_init_fn=seed_worker)
```

`CLASSIFIER/model/GEC/train.py::build_loader(dataset, batch_size, shuffle, seed)` wires both correctly — use it when constructing GEC loaders.

## Do not

- Do not write `np.random.permutation`, `np.random.seed`, `random.shuffle`, or `torch.manual_seed` outside `common/seeding.py`.
- Do not call `make_batches(items, batch_size)` without `rng=` — it emits a `DeprecationWarning` and falls back to global RNG. Treat that warning as a bug.
- Do not seed inside a model's `__init__` — seeding is the caller's responsibility.
