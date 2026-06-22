# Errors — fail loudly

## Principle

Prefer raising `ValueError` (or a more specific exception) with a descriptive message over silent fallbacks. Never use a default that silently changes behavior — `threshold=0.5` as a fallback when the caller forgot to pass one is exactly the kind of thing that hides bugs and leaks information.

Guard clauses at function entry are preferred over defensive checks scattered through logic.

## Canonical example

`CLASSIFIER/model/GEC/train.py::evaluate_classifier` raises when `threshold is None`:

```python
if threshold is None:
    raise ValueError(
        "evaluate_classifier requires an explicit threshold (typically the "
        "validation-derived best_threshold). Choosing a threshold from test "
        "metrics would leak."
    )
```

Do not soften this. If a future caller forgets the threshold, the loud error is the desired outcome.

## When you must support a default

Use a sentinel that makes the choice explicit:

```python
def compute(x, mode: Literal["youden", "fixed"] = "youden", *, fixed_threshold: float | None = None):
    if mode == "fixed" and fixed_threshold is None:
        raise ValueError("mode='fixed' requires fixed_threshold=")
```
