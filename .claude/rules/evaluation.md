---
paths:
  - "CLASSIFIER/model/**/*.py"
  - "CLASSIFIER/notebooks/**/*.ipynb"
---

# Evaluation integrity

## Principles

- Thresholds are ALWAYS derived on the validation set and stored in the checkpoint as `best_threshold`.
- Test-set evaluation MUST use the validation-derived threshold from the checkpoint — never re-optimize on test.
- Any new evaluation function should accept `threshold: float` as a required argument and never compute it internally from the data being evaluated. This makes test-set leakage impossible by construction.

## Concrete contracts

- `EvalConfig.threshold_mode` defaults to `"youden"`. Youden's J is computed on the data passed in — fine for validation, leakage on test. Always pair `threshold_mode="youden"` with `val_batches`, never `test_batches`.
- `CLASSIFIER/model/GEC/train.py::evaluate_classifier(model, loader, device, *, threshold)` raises `ValueError` if `threshold is None`. This is intentional — see `errors.md`.
- After training, read `checkpoint["best_threshold"]` and pass it into the test-set evaluation call.

## Pattern

```python
# Training: Youden on val, stored in checkpoint
ckpt, history = train_classifier(model, train_loader, val_loader, ..., cfg=cfg)
val_thr = ckpt["best_threshold"]

# Test: use the val-derived threshold
test_metrics = evaluate_classifier(model, test_loader, device, threshold=val_thr)
```

## Do not

- Do not hardcode `0.5` as a classification threshold. Use `EvalConfig(threshold_mode="fixed", fixed_threshold=X)` if you need a fixed cutoff and document why.
- Do not call `evaluate_classifier(..., threshold=None)` — it raises by design.
- Do not run Youden's J on the test set under any condition.
