"""The save/load round trip for adapter state that is not model weights.

`model_state_for_save` deliberately strips an adapter's composite state back to a
bare `nn.Module` state dict. Anything else `load_state` needs on reload must be
declared by `checkpoint_extras` so `save_run` can merge it into the full-state
checkpoint. When those two hooks drift apart the failure is invisible until a
saved run is reloaded — which, for a ladder using `defer_test_eval: true`, is
after every run has already been spent (DOCS/flipped/PLAN.md section G).

These tests pin the contract at the level where it actually broke.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_CLASSIFIER_ROOT.parent), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import LongitudinalAdapter  # noqa: E402
from adapters.tfgn import TFGNAdapter  # noqa: E402


def _bare_tfgn_adapter():
    """A TFGNAdapter shell: the persistence hooks under test touch only class-level
    state, so bypassing __init__ keeps these unit tests free of data/GPU setup."""
    return TFGNAdapter.__new__(TFGNAdapter)


def _dummy_state(**overrides):
    state = {
        "model_state": {"w": 1},
        "log_dt_scaler_mean": [2.0878107372894648],
        "log_dt_scaler_scale": [1.5163008403108555],
        "cent_mean": 36.951565,
        "cent_std": 9.779358,
    }
    state.update(overrides)
    return state


def test_base_adapter_declares_no_extras_by_default():
    """An adapter whose eval state is weights-only needs no extras."""
    assert LongitudinalAdapter.checkpoint_extras(object(), {}) == {}


def test_tfgn_checkpoint_extras_covers_every_key_load_state_requires():
    """The contract itself: what is saved must be exactly what is demanded back.

    This is the assertion that would have caught the Tier-4 blocker — the two
    hooks disagreed, and nothing else in the suite compared them.
    """
    extras = _bare_tfgn_adapter().checkpoint_extras(_dummy_state())
    assert set(extras) == set(TFGNAdapter.STATE_NORMALIZATION_KEYS)


def test_tfgn_checkpoint_extras_round_trips_through_a_checkpoint(tmp_path):
    """save_run's merge -> load_state's read, over a real checkpoint file."""
    torch = pytest.importorskip("torch")
    from SHARED.provenance import save_full_checkpoint

    state = _dummy_state()
    extras = _bare_tfgn_adapter().checkpoint_extras(state)
    ckpt_path = tmp_path / "checkpoint_roundtrip.pth"
    save_full_checkpoint(
        ckpt_path,
        model_state=_bare_tfgn_adapter().model_state_for_save(state),
        model_config={},
        training_config={},
        **extras,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in TFGNAdapter.STATE_NORMALIZATION_KEYS:
        assert ckpt[key] == state[key], f"{key} did not survive the checkpoint write"


def test_tfgn_checkpoint_extras_raises_when_train_fold_omits_a_statistic():
    """Fail at save time, not silently at reload time (.claude/rules/errors.md)."""
    incomplete = _dummy_state()
    del incomplete["cent_std"]
    with pytest.raises(ValueError, match="cent_std"):
        _bare_tfgn_adapter().checkpoint_extras(incomplete)


def test_tfgn_load_state_raises_on_a_checkpoint_without_the_statistics(tmp_path):
    """A pre-backfill checkpoint must be refused, not scored with wrong scaling."""
    torch = pytest.importorskip("torch")

    run_dir = tmp_path / "latest"
    run_dir.mkdir()
    torch.save({"model_state_dict": {"w": 1}}, run_dir / "checkpoint_old.pth")

    adapter = _bare_tfgn_adapter()
    adapter.device = "cpu"
    with pytest.raises(KeyError, match="normalization statistics"):
        adapter.load_state(run_dir)
