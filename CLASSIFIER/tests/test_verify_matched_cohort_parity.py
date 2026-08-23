"""Unit tests for scripts/verify_matched_cohort_parity.py's diff logic.

Only ``check_parity`` (pure function over two ID lists) and
``_require_matched_window`` (a guard clause) are exercised here — the
``build_*_bundle`` functions require the real DELCODE data tree and are covered
end-to-end by running the script itself (see Phase B.1.1/B.1.2 in
DOCS/timeline/MASTER_PLAN.md §3), not by this unit test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_matched_cohort_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_matched_cohort_parity", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vmcp = _load_module()


def test_check_parity_identical_set_and_order():
    ids = ["s1", "s2", "s3"]
    assert vmcp.check_parity(ids, list(ids)) is True


def test_check_parity_same_set_different_order():
    assert vmcp.check_parity(["s1", "s2", "s3"], ["s2", "s1", "s3"]) is False


def test_check_parity_different_set():
    assert vmcp.check_parity(["s1", "s2", "s3"], ["s1", "s2", "s4"]) is False


def test_require_matched_window_passes_on_2_3():
    adapter = SimpleNamespace(min_visits=2, max_visits=3)
    vmcp._require_matched_window(adapter, "some-exp-id")  # must not raise


def test_require_matched_window_raises_on_mismatch():
    adapter = SimpleNamespace(min_visits=1, max_visits=5)
    with pytest.raises(ValueError, match="min_visits=2,max_visits=3"):
        vmcp._require_matched_window(adapter, "some-exp-id")
