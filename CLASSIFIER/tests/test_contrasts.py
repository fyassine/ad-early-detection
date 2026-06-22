"""Tests for CLASSIFIER.common.contrasts — frozen pre-registered contrasts."""
from __future__ import annotations

import pytest

from CLASSIFIER.common.contrasts import CONTRASTS, Contrast, contrast_by_name, regions_referenced

# Slugs must match folder names under DATA/DELCODE/__fc_<slug>_*_flat__
VALID_REGIONS = {
    "wholebrain",
    "dmn",
    "hippo",
    "limbic",
    "dan",
    "dmn-hippo",
    "dmn-limbic",
    "dmn-hippo-limbic",
    "dmn-hippo-limbic-dan",
}


def test_contrast_names_unique():
    names = [c.name for c in CONTRASTS]
    assert len(names) == len(set(names))


def test_all_regions_are_valid_slugs():
    for c in CONTRASTS:
        assert c.region_a in VALID_REGIONS, f"{c.name}: unknown region {c.region_a}"
        assert c.region_b in VALID_REGIONS, f"{c.name}: unknown region {c.region_b}"
        assert c.region_a != c.region_b, f"{c.name}: contrast must compare different regions"


def test_contrast_dataclass_is_frozen():
    c = CONTRASTS[0]
    with pytest.raises(Exception):
        c.name = "modified"  # type: ignore[misc]


def test_lookup_by_name():
    c = contrast_by_name("H1_wholebrain_vs_dmn")
    assert isinstance(c, Contrast)
    assert c.region_a == "wholebrain"


def test_lookup_unknown_raises():
    with pytest.raises(KeyError):
        contrast_by_name("not-a-real-contrast")


def test_regions_referenced_subset_of_valid():
    refs = set(regions_referenced())
    assert refs.issubset(VALID_REGIONS)
    assert "wholebrain" in refs
    assert "dmn" in refs
