"""
Pre-registered cross-region contrasts for the H1/H2/H3 hypotheses.

These are the only contrasts that count as confirmatory tests for the
cross-region comparison; everything else (full 8x8 heatmap) is exploratory
and should be reported as such. The list is frozen — adding a contrast after
seeing results would inflate the family-wise error rate.

Region keys must match the DELCODE folder slugs (without the `__fc_` prefix
or `_flat__` suffix) used by the comparison notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Contrast:
    name: str
    region_a: str
    region_b: str
    hypothesis: str
    rationale: str


CONTRASTS: Tuple[Contrast, ...] = (
    Contrast(
        name="H1_wholebrain_vs_dmn",
        region_a="wholebrain",
        region_b="dmn",
        hypothesis="H1",
        rationale="Does narrowing from whole-brain to DMN preserve or lose AD signal?",
    ),
    Contrast(
        name="H1_wholebrain_vs_best_memory",
        region_a="wholebrain",
        region_b="dmn-hippo-limbic",
        hypothesis="H1",
        rationale="Does the full memory system match the whole-brain ceiling?",
    ),
    Contrast(
        name="H2_dmn_vs_dmn_hippo",
        region_a="dmn",
        region_b="dmn-hippo",
        hypothesis="H2",
        rationale="Does adding hippocampus to DMN add subcortical memory signal?",
    ),
    Contrast(
        name="H2_dmn_vs_dmn_limbic",
        region_a="dmn",
        region_b="dmn-limbic",
        hypothesis="H2",
        rationale="Does adding limbic cortex to DMN add medial-temporal signal?",
    ),
    Contrast(
        name="H2_dmn_hippo_vs_dmn_hippo_limbic",
        region_a="dmn-hippo",
        region_b="dmn-hippo-limbic",
        hypothesis="H2",
        rationale="Marginal value of limbic on top of DMN+hippocampus.",
    ),
    Contrast(
        name="H3_memory_vs_memory_plus_dan",
        region_a="dmn-hippo-limbic",
        region_b="dmn-hippo-limbic-dan",
        hypothesis="H3",
        rationale="Does adding the antagonistic DAN add signal or noise to memory system?",
    ),
    Contrast(
        name="SANITY_hippo_vs_dmn_hippo",
        region_a="hippo",
        region_b="dmn-hippo",
        hypothesis="sanity",
        rationale="Does DMN cortex actually contribute beyond hippocampus alone?",
    ),
)


def contrast_by_name(name: str) -> Contrast:
    for c in CONTRASTS:
        if c.name == name:
            return c
    raise KeyError(f"No registered contrast named {name!r}. Known: {[c.name for c in CONTRASTS]}")


def regions_referenced() -> Tuple[str, ...]:
    """Unique region slugs appearing in any contrast — useful for sanity-checking checkpoint coverage."""
    seen: list[str] = []
    for c in CONTRASTS:
        for r in (c.region_a, c.region_b):
            if r not in seen:
                seen.append(r)
    return tuple(seen)
