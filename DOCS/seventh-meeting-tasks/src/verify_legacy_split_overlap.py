"""
Reproduces the historical pretrain/downstream (GAAE/GEC) subject-overlap check
referenced in gaae-downstream-leakage-investigation.md, against the legacy split
JSONs preserved in the sibling _ad-early-detection checkout.

Requires /mnt/e/fyassine/_ad-early-detection/data/Data-Delcode/{gaae,gec}_data_splits.json
to still be present on disk.
"""
import json
from pathlib import Path

LEGACY_ROOT = Path("/mnt/e/fyassine/_ad-early-detection/data/Data-Delcode")

gaae = json.load(open(LEGACY_ROOT / "gaae_data_splits.json"))
gec = json.load(open(LEGACY_ROOT / "gec_data_splits.json"))

gaae_train = set(gaae["train"].keys())
gec_test = set(gec["test"].keys())
gec_val = set(gec.get("validation", gec.get("val", {})).keys())

print(f"GAAE train size: {len(gaae_train)}")
print(f"GEC test size:   {len(gec_test)}")
print(f"GEC val size:    {len(gec_val)}")
print(f"Overlap GAAE-train & GEC-test: {len(gaae_train & gec_test)}  (expected 0 — protected)")
print(f"Overlap GAAE-train & GEC-val:  {len(gaae_train & gec_val)}  (expected large — the leak)")
