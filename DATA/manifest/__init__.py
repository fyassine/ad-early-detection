"""Cross-cohort provenance manifest (DELCODE / ADNI / OASIS-3).

See DOCS/meetings/ninth-meeting/comparison-plan-v2.md §3 (Phase A.0) for the
motivation: two silent-count bugs in two weeks, both caused by counting
directory existence instead of file content. One ``cohort_manifest.csv`` per
dataset is built once here, asserted, and meant to be the single thing every
downstream step (FC extraction, splits, dataset loaders) consumes instead of
re-globbing the flat directories itself.
"""
