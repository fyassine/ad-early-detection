# The old split: what was recovered, what was recreated, what stays lost

Follow-up to Confound 3 of [`../gaae-downstream-leakage-investigation.md`](../gaae-downstream-leakage-investigation.md),
which stated the historical split content is unrecoverable. That is **still true for the
split the "before" run actually used**, but it undersold what exists: the Feb-2026 legacy
split is preserved verbatim as an artifact, and one of its two halves regenerates exactly
from code. This directory holds both, plus a harness that proves which is which.

## TL;DR

Three different things were being conflated under "the old split":

| | Recoverable? | Status |
|---|---|---|
| The split the **"before" GELSTM run** (`gelstm_2026-05-20`) used | **No** | Different data lineage entirely — not the same object as the legacy split. See below. |
| The **Feb-2026 legacy split artifact** (`gaae`/`gec_data_splits.json`) | **Yes — already on disk** | Preserved verbatim here. Nothing needed recreating; it never had to be reconstructed. |
| The legacy split **regenerated from its generator code** | **Partially** | GEC: exact. GAAE: no (order-dependent). |

The headline: **the leak finding does not depend on any of this being resolved.** It
reproduces at 69/106 from the preserved artifact and 70/106 from an independent
regeneration — see [Why the leak survives anyway](#why-the-leak-survives-anyway).

## Contents

| File | What it is |
|---|---|
| `gec_data_splits.json` | Preserved Feb-2026 downstream (GEC) split. Authoritative artifact. |
| `gaae_data_splits.json` | Preserved Feb-2026 pretrain (GAAE) split. Authoritative artifact. |
| `create_gec_data_splits.py.orig` | The generator, copied unmodified from `_ad-early-detection`. Reference only — its hardcoded paths point at a directory layout that no longer exists. |
| `create_gaae_data_splits.py.orig` | Same, for GAAE. |
| `recreate_legacy_splits.py` | Runs both generators' logic against the preserved raw data and scores the output against the JSONs above. Read-only unless `--out` is passed. |

Both JSONs are byte-identical copies of `../src/splits/legacy/`, which are themselves
copies of `/mnt/e/fyassine/_ad-early-detection/data/Data-Delcode/`. sha256:

```
d8732bf449ff6033bf9aa9e235422a1da229bf7991e597a7b2872f4e6d74e0b8  gec_data_splits.json
2165e2e7c2b4843fcbef75ceb9bdd889174725b2a03f24e4737bcc1af4aace49  gaae_data_splits.json
```

## The "before" run's split is a different object — don't conflate them

The legacy split and the "before" run are **not the same cohort**, and no amount of
recovery work connects them:

| | Legacy Feb-2026 split | "Before" run (`gelstm_2026-05-20`) |
|---|---|---|
| Downstream test size | 106 patients (92 mci / 14 converter) | 16 subjects (10 neg / 6 pos) |
| Data source | `Data-Delcode/*/raw/*pearson*.npz` | `__fc_wholebrain_sch200_flat__/matrices` |

Even allowing for GELSTM's ≥2-visit filter, 106 → 16 with the positive rate inverting
(13% → 38%) is not a subsetting relationship. These are separate preprocessing
generations. Confirmed against `../src/run_summaries/before.json`.

For reference, today's downstream split is 34 test patients (20 mci / 14 converter) —
also not 16. **No artifact on disk pins the "before" run's split membership**, and per the
original investigation there is no commit to recover it from either (history squashed
2026-06-22, splits gitignored in every checkout). That gap is closed only by prevention,
not archaeology — see [Hardening](#hardening-the-real-lesson).

## Regeneration fidelity

`python recreate_legacy_splits.py`, run against sklearn 1.7.2 / seed 42:

```
=== GEC (downstream): recreated vs preserved artifact ===
  train      : recreated=319 preserved=319 jaccard=1.000  EXACT
  validation : recreated=106 preserved=106 jaccard=1.000  EXACT
  test       : recreated=107 preserved=106 jaccard=0.991  DIFFERS (+1 / -0)

=== GAAE (pretrain): recreated vs preserved artifact ===
  train      : recreated=437 preserved=436 jaccard=0.898  DIFFERS (+24 / -23)
  validation : recreated=146 preserved=146 jaccard=0.791  DIFFERS (+17 / -17)
  test       : recreated=254 preserved=254 jaccard=0.924  DIFFERS (+10 / -10)
```

### GEC reproduces exactly — except one patient, and that patient is a data bug

319/319 train and 106/106 validation come back bit-for-bit. This independently confirms
the preserved raw data *is* the Feb-era data (its mtimes, 01:39 and 03:07, predate the
03:22 generator run), and that the generator is deterministic given its input.

The single extra test patient is **`sub-dca63a3ab`, which physically exists in both raw
cohort directories**:

```
Delcode_MCI_SCD_exclude_converter_graph_data/raw/sub-dca63a3ab_pearson_correlation_matrix.npz      (Dec  2 2025)
Delcode_Converter_graph_data/raw/sub-dca63a3ab_M12_pearson_correlation_matrix.npz                  (Feb  6 2026)
Delcode_Converter_graph_data/raw/sub-dca63a3ab_M24_pearson_correlation_matrix.npz                  (Feb  6 2026)
```

This subject converted and was added to the converter cohort in February, but its stale
December MCI baseline was never deleted. The generator globs each directory independently,
so it enters the split twice — landing in mci-test and converter-validation
simultaneously. The preserved JSON has it in validation only; the code as written puts it
in both. So the on-disk `gec_data_splits.json` (written 04:04) is not quite what the
on-disk generator (04:04 > mtime 03:22, so unedited after) produces — a residual
inconsistency of exactly one patient that cannot be resolved from surviving artifacts.

It doesn't matter for any conclusion here, but it is a live label-integrity bug: **the
same subject carries two contradictory cohort labels in the raw data.** Worth checking
whether the current pipeline inherited it.

### GAAE does not reproduce — the generator is order-dependent by construction

`stratified_split_by_files()` sorts patients by file count descending, then splits.
Python's sort is stable, so ties keep their `glob()` order — and `glob()` returns raw
filesystem order, which is not guaranteed across machines or filesystems. The ties are
not an edge case; they're everything:

| cohort | patients | file-count distribution |
|---|---|---|
| ad | 104 | all 1 file |
| healthy | 201 | all 1 file |
| mci | 462 | all 1 file |
| converter | 70 | 1–6 files, every patient in a tied group |

**100% of patients sit in a tied group**, so the entire GAAE split ordering is decided by
filesystem enumeration order rather than by the seed. `RANDOM_SEED = 42` gives false
reassurance here: the seed fixes the permutation, but not the list being permuted. The
GEC generator avoids this only by accident — it never sorts, feeding `glob()` order
straight to `train_test_split`.

This is why GAAE lands at jaccard 0.79–0.92 rather than 1.000, and why the exact Feb GAAE
split exists **only** as the preserved `gaae_data_splits.json` in this directory. It is
not regenerable, and deleting that file would destroy the evidence permanently.

## Why the leak survives anyway

The leak is a **structural property of the generator's design**, not of one particular
ordering. The recreated GAAE split shares only ~90% of its training set with the
preserved one, yet lands on the same number:

| source | GAAE train ∩ GEC **test** | GAAE train ∩ GEC **val** |
|---|---|---|
| Preserved artifact (authoritative) | 0 / 106 | **69 / 106 (65%)** |
| Independent regeneration | 0 / 106 | **70 / 106 (66%)** |

Test is protected in both because the code explicitly reserves it; validation is leaked in
both because the code never mentions it. Any ordering produces the same outcome — roughly
two-thirds of the downstream validation cohort sitting in the encoder's training set —
because `create_gaae_data_splits.py` only ever reads `gec_splits["test"]` (line 51) and
lets validation fall through into its free 60/20/20.

**For the supervisor:** the 65% figure is not a fragile artifact of one lost file. It's
reproducible from scratch, and it's what the code must do given its structure.

## Hardening: the real lesson

The reason the "before" split is gone forever isn't that a file was deleted — it's that
**no run ever recorded what it trained on.** Re-derivation failed here for three
independent reasons, each sufficient on its own: the splits were gitignored, the history
was squashed, and the generator wasn't deterministic anyway.

The investigation doc's recommendation stands and is worth the small effort:

> No GAAE checkpoint or downstream run currently records a split-file hash or subject-ID
> list — this investigation would have been far more tractable if one did, and any future
> recurrence of this exact question will hit the same wall.

Concretely, and in rough priority order:

1. **Persist a subject-ID list + content hash into every `run_summary.json`.** This alone
   would have answered the whole question in one command. Path strings do not work —
   both the old and new runs cite the *same* now-nonexistent path.
2. **Sort every `glob()` before it feeds a split.** A one-word fix (`sorted(...)`) that
   converts GAAE's split from irreproducible to seed-reproducible. Today `seed=42` in that
   file does not mean what it appears to mean.
3. **Wire `assert_cohort_policy(policy="disjoint")` into `run_full_audit()`** so a
   pretrain/downstream overlap regression fails loudly instead of passing silently —
   it already exists in `SHARED/sanity.py` but nothing calls it on that path.
4. **De-duplicate the raw cohort dirs** and assert each subject carries exactly one cohort
   label (`sub-dca63a3ab` above).

Items 2–4 are code changes outside this investigation's documentation-only scope and were
not applied.

## Reproducing

```bash
source .venv/bin/activate
python DOCS/seventh-meeting-tasks/old-split/recreate_legacy_splits.py
```

Requires the sibling checkout `/mnt/e/fyassine/_ad-early-detection` (separate `.git`, last
touched ~Feb 2026) to still be present — it holds the raw graph data. The script raises
`FileNotFoundError` naming the missing directory if it isn't. It writes nothing unless
`--out DIR` is passed.

**The two JSONs in this directory are the only surviving copies of the Feb-2026 split
outside that sibling checkout, and the GAAE one cannot be regenerated. Treat them as
archival.**
