# GELSTM capacity analysis and the case for a GRU

Context: comparing GEC-MLP vs GELSTM heads trained on the DELCODE mci/converter
subset (133 CV subjects, ~99–106 per fold). Both heads sit on top of the frozen
952k-param GAAE encoder, which is pretrained self-supervised on all 472 subjects
(~812 scans) and contributes zero trainable parameters to the downstream fit.

## The real denominator is subjects, not scans

The downstream heads are fit to one binary label per subject, not per scan.
133 CV subjects (≈100 in-fold) is the effective sample size — a subject's
multiple visits are correlated and collapse to a single label.

## Apples-to-apples learnable capacity

| | GEC-MLP | GELSTM | ratio |
|---|---|---|---|
| Trainable params | 143,745 | 240,257 | 1.67× |
| Params / subject (n=133) | 1,081 | 1,806 | — |
| Input | 396-d flat vector | 65-d × ≤6 steps, recurrent | — |
| Frozen GAAE extractor | 952,136 (external) | 952,136 (in-module) | — |

240,257 = LSTM (231,936) + classifier head (8,321). The LSTM alone is bigger
than the entire GEC-MLP.

LSTM breakdown (input=65 = 64 latent + 1 Δt, hidden=128, 2 layers):
- Layer 1: 4 × 128 × (65 + 128) + biases = 99,840
- Layer 2: 4 × 128 × (128 + 128) + biases = 132,096

So the raw 1.67× param ratio *understates* the real gap: the LSTM's 232k
weights are applied at every visit and threaded through a carried 128-d
hidden state, vs the MLP's 396→256→128→64→1 single-pass narrowing. Repeated
application + carried state + last-hidden readout is exactly the kind of
function class that memorizes a ~100-subject cohort, independent of param
count.

## Why this matters for n≈100

By the classic ≪1 param/sample heuristic, both heads are already
over-parameterized; the LSTM's recurrent inductive bias pushes it further
into the regime where the constant-predictor collapse was observed. Shrinking
the LSTM attacks capacity, but the deeper issue — recurrence being a poor fit
at this n — isn't something raw param count fully captures.

## Sizing the LSTM down

| config | LSTM | head | learnable | params/subj | vs current |
|---|---|---|---|---|---|
| current: h128 L2, head64 | 231,936 | 8,321 | 240,257 | 1,806 | 1× |
| h64 L1, head64 | 33,536 | 4,225 | 37,761 | 284 | 6.4× smaller |
| h32 L1, head32 | 12,672 | 1,089 | 13,761 | 103 | 17× smaller |
| h32 L1, no head | 12,672 | 33 | 12,705 | 96 | 19× smaller |
| h16 L1, no head | 5,312 | 17 | 5,329 | 40 | 45× smaller |

Recommended target: **h32, L1, small/no head** (~13k params, ~100/subject) —
brings the recurrent model below the MLP's footprint while keeping recurrence.

Levers, in order of leverage:
1. `lstm_layers` 2→1 — biggest single cut (−132k), nearly free in capacity terms.
2. `lstm_hidden` 128→32/16 — quadratic savings.
3. Shrink/remove `classifier_hidden` (8.3k → ~1k or 0).
4. **GRU instead of LSTM** — 3 gates vs 4 ≈ 25% fewer recurrent params for the
   same hidden size, often matches LSTM on short sequences (DELCODE has ≤6
   visits/subject, well within GRU's comfort zone — the extra LSTM cell-state
   path buys little when sequences are this short).
5. Narrow the LSTM input — re-bottleneck the 64 GAAE dims (e.g. FDR/PCA → 16)
   so `input_size` drops 65→17.
6. Drop the dead 507k decoder from the GELSTM checkpoint (hygiene only, see
   below — it already receives no gradient).
7. Avoid bidirectional (doubles params).

## A GRU-specific endpoint

If the goal is a recurrent model honestly matched to ~100 training subjects,
the target is roughly: **GRU, hidden 16–32, 1 layer, no hidden head** (~5–13k
params). At that size it's worth asking whether mean/attention-pooling visits
into a tiny MLP (i.e. converging toward the GEC design) isn't simply the more
principled choice for this n — the GRU still carries a sequential inductive
bias the data may not support, just at a capacity that's no longer the
dominant risk factor.

## Separate finding: dead decoder params in GELSTM

`get_trainable_params()` reports 747,401 trainable, but 507,144 of those are
the GAAE decoder:
- `freeze_encoder()` only freezes the encoder-side GAT/BN/FiLM modules, never
  the decoder.
- The decoder is never called in `forward` (the LSTM only uses
  `encoder.encode`), so it receives no gradient — harmless to accuracy, but:
  - bloats every GELSTM checkpoint by ~2 MB,
  - makes "trainable params" misleading (747k reported vs 240k real),
  - would silently start training if anyone ever wires the decoder into a loss.

Fix: either have `freeze_encoder()` also freeze the decoder, or don't
construct a decoder the GELSTM never uses. This is orthogonal to the
GRU/capacity discussion — checkpoint hygiene only.

## Bottom line

- Apples-to-apples learnable capacity: GELSTM 240k vs GEC-MLP 144k ≈ 1.67×,
  not the ~5× the raw "GAAE is big" intuition might suggest (that 952k figure
  is the frozen, externally-pretrained encoder, not part of the downstream fit).
- The bigger driver of the generalization gap is the recurrent inductive bias
  (state + per-step reuse + last-hidden readout), not the headline param
  count.
- A GRU at h16–h32, 1 layer, no head is the right-sized recurrent baseline to
  test against the GEC-MLP for this cohort size.
