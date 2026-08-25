<a id="top"></a>

# TFGN: Summary

One-page version of [`METHODS.md`](METHODS.md) — the full methods, results, scorecard and
limitations for the temporal-first ablation ladder. Every claim below links to the section
that backs it; read there for numbers, tables and pre-registration detail.

## The idea

Every prior longitudinal classifier in this project pools each visit's whole-brain
connectivity graph to one vector *before* modelling the trajectory — region identity is
destroyed before the temporal model ever sees it. **TFGN flips the order**: a per-region
LSTM models each of 200 regions' own trajectory first, and pooling happens only afterward.
See [§1.1 Motivation and hypothesis](METHODS.md#11-motivation-and-hypothesis) and
[§1.4 The TFGN architecture](METHODS.md#14-the-tfgn-architecture).

## What was tested

76 ablation runs + 1 escalation arm, 4 seeds each, all trained and selected on a pooled
248-subject ADNI+DELCODE cross-validation pool built for this work
([§1.2 Data](METHODS.md#12-data-preprocessing-and-labels)). The ablation ladder walks the
architecture on piece at a time — gate, graph encoder, fusion, pooling, dual-score head —
against a pre-registered stopping rule so nothing survives on vibes
([§1.7 The ladder](METHODS.md#17-the-ablation-ladder--arms),
[§1.8 Evaluation protocol](METHODS.md#18-evaluation-protocol)).

**Only 3 of 19 arms were ever scored on the held-out in-domain test (n=64) or the external
OASIS-3 cohort (n=60)** — those are one-shot resources, spent exactly once on the frozen
winner and its two designated secondaries. See the scope table in
[§1.8](METHODS.md#18-evaluation-protocol).

## The result

**The flip wins in-domain, at far lower capacity.** The selected model — plain LSTM,
mean-pool, linear head, no graph stage at all — beats both spatial-first baselines while
being **14× smaller** than one and **7.6× smaller (trainable)** than the other, so the gain
is architectural, not capacity. See
[§2.1 Scorecard](METHODS.md#21-scorecard) and
[§2.3 The winner](METHODS.md#23-the-winner-and-why-the-win-is-not-capacity).

**The advantage is specifically a long-sequence advantage.** Under a short, competitor-
matched window the two architectures cross over — spatial-first gains, temporal-first
loses. See [§2.6 Matched-window head-to-head](METHODS.md#26-matched-window-head-to-head--a-crossover-not-a-defeat).

**Nothing added on top of the bare flip survived the ablation.** Gate, graph fusion,
attentive pooling and the reconstruction objective were all dropped or void; only the
dual-score interpretability head was kept (classification-neutral by design). See
[§2.2 Stopping-rule verdicts](METHODS.md#22-stopping-rule-verdicts-every-rung-against-s1) and
[§2.10 Scaling gate: closed](METHODS.md#210-scaling-gate-closed).

**The interpretability map is reproducible but not disease-specific.** It is stable across
seeds and correlates with an independent, model-free drift measure, but does not show the
pre-registered DMN enrichment. See
[§2.9 Interpretability validation](METHODS.md#29-interpretability-validation--reproducible-but-not-dmn-specific).

## The open problem

**External transfer fails.** Every model tested on OASIS-3 lands at chance (AUC ≈0.49),
despite 0.77–0.79 in-domain. The leading cause is a cohort-identity shortcut: a probe
decodes ADNI-vs-DELCODE from the model's own latent at ≈0.86 AUC, well past the
pre-registered 0.75 escalation threshold — which had been firing since the first ladder
runs and was not caught until the final read pulled every arm together. The pre-registered
adversarial fix was tried and made things worse on both axes it targeted. See
[§2.7 Tier-4 held-out reads](METHODS.md#27-tier-4-held-out-reads-one-pass-spent-once) and
[§2.8 The cohort shortcut](METHODS.md#28-the-cohort-shortcut-and-a-mitigation-that-was-tried-and-failed).

## Bottom line

Deferring spatial pooling until after per-region temporal encoding is a real,
architecturally-cheap win for in-domain prediction on long trajectories — but the pooled
training protocol as built does not produce a representation that generalises to an unseen
cohort, and that is now the open problem, not a side note. Full caveats and every number:
[§3 Summary and limitations](METHODS.md#3-summary-and-limitations).

---

[↑ Full methods and results →](METHODS.md#top)
