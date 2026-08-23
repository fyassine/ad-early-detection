[← §6 — Evidence tables](SECTION_06_EVIDENCE_TABLES.md) | [Master Plan](MASTER_PLAN.md) | [§8 — Remaining work, day by day →](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md)

---

# §7 — Positioning: SOTA, novelty, venue

## Verdict in one paragraph

As a Master's thesis this work is comfortably above average, but not for the reason an
earlier draft argued. Its strongest assets are methodological — a pre-registered ablation, an
executable leakage audit, a bit-reproducibility contrast, and a 19-run variance decomposition
of a published baseline — not its accuracy numbers. As a paper in the framing *"our GELSTM
beats SOTA"* it is **not publishable**: N=95/25 matched cohort, single site, both headline
p-values were invalid before correction. As a paper in the framing *"a published baseline is
irreproducible, and on small longitudinal cohorts the graph encoder does not earn its place"*
it is **genuinely publishable** — real precedent-supported novelty, a fitting venue deadline.

**Revision after external validation:** every external arm lands at chance. This doesn't kill
the paper — it changes which sentence is the paper. "Our lean model generalizes" is gone.
What replaces it: *neither the published baseline nor a lean alternative transfers off the
cohort it was tuned on, and the gap between within-cohort CV and held-out test is where the
field's reported numbers live.* BrainTokenGT's within-ADNI CV 0.705 vs test 0.427 on the same
splits is that gap in one line.

## Competitive landscape

**Baseline being audited:** Brain-TokenGT (Dong et al., MICCAI 2023, arXiv:2307.00858) —
tokenises node + spatio-temporal edge embeddings, reads out through a transformer. The
open-source release ships with `train_give=False` (recurrent core frozen at random init),
diverges when enabled, and after stabilisation is not reproducible at fixed seed (§5).

**The negative-result literature — key context, and the biggest novelty threat:**

| Paper | Finding | Relationship to this work |
|---|---|---|
| *Rethinking Functional Brain Connectome Analysis* (npj AI 2026, arXiv:2501.17207) | Across 4 large-scale studies, message aggregation "consistently degrades" predictive performance | **Biggest validation and biggest novelty threat.** Our `none ≈ pretrained_frozen` independently replicates this, at far larger scale, published first. **Our claim must be the regime extension** (longitudinal, small-N, self-supervised pretraining ablated), never the discovery. |
| *On the Limits of Applying Graph Transformers* (arXiv:2503.15902) | Graph transformers offer no advantage over conventional GNNs; both retain accuracy with all edges removed | Precedent for the BrainTokenGT audit's spirit; cite as prior art for edge-irrelevance, don't claim it as new. |
| BrainGB (arXiv:2204.07054) | Standardised brain-GNN benchmark | Cite for "the field needed a benchmark because pipelines were incomparable." |

**What we are not competing with:** Brain-JEPA (NeurIPS 2024, UK Biobank scale), BrainLM,
Brain Graph Foundation Model — 10⁴–10⁵ subjects vs our 167. Correct citation for "what
large-scale pretraining buys" and for the pretext-misalignment argument (§5, explanation 3),
never a benchmark target.

**Reported MCI→AD conversion accuracies (0.85–0.92, some 97%) are not a leaderboard.** The
leakage literature explains why: Rosenblatt et al. (*Nat Commun* 2024, five leakage forms
across four connectome datasets), Shim et al. (*Sci Rep* 2021, feature-selection leakage),
Yagis et al. (*Sci Rep* 2021, slice-level CV inflating accuracy 30–55 points). **Drafting
rule: never place our numbers beside those numbers.** State the comparison isn't meaningful,
cite the leakage literature, compete on protocol instead.

**Time-aware sequence modelling:** T-LSTM and ATTAIN are both *stronger* than our
Δt-concatenation — claim no novelty for the mechanism. The real contribution is exposing that
DELCODE's 90%-identical intervals make Δt untestable there at all, an unremarked weakness in
this whole subfield (models routinely evaluated on protocol-driven cohorts with little
irregularity to exploit).

## Novelty audit, claim by claim

| Claim | Status | Verdict |
|---|---|---|
| "GELSTM outperforms Brain-TokenGT" | Direction robust; corrected p-values now valid-but-underpowered (§6) | Publishable as effect-size/range; weak as a headline alone |
| "Message aggregation doesn't help" | Published first by npj AI 2026 at 4-study scale | Not novel — frame as replication |
| "Edges may be irrelevant" | Published by arXiv:2503.15902 | Not novel — cite as prior art |
| **"Within-seed variance exceeds between-seed variance in a published brain-graph baseline"** | No surveyed paper reports same-seed replicates at all | **Novel and defensible — strongest single contribution** |
| **"Message-aggregation critique also holds longitudinally at N<150"** | npj AI 2026 is cross-sectional, large-N | Novel as a regime extension; now two-cohort-plus, weaker than hoped since both externals are at chance |
| **"Neither model transfers off its tuning cohort"** | New, resolved 23 Aug | **Second-strongest contribution** — the Track 2 triage clearing is what makes this writable |
| **"Reconstruction pretraining buys optimisation stability, not peak performance"** | Four-arm pre-registered separation of pretraining/architecture/encoder-existence | Novel — npj AI 2026 doesn't ablate self-supervised pretext |
| **"Δt-conditioning is untestable on protocol-driven cohorts"** | Quantified across 3 cohorts (§6, §3 D5) | **Novel, useful, safest claim in the thesis** — a property of the cohorts, survives the at-chance external result untouched |
| "Interpretability / biomarker maps" | Integrated Gradients over an encoder of unproven contribution | Not a claim — descriptive only |

## Venue shortlist

| Venue | Deadline | Fit |
|---|---|---|
| **IPMI 2027** | 7 Dec 2026, no extensions | **Best fit** — rewards methodological rigour and negative results over leaderboard wins |
| MICCAI 2027 | ~Feb 2027 | Natural home (baseline is MICCAI 2023) but favours novel method over audit |
| MIDL 2027 | TBA | Short-paper route suits the reproducibility finding alone |
| *Imaging Neuroscience* / *Human Brain Mapping* | Rolling | Journal route, room for full protocol argument |
| TMLR / reproducibility venues | Rolling | Would take the variance-decomposition finding alone, fastest |

**Recommended: IPMI 2027**, reproducibility + failure-to-transfer framing.

## Framing template

> We audit a published spatiotemporal graph transformer for longitudinal connectome analysis
> and find that its open-source release trains no temporal component by default, diverges
> when that component is enabled, and — after stabilisation — is **not reproducible at a
> fixed random seed**: across 19 runs of one configuration, within-seed standard deviation
> (0.101) exceeds between-seed standard deviation (0.071), and a single seed spans an AUC
> range of 0.35. Multi-seed error bars for this model therefore do not measure seed
> sensitivity, and significance tests built on them are invalid. Motivated by this, we
> evaluate a deliberately parameter-lean recurrent alternative under a pre-registered
> ablation and find that removing the graph encoder entirely does not degrade performance —
> extending to the longitudinal, small-N clinical regime a result previously established only
> cross-sectionally at large N. Under external validation on two further cohorts, **neither
> the baseline nor the lean alternative separates converters from non-converters** (ADNI test
> AUC 0.43–0.50, OASIS-3 0.53–0.57), while within-cohort cross-validation for the baseline
> reads 0.705 on the same splits — a within-versus-held-out gap of the same magnitude as the
> improvements reported in this literature. We further show that Δt-conditioning cannot be
> evidenced on protocol-driven cohorts, since 90% of inter-visit intervals in our primary
> cohort are identical against 22% and 8% in two irregularly sampled ones.

**Never write:** "we achieve state-of-the-art performance." "Our approach generalizes across
cohorts" — it does not, on this evidence.
**Always write:** "our pipeline re-runs byte-identically; the published baseline spans 0.35
AUC at fixed seed." Per-cohort tables with degeneracy counts — a mean hiding three
all-positive runs out of four is the failure this thesis audits in others.

---

[← §6 — Evidence tables](SECTION_06_EVIDENCE_TABLES.md) | [Master Plan](MASTER_PLAN.md) | [§8 — Remaining work, day by day →](SECTION_08_REMAINING_WORK_DAY_BY_DAY.md)
