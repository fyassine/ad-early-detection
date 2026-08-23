# SOTA positioning, novelty threats, and publishability verdict

**Written:** 2026-08-22 · **Revised:** 2026-08-22 (evening, after external validation landed) · **Thesis submission:** 2026-09-03
**Scope:** where this work sits against the 2023–2026 literature, which claims survive
peer review, and which venue could take them.

---

## 1. Verdict in one paragraph

As a Master's thesis this work is **comfortably above average**, but not for the reason the
original draft argued. Its strongest assets are methodological — a pre-registered ablation,
an executable leakage audit, a bit-reproducibility contrast, and a 19-run variance
decomposition of a published baseline — not its accuracy numbers. As a paper in the framing
*"our GELSTM beats SOTA"* it is **not publishable**: N=95/25 matched cohort, single site,
and both headline *p*-values are invalid by this project's own audit. As a paper in the
framing *"a published baseline is irreproducible, and on small longitudinal cohorts the
graph encoder does not earn its place"* it is **genuinely publishable**, because that
combination has real precedent-supported novelty and a fitting venue deadline.

**Revision, 22 Aug evening.** External validation ran on ADNI and OASIS-3 and **every arm
lands at chance on held-out test** (ADNI: GELSTM 0.50/0.48, BrainTokenGT 0.43; OASIS-3:
0.53/0.57/0.54). This does not kill the paper — it changes which sentence is the paper.
"Our lean model generalizes" is gone. What replaces it is stronger for the venue actually
targeted: *neither the published baseline nor a lean alternative transfers off the cohort it
was tuned on, and the gap between within-cohort CV and held-out test is where the field's
reported numbers live.* BrainTokenGT reaching within-ADNI CV 0.705 and test 0.427 on the same
splits is that gap in a single line. **Caveat that governs everything below:** the
pre-registered A.5 gate reads at-chance *within-cohort CV* as a possible pipeline fault, and
that triage is not finished — nothing here can be written up until it is.

---

## 2. The competitive landscape

### 2.1 The baseline being audited

| | |
|---|---|
| **Brain-TokenGT** — Dong et al., MICCAI 2023 ([arXiv:2307.00858](https://arxiv.org/abs/2307.00858)) | Tokenises node + spatio-temporal edge embeddings (GIVE), reads out through a transformer with type/node identifiers (BIGTR). Evaluated on two public longitudinal AD-continuum fMRI datasets across three tasks including MCI dementia-conversion. Temporal core is EvolveGCN-H / GRCU. |

**Relevance:** it is the closest published match to this thesis's problem statement, and its
open-source release is what this project audited. The paper reports outperforming benchmark
models; this project's finding is that the released implementation ships with `train_give=False`
(recurrent core frozen at random init), diverges when enabled, and after stabilisation is not
reproducible at fixed seed.

### 2.2 The negative-result literature — this is the key context

| Paper | Finding | Relationship to this work |
|---|---|---|
| **Rethinking Functional Brain Connectome Analysis: Do Graph Deep Learning Models Help?** — npj Artificial Intelligence 2026 ([arXiv:2501.17207](https://arxiv.org/abs/2501.17207)) | Across **four large-scale neuroimaging studies**, message aggregation "does not help with predictive performance as typically assumed, but rather **consistently degrades it**." Proposes a hybrid linear + graph-attention dual-pathway model. Urges caution in adopting complex DL for connectome analysis. | **Both the biggest validation and the biggest novelty threat.** Our `none` ≈ `pretrained_frozen` result independently replicates theirs. But they published it first, at far larger scale. **Our claim must be the regime extension, never the discovery.** Their study is cross-sectional and large-N; ours is longitudinal, small-N, single-site, and additionally ablates *self-supervised pretraining*, which they do not. |
| **On the Limits of Applying Graph Transformers for Brain Connectome Classification** — [arXiv:2503.15902](https://arxiv.org/abs/2503.15902), Lara-Rangel & Heinbaugh 2025 | Graph transformers offer no advantage over conventional GNNs on NeuroGraph; **both retain accuracy with all edges removed**. Recommends reassessing whether the benchmarks capture meaningful connectivity. | Direct precedent for the BrainTokenGT audit's *spirit*. Strengthens our §4.1 argument that kNN-8 binarised edges are a lossy re-encoding of the node features. Cite as prior art; do not claim the edge-irrelevance insight as new. |
| **BrainGB** ([arXiv:2204.07054](https://arxiv.org/abs/2204.07054)) | Standardised brain-GNN benchmark. | Cite for "the field needed a benchmark because pipelines were incomparable." |

### 2.3 The scale frontier (what we are *not* competing with)

| Model | Scale | Why it matters here |
|---|---|---|
| **Brain-JEPA** — NeurIPS 2024 Spotlight ([arXiv:2409.19407](https://arxiv.org/abs/2409.19407)) | JEPA on fMRI with brain-gradient positioning + spatiotemporal masking; pretrained on UK Biobank; SOTA on demographics, cognition, diagnosis/prognosis incl. NC/MCI; beats BrainLM on 7/11 tasks. | The correct citation for "what large-scale pretraining buys" and for "predicting in latent space beats reconstructing in input space" — which is exactly our §2.5.7 pretext-misalignment argument. **Do not benchmark against it**; 10⁴–10⁵ subjects vs our 167. |
| **BrainLM** (ICLR 2024), **Brain Graph Foundation Model** ([arXiv:2506.02044](https://arxiv.org/abs/2506.02044)) | Same frontier. | Frame as the *other* coherent answer to small-N: borrow constraint from elsewhere. Unavailable to a single-site cohort. |

### 2.4 Reported MCI→AD conversion accuracies — and why they are not a leaderboard

Published rs-fMRI conversion figures span roughly **AUC 0.85–0.92**, with several
multimodal studies claiming **97% accuracy**. These are **not a benchmark this thesis should
try to beat**, and the draft must say why:

- [Rosenblatt et al., *Nature Communications* 2024](https://pubmed.ncbi.nlm.nih.gov/38234740/) —
  five forms of leakage evaluated across four large connectome datasets.
- [Shim et al., *Sci Rep* 2021](https://www.nature.com/articles/s41598-021-87157-3) —
  feature-selection leakage drastically inflates neuropsychiatric biomarker accuracy.
- [Yagis et al., *Sci Rep* 2021](https://www.nature.com/articles/s41598-021-01681-w) —
  **slice-level CV inflates accuracy by 30–55 points** on OASIS/ADNI/PPMI.

> **Drafting rule:** never place our numbers in a table beside those numbers. State the
> comparison is not meaningful, cite the leakage literature, and compete on protocol instead.

### 2.5 Time-aware sequence modelling

**T-LSTM** (Baytas et al., KDD 2017) decomposes cell memory with elapsed-time decay;
**ATTAIN** (IJCAI 2019) adds attention over prior visits. Our Δt-concatenation is *weaker*
than both. **Claim no novelty for the mechanism** — and note that §3.2's finding (90% of
DELCODE intervals identical) exposes an unremarked weakness in this whole subfield:
time-aware models are routinely evaluated on protocol-driven cohorts with little
irregularity to exploit.

---

## 3. Novelty audit — claim by claim

| Claim | Status | Verdict |
|---|---|---|
| "GELSTM outperforms Brain-TokenGT" | Direction robust (0.878 vs grand mean 0.603, best-ever BTGT run 0.818 < our mean); *p*-values invalid; N=25 test | **Publishable only as effect-size/range.** Weak as a headline. |
| "Message aggregation doesn't help" | Published first by npj AI 2026 at 4-study scale | **Not novel.** Must be framed as replication. |
| "Edges may be irrelevant" | Published by arXiv:2503.15902 | **Not novel.** Cite as prior art. |
| **"Within-seed variance exceeds between-seed variance in a published brain-graph baseline"** | No brain-graph paper surveyed reports same-seed replicates at all | **Novel and defensible.** The strongest single contribution. |
| **"The message-aggregation critique also holds longitudinally at N<150"** | npj AI 2026 is cross-sectional, large-N | **Novel as a regime extension — and now two-cohort-plus.** On ADNI `none` 0.496 ≈ `pretrained_frozen` 0.480; on OASIS-3 0.530 ≈ 0.565. The arms are indistinguishable in all three cohorts. Weaker than hoped (both are at chance externally, so the equality is partly uninformative), but the DELCODE version of the claim is unchanged and now has a replication attempt attached rather than a promise of one. |
| **"Neither model transfers off its tuning cohort"** | New, 22 Aug | **Potentially the second-strongest contribution**, and a natural companion to the variance decomposition — same theme: reported numbers do not survive a change of conditions. **Blocked on the A.5 triage.** |
| **"Reconstruction pretraining buys optimisation stability, not peak performance"** | Four-arm pre-registered separation of pretraining / architecture / encoder-existence | **Novel.** npj AI 2026 does not ablate self-supervised pretext. |
| **"Δt-conditioning is untestable on protocol-driven cohorts"** | Quantified across three cohorts: DELCODE 90% of intervals at 12 mo vs ADNI 21.9% (CV 0.647) and OASIS-3 8.1% (CV 0.574); DELCODE ablation Δ AUC = 0.000 | **Novel, useful, and now the safest claim in the thesis.** It is a property of the cohorts, not of the models, so it survives the at-chance external result untouched. The one thing missing is the Δt ablation *run on* ADNI — which is not worth running until the A.5 triage explains why ADNI is at chance. |
| "Interpretability / biomarker maps" | Integrated Gradients over an encoder of unproven contribution | **Not a claim.** Descriptive only. |

---

## 4. What would make this a strong paper

Ranked by marginal publishability per unit effort. **Re-ranked 22 Aug evening** — item 1 has
been executed and its outcome moved everything else.

1. ~~**ADNI external validation of `none` vs `pretrained_frozen`.**~~ **Done, 22 Aug.**
   Wiring landed, 24 runs across ADNI and OASIS-3 collected. Outcome: at chance in both
   cohorts, with half the GELSTM runs degenerate (all-positive predictions). The
   cross-cohort negative result exists — it is just not the cross-cohort result that was
   anticipated.
2. **Finish the A.5 triage. This is now the highest-value item by a wide margin**, because
   it decides whether the headline of §6 is *"neither model transfers"* (a finding) or
   *"our external pipeline had a bug"* (nothing). Four cheap checks, in order: label-construct
   equality across `diagnosis`/`label` columns; Δt normalisation (`MAX_INTERVAL_MONTHS = 108`
   against ADNI intervals to 2031 days); the metadata floor on ADNI/OASIS-3 for a reference
   line; zero-shot DELCODE→ADNI, the missing half of the dual gate. Nothing in this list is a
   research problem.
3. **Report the BrainTokenGT variance decomposition as the headline finding**, with the
   bit-reproducibility contrast beside it, and now with the CV-vs-test gap as its
   second panel: BrainTokenGT scores within-ADNI CV **0.705** and ADNI test **0.427**.
   Already collected; costs only rewriting.
4. **The fine-tuned-arm result is now usable** (`STABILITY_AUDIT.md` Finding 2): at a fixed
   commit the arm reproduces to ≈0.01 AUC while spanning 0.42–0.84 across seeds — the exact
   inverse of BrainTokenGT's profile, and a much better contrast than the reproducibility
   claim on its own. One run (seed 45 at HEAD) completes it.
5. **A learning-rate-scaled fine-tuning arm.** Still not implemented —
   `adapters/gelstm.py:274` builds a single Adam group at one LR. Cheap, removes an easy
   objection, but it is now a *paper* item, not a thesis item: it cannot change any claim
   before the 28 Aug hand-off and the claim can be narrowed in words instead.
6. **Weighted-edge ablation.** Removing kNN-8 binarisation separates "message passing doesn't
   help" from "this graph construction destroys the information." Still the first question a
   reviewer of the negative result will ask, and the at-chance external result makes it more
   interesting, not less.
7. ~~OASIS-3 as a third cohort. Only after ADNI.~~ **Ran alongside ADNI.** Report as the
   underpowered secondary probe it is: test n=13, AUC quantised to steps of 0.024. Never
   pooled with ADNI, never called a co-equal third cohort.

## 5. Venue shortlist

| Venue | Deadline | Fit |
|---|---|---|
| **IPMI 2027** | **7 Dec 2026**, no extensions ([call](https://2027.ipmi-conf.org/)) | **Best fit.** IPMI rewards methodological rigour and negative results more than leaderboard wins. Timeline is comfortable post-submission. |
| MICCAI 2027 | ~Feb 2027 | Natural home given the baseline is MICCAI 2023, but MICCAI favours novel method over audit. |
| MIDL 2027 | TBA (MIDL 2026 archived) | Short-paper route suits the reproducibility finding alone. |
| *Imaging Neuroscience* / *Human Brain Mapping* | Rolling | Journal route; more room for the full protocol argument. Good if ADNI + OASIS-3 both land. |
| TMLR / reproducibility venues | Rolling | Would take the variance-decomposition finding on its own, fastest. |

**Recommended:** target **IPMI 2027** with the reproducibility + failure-to-transfer framing.
The condition has changed: it is no longer "conditional on ADNI landing" (it landed) but
**conditional on the A.5 triage explaining the at-chance result**. If the triage says
"pipeline fault," fall back to the short reproducibility paper, which needs no new
experiments at all and is unaffected by any of this.

---

## 6. Framing template

Use this shape; it is defensible today.

> We audit a published spatiotemporal graph transformer for longitudinal connectome
> analysis and find that its open-source release trains no temporal component by default,
> diverges when that component is enabled, and — after stabilisation — is **not reproducible
> at a fixed random seed**: across 19 runs of one configuration, within-seed standard
> deviation (0.101) exceeds between-seed standard deviation (0.071), and a single seed spans
> an AUC range of 0.35. Multi-seed error bars for this model therefore do not measure seed
> sensitivity, and significance tests built on them are invalid. Motivated by this, we
> evaluate a deliberately parameter-lean recurrent alternative under a pre-registered
> ablation and find that removing the graph encoder entirely does not degrade performance —
> extending to the longitudinal, small-N clinical regime a result previously established
> only cross-sectionally at large N. Under external validation on two further cohorts,
> **neither the baseline nor the lean alternative separates converters from non-converters**
> (ADNI test AUC 0.43–0.50, OASIS-3 0.53–0.57), while within-cohort cross-validation for the
> baseline reads 0.705 on the same splits — a within-versus-held-out gap of the same
> magnitude as the improvements reported in this literature. We further show that
> Δt-conditioning cannot be evidenced on protocol-driven cohorts, since 90% of inter-visit
> intervals in our primary cohort are identical against 22% and 8% in two irregularly
> sampled ones.

**Conditional on the A.5 triage.** Until it clears, the external sentence must read *"we
observe at-chance external performance whose cause we do not isolate"* — which is still worth
writing, and is not the same sentence.

**Never write:** "we achieve state-of-the-art performance."
**Never write:** "our approach generalizes across cohorts." It does not, on this evidence.
**Always write:** "our pipeline re-runs byte-identically; the published baseline spans 0.35 AUC at fixed seed."
**Always write:** per-cohort tables with degeneracy counts. A mean that hides three
all-positive-prediction runs out of four is the same failure this thesis audits in others.

---

## 7. Sources

- [Beyond the Snapshot: Brain Tokenized Graph Transformer (MICCAI 2023)](https://arxiv.org/abs/2307.00858)
- [Rethinking Functional Brain Connectome Analysis (npj AI 2026)](https://www.nature.com/articles/s44387-025-00067-x) · [arXiv](https://arxiv.org/abs/2501.17207)
- [On the Limits of Applying Graph Transformers for Brain Connectome Classification](https://arxiv.org/abs/2503.15902)
- [Brain-JEPA (NeurIPS 2024)](https://arxiv.org/abs/2409.19407)
- [A Brain Graph Foundation Model](https://arxiv.org/abs/2506.02044)
- [BrainGB](https://arxiv.org/abs/2204.07054)
- [Effects of data leakage on connectome-based ML models (Nat Commun 2024)](https://pubmed.ncbi.nlm.nih.gov/38234740/)
- [Inflated prediction accuracy from feature-selection leakage (Sci Rep 2021)](https://www.nature.com/articles/s41598-021-87157-3)
- [Data leakage in brain MRI classification (Sci Rep 2021)](https://www.nature.com/articles/s41598-021-01681-w)
- [Variability and reproducibility in DL for medical image segmentation (Sci Rep 2020)](https://www.nature.com/articles/s41598-020-69920-0)
- [Non-determinism in TensorFlow ResNets](https://arxiv.org/pdf/2001.11396)
- [IPMI 2027 call for papers](https://2027.ipmi-conf.org/)
