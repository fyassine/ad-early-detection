# Thesis Outline

Revised table of contents, superseding the version reviewed in August 2026. Every change
here traces to one of three rounds of supervisor feedback; the mapping is recorded in
[What changed](#what-changed) and in the [provenance](#provenance-table) and
[claims](#claims-register) tables at the end.

This document is an outline. It fixes structure, section titles, the order of the argument
and the wording rules that the prose must obey. It does not contain chapter prose.

---

## Contents

1. [Title and scope](#title-and-scope)
2. [Research questions](#research-questions)
3. [What changed](#what-changed)
4. [The outline](#the-outline)
5. [Provenance table](#provenance-table)
6. [Claims register](#claims-register)
7. [Methods and results separation](#methods-and-results-separation)
8. [Contribution coverage](#contribution-coverage)
9. [Challenge coverage](#challenge-coverage)
10. [Abbreviations](#abbreviations)

---

<a id="title-and-scope"></a>

## 1. Title and scope

**Recommended title**

> Longitudinal Graph Representation Learning for MCI-to-AD Conversion Prediction:
> A Cross-Cohort Evaluation

The previous working title, "Early Detection of Alzheimer's Disease", is broader than what
the thesis evaluates. The recommended title names the clinical target (MCI-to-AD
conversion), the method family (longitudinal graph representation learning) and the
evaluation axis that carries the main negative result (cross-cohort).

German title, to be confirmed:

> Longitudinales Graph-Repräsentationslernen zur Vorhersage der Konversion von MCI zu
> Alzheimer-Demenz: Eine kohortenübergreifende Evaluation

**Scope statement.** Placed verbatim in §1.1 and again at the head of Chapter 8:

> The primary scope of this thesis is the methodological evaluation of supervised
> longitudinal classification approaches for MCI-to-AD conversion. It does not claim to
> develop or validate a normative healthy-reference framework for individualized
> abnormality detection.

**One-sentence description**, for the abstract and the §1.4 objectives:

> A methodological evaluation of graph-based longitudinal classification approaches for
> MCI-to-AD conversion, including cross-cohort transfer and reproducibility analyses.

---

<a id="research-questions"></a>

## 2. Research questions

Four questions, stated at the head of Chapter 4 (§4.1) and answered in the same order in
Chapter 6 (§6.2 to §6.5). They are the organising spine of both chapters.

| | Question | Posed in | Answered in | Gap it addresses |
|---|---|---|---|---|
| **RQ1** | Does longitudinal modelling add information beyond static functional-connectivity representations? | §4.1 | §6.2 | §2.4.3 |
| **RQ2** | Does spatial-first graph representation learning add value? | §4.1 | §6.3 | §2.4.2 |
| **RQ3** | Does temporal-first modelling improve over spatial-first modelling? | §4.1 | §6.4 | §2.4.1 |
| **RQ4** | Do the findings transfer to an independent cohort? | §4.1 | §6.5 | §2.4.4 |

RQ4 is worded as transfer rather than validation. OASIS-3 is framed throughout as a
cross-cohort transfer evaluation, not as a validation of an identically defined clinical
target.

---

<a id="what-changed"></a>

## 3. What changed

| Feedback | Action |
|---|---|
| §1.3 is generic to medical ML and too broad for a core introduction section | Renamed to "Deep Learning in Neurodegenerative Disease" and rewritten as existing work paired with the challenges it meets, forward-referencing Chapter 2. Its four sample-constraint bullets move to §2.4.3 and Chapter 3. |
| §2.5 has ten research gaps, some at hyperparameter level | Reduced to four (§2.4.1 to §2.4.4), each mapped to one research question. The "summary of positions" subsection becomes a closing paragraph. |
| Chapter 4 has 19 sections and is hard to follow | Reduced to 7. All verdicts, void rungs and the cohort-escalation outcome move to Chapter 6. |
| "The SOTA Repair Framework" is opaque | Chapter 5 retitled "Baselines". |
| "Comprehensive Ablations" reads oddly | Chapter 6 retitled "Experimental Results and Ablations". |
| Make the pre-existing framework explicit | Ten provenance insertions, tracked in the [provenance table](#provenance-table). |
| Make 2 to 4 research questions explicit and structure Methods and Results by them | Four RQs, §4.1 and §6.2 to §6.5. |
| Separate Methods from Results strictly | Chapter 4 states what was built and how it would be judged; Chapter 6 states what happened. Checklist in [§7](#methods-and-results-separation). |
| Do not claim causality for the cohort probe | [Claims register](#claims-register), rows 1 to 3. |
| Tone down comparative claims about Brain-TokenGT | [Claims register](#claims-register), rows 4 to 6. |
| "Pre-registered" overstates what was done | Replaced by "pre-specified" throughout. See the note under the claims register. |
| Too much implementation detail in the main text | Appendices B and C absorb configuration knobs, run identifiers, registry entries and code paths. |

---

<a id="the-outline"></a>

## 4. The outline

Annotations per section: **what it argues**, `backed by:` the source, and where the section
has moved, `was:` its number in the reviewed version.

### Chapter 1: Introduction

**1.1 Clinical motivation and the diagnostic paradigm shift**
Why prediction at the MCI stage is the decision point of interest, and what a conversion
label means clinically. Carries the scope statement verbatim.
`was:` 1.1

**1.2 Resting-state fMRI as a non-invasive window**
Why functional connectivity is the modality, relative to CSF, PET and structural MRI.
`was:` 1.2

**1.3 Deep learning in neurodegenerative disease**
A short survey of what has been built for neurodegenerative prediction from imaging, paired
with the recurring difficulties: small labelled cohorts relative to feature dimensionality,
irregular follow-up, multi-site heterogeneity. Ends by pointing forward to Chapter 2 for
the detailed treatment.
`was:` 1.3, previously "The Actual Research Setting: Deep Learning Under Severe Sample
Constraint". The four bullets it carried are relocated to §2.4.3 and Chapter 3.

**1.4 Research questions, objectives and contributions**
RQ1 to RQ4 stated once, then the contribution list keyed to the chapters that deliver each.
`backed by:` `DOCS/critical_points/contributions.md`
`was:` 1.4

**1.5 Thesis organisation**
`was:` 1.5

### Chapter 2: Foundations and Related Work

**2.1 Brain connectivity and parcellation schemes**
Atlases, connectivity estimation, the 200-region choice and what it commits to.
`was:` 2.1

**2.2 Graph representation learning in neuroimaging**
Graph neural networks on connectomes, graph autoencoders, pooling operators.
`was:` 2.2

**2.3 Longitudinal and spatiotemporal modelling**
Recurrent approaches and spatiotemporal graph transformers, including Brain-TokenGT as the
published comparator introduced properly in Chapter 5.
`was:` 2.3

**2.4 Open problems addressed in this thesis**
Four problems, reduced from ten. Closes with a short paragraph stating the position this
thesis takes on each, which was previously the separate subsection 2.5.10.
`was:` 2.5

> **2.4.1 Where spatial aggregation belongs in a longitudinal pipeline.**
> Published longitudinal connectome classifiers aggregate a visit's graph to a single
> vector before the temporal model sees it, so region identity is discarded before any
> trajectory is modelled. Whether that ordering is necessary, and whether the graph
> topology carries signal beyond the connectivity matrix it is built from, are open. This
> is the gap RQ3 addresses.
> `was:` 2.5.1 and 2.5.2 merged

> **2.4.2 What self-supervision on connectomes learns.**
> Reconstruction is the default pretext task for connectome autoencoders, but the objective
> rewards subject idiosyncrasy, and between-patient variation within a cohort can exceed
> between-group variation. Whether a reconstruction-pretrained encoder yields features that
> are useful for a downstream group contrast is open. This is the gap RQ2 addresses.
> `was:` 2.5.7, extended with the GAAE-versus-GEC observation
> `backed by:` `DOCS/critical_points/contributions.md`, "GAAE failing at classification"

> **2.4.3 Evidence standards under small samples.**
> At cohort sizes in the low hundreds, three practices are unsettled: which floor a model
> must clear before an architectural claim is admissible, whether same-seed replicate
> variance is reported at all, and how longitudinal evaluation is protocolised. Treated
> here as one question about what a claim must survive.
> `was:` 2.5.3, 2.5.4 and 2.5.5 merged, reframed away from the hyperparameter-level phrasing

> **2.4.4 Time and cohort as confounds.**
> Follow-up duration correlates with the outcome and is rarely controlled; visit-time
> semantics differ between cohorts in ways that silently drop or misalign scans; and
> pooling cohorts to buy sample size introduces cohort identity as a decodable nuisance
> variable. This is the gap RQ4 addresses.
> `was:` 2.5.6, 2.5.8 and 2.5.9 merged
> `backed by:` `DOCS/critical_points/challenges.md`

### Chapter 3: Data, Cohorts and Preprocessing

**3.1 The DELCODE cohort and the conversion label**
Cohort composition, and the label definition together with the follow-up window that
defines it. States the tradeoff explicitly: the window choice trades cohort size against
label noise.
`backed by:` `DOCS/critical_points/challenges.md`, "Label definition"
`was:` 3.1

**3.2 Longitudinal sampling and follow-up regularity**
Visit spacing, missing follow-ups, variable sequence length, and the masking and
time-conditioning consequences. Records that DELCODE follow-up is more regular than the
irregular-sampling literature assumes, and that the pooled protocol reintroduces
irregularity.
`backed by:` `DOCS/critical_points/challenges.md`, "Irregular longitudinal sampling"
`was:` 3.2

**3.3 External cohorts: ADNI and OASIS-3**
Opens with the project-context sentence: as part of the broader project, ADNI and OASIS-3
were included as additional cohorts for cross-cohort validation; during the thesis these
datasets were processed and integrated into the existing analysis framework. Then covers
field strength, acquisition protocol and phase differences.
`backed by:` `DOCS/critical_points/challenges.md`, "Site and protocol heterogeneity"
`was:` 3.3. Provenance item 2.

**3.4 The pooled ADNI and DELCODE protocol**
Opens by stating that the pooled protocol implemented during the thesis builds on the
pre-existing DELCODE split definitions and leakage-control framework, extending those
structures to the additional cohort. The technical description of the asset-building
pipeline follows unchanged.
`was:` 3.4. Provenance item 3.

**3.5 Preprocessing and confound management**
Opens by stating that DELCODE had been preprocessed before the start of this thesis using
the laboratory's existing preprocessing pipeline; that during the thesis the same
established framework was applied to ADNI and OASIS-3; and that the cohort-specific
adaptations required for processing and integrating those datasets were implemented as part
of the thesis. Then confound regression, motion handling and the attrition accounting.
`was:` 3.5. Provenance item 4.

**3.6 Group-level characterisation before modelling**
Converters against non-converters characterised with PERMANOVA on the connectivity
matrices, the network-based statistic for cluster-level differences, and per-edge effect
sizes under false-discovery-rate control. Establishes whether a group difference is present
before a deep model is asked to find one.
`backed by:` `DOCS/critical_points/contributions.md`, "Classical statistics before any
model was fitted"
`was:` absent from the reviewed version

**3.7 Split integrity and leakage control**
Patient-level splitting, the isolated test manifest, post-conversion visit exclusion,
validation-derived thresholds, and the executable audit that enforces all of it. The seven
separate bullets of the reviewed version collapse into one section; the audit output moves
to Appendix D.
`was:` 3.7

### Chapter 4: Methodology

Seven sections, from nineteen. Chapter 4 states what was built and how it would be judged.
Every outcome, verdict and void rung is in Chapter 6.

**4.1 Pre-existing framework, research questions and scope**
Opens with the project-background paragraph: this thesis was conducted within an ongoing
research programme on individualized functional-connectivity abnormality detection; prior
to the start of the thesis, an existing methodological and software framework for fMRI
preprocessing, functional-connectivity analysis, subject-level graph construction and
graph-autoencoder-based modelling was available; the present thesis builds on this
foundation and applies and extends it toward longitudinal MCI-to-AD conversion prediction,
cross-cohort evaluation, and the longitudinal modelling and benchmarking experiments
described below. RQ1 to RQ4 are then stated, with a pointer to where each is answered.
`was:` absent. Provenance item 1.

**4.2 Problem formulation and graph construction**
States first that subject-level functional-connectivity graph modelling was already part of
the pre-existing project framework and that the present analyses use this representation
for the longitudinal experiments. Then the formal problem statement, node features and edge
construction.
`was:` 4.1 and 4.2 merged. Provenance items 6 and 7.
The phrase "rather than an inherited convention" is replaced by: the staged spatial-first
pipeline formed part of the pre-existing modelling framework; in this thesis the
consequences of this design are treated as an explicit hypothesis and tested through
controlled ablations and comparison with the temporal-first alternative.

**4.3 Static representation learning: GAAE and VGAE**
`title:` `Static Representation Learning: GAAE and VGAE`
States that the GAAE stage builds on a graph-attention-autoencoder implementation available
in the pre-existing project codebase and that the configuration and modifications used for
the experiments reported here are what follows. For the pooled GAAE, states that the change
concerns the training cohort and experimental setting rather than the graph-autoencoder
concept. Covers the VGAE variants and the free-bits objective at the level needed to
understand them, with the loss variants and their failure modes in Appendix B. Defines the
reconstruction-fidelity measure (scale-free Pearson correlation between input and
reconstruction) as a method; its value is reported in §6.3.
`backed by:` `DOCS/critical_points/contributions.md`, VGAE and reconstruction-fidelity
entries
`was:` 4.3. Provenance item 8.

**4.4 Spatial-first trajectory modelling**
Opens with the Part A framing: this part builds on the pre-existing spatial-first
graph-learning and graph-autoencoder framework available at the start of the thesis and is
applied here to longitudinal MCI conversion prediction. Covers the GE-LSTM and GE-GRU
recurrent core with gate updates conditioned on the inter-visit interval, the regularisation
scheme, and the four encoder arms that constitute the RQ2 ablation. States that the encoder
handling inherited from the pre-existing implementation was modified during the thesis to
enable the controlled fine-tuning ablation, then describes the gradient-control mechanism.
`backed by:` `DOCS/critical_points/contributions.md`, "GE-LSTM / GE-GRU time-conditioned
trajectory head"
`was:` 4.4, 4.5 and 4.6 merged. Provenance items 5 and 9.

**4.5 Temporal-first modelling: the TFGN architecture**
The hypothesis behind inverting the pooling order, the per-region recurrent core, the
optional graph stage, fusion, readout and the dual-score head. Capacity accounting against
the subject-level denominator is stated here as a design argument. No configuration is
described as winning; selection outcomes are in §6.4.
`backed by:` `DOCS/critical_points/contributions.md`, "Inverting the pooling order" and
"Capacity accounting against the real denominator"; `DOCS/flipped/METHODS.md` §1.1 and §1.4
`was:` 4.7 and 4.8 merged

**4.6 Experimental protocol**
The pre-specified design decisions and the ablation ladder presented as a design: what each
rung tests and what its keep-or-drop criterion is. The four-tier evaluation protocol: floor
gates, the selection rule, robustness checks, and the one-pass held-out estimation, with the
scope table stating which arms were eligible for which tier. Class-imbalance handling by
cost-sensitive weighting, and the threshold policy (best F1, derived out-of-fold, with an
error raised when a caller omits one). Run identifiers and registry entries move to
Appendix C.
`backed by:` `DOCS/flipped/METHODS.md` §1.5 to §1.8;
`DOCS/critical_points/contributions.md`, "Class-imbalance and threshold policy"
`was:` 4.9 to 4.14 and 4.18 merged. Verdicts previously embedded in 4.13 move to §6.4.

**4.7 Interpretability and robustness instrumentation**
What was built to interrogate the model, described as methods only. The disease axis and
per-scan disease score, the orthogonal-residual PCA trajectory space, latent steering back
to connectivity, the per-dimension and per-region Fisher discriminant ratio, mean against
attention pooling, and the four perturbation methods separated by failure mode (feature
noise with the graph held fixed; matrix noise with graph rebuild; edge drop and random-add;
conditioning noise on age and sex). The scanner-drift simulation and its calibration to a
measured heterogeneity inventory, stated as a simulation rather than a validation. The
cohort stability rate as the robustness metric.
`backed by:` `DOCS/critical_points/contributions.md`, latent-space and perturbation entries
`was:` 4.17 partially; the rest was distributed across Chapter 6 in the reviewed version

**Removed from Chapter 4.** Old 4.13's per-rung outcomes, 4.15 ladder verdicts and 4.16 the
cohort-conditioning escalation and its outcome all move to Chapter 6. Old 4.19, the scope
of the time-to-event extension, becomes one paragraph in §8.1 and a future-work item in
§9.2.

### Chapter 5: Baselines

**5.1 The baseline ladder**
Each floor and why it is included: demographics only, the static single-visit baseline,
logistic regression on connectivity drift, spatial-first GE-LSTM with and without a
pretrained encoder, and Brain-TokenGT. States which contrast each pair isolates.
`backed by:` `DOCS/flipped/METHODS.md` §1.7
`was:` 5.1 partially

**5.2 Brain-TokenGT as a published comparator**
The published architecture, its reproduction here, and its behaviour under the evaluated
configuration, including the recurrent core as released and its behaviour when enabled. The
stabilisation configuration is summarised in one paragraph; the knob-level settings move to
Appendix B.
`backed by:` `DOCS/meetings/ninth-meeting/FAIRNESS_AND_LIMITATIONS_COMPARISON.md` §3
`was:` 5.1, 5.2 and 5.3 merged, retitled from "Uncovering Baseline Failure Modes"

**5.3 Run-to-run variability and reproducibility**
The same-seed replicate audit: design, result, and the mechanism (nondeterministic
scatter and gather operations that do not engage deterministic algorithms). Records that the
same behaviour persists under the pooled protocol.
Closing subsection: **"The trajectory implementation used in this thesis is
bit-reproducible"**, stated of the specific model, namely that the GE-LSTM none-arm
implementation evaluated here re-runs byte-identically at a fixed seed.
`was:` 5.4. Provenance item 10, replacing the title "this thesis's own pipeline" and the
phrase "our pipeline re-runs byte-identically".

**5.4 Comparability of the head-to-head**
What can and cannot be equalised: cohort windowing (resolved by the matched-window arm),
capacity and tokenisation overhead (not resolvable), tuning budget asymmetry, and temporal
representation. States the consequences for the statistical comparison, including which
comparisons survive.
`backed by:` `DOCS/meetings/ninth-meeting/FAIRNESS_AND_LIMITATIONS_COMPARISON.md` §2, §4,
§5, §6
`was:` 5.5 and 5.6 merged

### Chapter 6: Experimental Results and Ablations

Ordered by research question. Reports what happened; interpretation is Chapter 8.

**6.1 Reporting rules for this chapter**
Which metric is primary, what the independent unit is, what resolution the sample size
supports, and the statement that every keep-or-drop verdict is a screening heuristic at four
seeds rather than a significance test.
`was:` 6.1

**6.2 RQ1: does longitudinal modelling add information beyond static representations?**
The demographics-only and static single-visit floors, the logistic-on-drift linear floor,
and the sequence-length sensitivity analysis decomposing how performance depends on the
minimum visit count.
`was:` 6.5 and parts of 6.2
`RQ:` 1

**6.3 RQ2: does spatial-first graph representation learning add value?**
The encoder ablation across the four arms, the reconstruction-fidelity measurement, and the
GAAE-against-GEC contrast. Reports the pooling comparison (mean against attention) for the
graph encoder classifier.
`was:` 6.11 and parts of 6.8
`RQ:` 2

**6.4 RQ3: does temporal-first modelling improve over spatial-first modelling?**
The ablation-ladder scorecard, the per-rung verdicts including the rungs that were void for
lack of a gradient path and the rung kept by design independently of AUC, the selected
configuration and the capacity comparison, the matched-window head-to-head against the
short-window comparator, and the direct GE-LSTM against Brain-TokenGT comparison. Reports
the reconciliation of the two delta-AUC statistics.
`backed by:` `DOCS/flipped/METHODS.md` §2.1 to §2.6
`was:` 6.2, 6.6, 6.9, 6.10, 6.12 merged, and receives old 4.13 and 4.15
`RQ:` 3

**6.5 RQ4: cross-cohort transfer**
The one-pass held-out reads, the OASIS-3 transfer result, the per-cohort split between ADNI
and DELCODE for every arm, the cohort-identity probe and its decoding performance against
the pre-specified escalation threshold, and the cohort-conditioning escalation that was run
as specified together with its outcome on both axes it targeted. Reports that the threshold
was exceeded from the first ladder runs and was not read against its own rule until the
final pass, and that in-domain results are out-of-fold quantities unaffected by it.
Reported under the wording constraints in the [claims register](#claims-register): the
probe result is described as consistent with substantial cohort dependence that may
contribute to the poor transfer, without a causal claim.
`backed by:` `DOCS/flipped/METHODS.md` §2.7 and §2.8
`was:` 6.3, 6.4, 6.13, and receives old 4.16
`RQ:` 4

**6.6 Interpretability and latent-space results**
Seed stability of the saliency map and its correlation with the independent model-free drift
measure, the pre-specified network-enrichment test and its outcome, the disease axis and the
per-scan disease score, latent steering decoded back to connectivity, the Fisher
discriminant ratio per latent dimension and per region, and the UMAP projections with
decision overlays. States the reduced statistical power of the map analysis and the
cortical-only limitation of the enrichment statistic.
`backed by:` `DOCS/flipped/METHODS.md` §2.9
`was:` 6.7 and 6.14 merged

**6.7 Robustness and perturbation results**
Results for the four perturbation methods, reported by failure mode, and the cohort
stability rate. The demographic-conditioning result is reported as the direct test of
whether the decision rides on the graph or on age and sex. The scanner-drift simulation is
reported with its stated non-validation caveat.
`was:` distributed across the reviewed Chapter 6

**6.8 Ablations not run**
`was:` 6.15

### Chapter 7: Software and Experiment Infrastructure

**7.1 Registry-driven experiment execution**, including the extensions built for the pooled
ladder. **7.2 Layered architecture and the fail-loudly policy.** **7.3 Testing and
continuous integration.** **7.4 Dual-host GPU dispatch.** **7.5 Interactive research
dashboard.**
`was:` 7.1 to 7.6, with 7.2 and 7.3 merged

### Chapter 8: Discussion and Limitations

**8.1 Scope of the claims**
Opens with the scope statement repeated verbatim. States what the evaluated protocol
licenses and what it does not, including that the time-to-event extension was scoped out.

**8.2 Capacity and inductive bias under sample constraint**
What the parameter-per-subject accounting implies, across the independent comparisons that
point the same way.

**8.3 What self-supervised pretraining contributed**
The reconstruction objective assessed against RQ2, across both the DELCODE-only and the
pooled protocols.
`was:` 8.2

**8.4 Cohort dependence and the limits of transfer**
The central discussion section. Cohort identity is strongly decodable from the pooled
embedding and in-domain performance differs substantially between DELCODE and ADNI, which
together indicate that the learned representation remains strongly cohort-dependent. This
is discussed as consistent with cohort-specific representations contributing to the poor
transfer, without attributing causality, and without concluding anything about which
architecture transfers where. Also discusses what the delayed reading of the escalation
threshold says about the check as a process.
`was:` 8.3 and 8.4 merged

**8.5 Threats to validity**
Cohort size as the dominant constraint, the conversion label as a censored observation,
informative attrition, single-site development, and the limits of a negative result.
`was:` 8.5

**8.6 Clinical relevance**
`was:` 8.6

### Chapter 9: Conclusion and Outlook

**9.1 Concluding remarks**

**9.2 Future work**
Leads with: developing cohort-invariant models based on harmonised clinical target
definitions and individualized deviation from an appropriate reference population remains a
topic for future work. Then isolating the inter-visit-interval contribution on a genuinely
irregular cohort, testing the graph construction rather than only the encoder, replacing the
pretext objective, recovering full-power interpretability statistics, the parcellation
ablation, the time-to-event head, multimodal fusion, and adopting same-seed replication as
protocol.
`was:` 9.2

### Appendices

- **A. Mathematical derivations and loss formulations**
- **B. Implementation details.** The Brain-TokenGT stabilisation configuration; the VGAE
  free-bits objective, its two variants and why the obvious alternatives fail (a plain clamp
  gives no gradient below the floor; a softplus relaxation converged to a more collapsed
  posterior); the edge-perturbation deduplication fix and the tests pinning it; determinism
  and seeding settings, stated per protocol; complete hyperparameter tables.
  `was:` new, absorbing detail from old 4.x, 5.3 and Appendix D
- **C. Reproducibility table.** Experiment identifiers, run identifiers, registry entries and
  code paths, so the main text cites one table instead of interrupting the narrative.
  `was:` new
- **D. Split integrity and overlap verification logs**
- **E. Complete cross-validation metrics**
- **F. Software dependencies and hardware environment**

---

<a id="provenance-table"></a>

## 5. Provenance table

The ten items asking that pre-existing components be made explicit, each with its
destination. Every item is a statement about what existed before the thesis and what was
built during it.

| # | Item | Destination |
|---|---|---|
| 1 | Project-background paragraph on the ongoing research programme and the pre-existing framework | §4.1, opening paragraph of the Methods |
| 2 | ADNI and OASIS-3 included as part of the broader project; processed and integrated during the thesis | §3.3, opening |
| 3 | Pooled protocol builds on pre-existing DELCODE split definitions and leakage-control framework | §3.4, opening |
| 4 | DELCODE preprocessed before the thesis; same framework applied to ADNI and OASIS-3 during it; cohort-specific adaptations are thesis work | §3.5, opening |
| 5 | Part A builds on the pre-existing spatial-first graph-learning and graph-autoencoder framework | §4.4, opening |
| 6 | Replace "rather than an inherited convention" with the explicit-hypothesis wording | §4.2 |
| 7 | Subject-level functional-connectivity graph modelling was already part of the project framework | §4.2, before the construction description |
| 8 | GAAE builds on a pre-existing graph-attention-autoencoder implementation; pooled GAAE changes cohort and setting, not the concept | §4.3, opening |
| 9 | Inherited encoder handling was modified during the thesis to enable the controlled fine-tuning ablation | §4.4, before the gradient-control description |
| 10 | Retitle away from "this thesis's own pipeline"; attribute bit-reproducibility to the GE-LSTM none-arm implementation | §5.3, closing subsection |

---

<a id="claims-register"></a>

## 6. Claims register

Wording rules the prose is written against. The left column collects phrasings present in
the current writeups.

| Do not write | Write instead |
|---|---|
| the cohort probe explains why OASIS-3 fails; the cohort shortcut is the diagnosed cause; the OASIS-3 result is the logical consequence | the strong decodability of cohort identity from the learned representation is consistent with substantial cohort dependence and may contribute to the poor transfer to OASIS-3 |
| external validation failed | cross-cohort transfer to OASIS-3 was at chance under the evaluated protocol |
| TFGN learns a generalisable, AD-specific representation | within the evaluated ADNI and DELCODE protocol, temporal-first modelling performs well |
| each model only generalises to the protocol it was developed on; Brain-TokenGT therefore generalises to OASIS-3 | no such conclusion is drawn; the observed run-to-run variability is too large to support it |
| clean win | under the evaluated configuration, X showed lower held-out performance |
| classic small-sample overfitting signature | the observed validation-to-test decrease is consistent with overfitting in this sample-size regime |
| Brain-TokenGT is unstable | under the evaluated configuration, Brain-TokenGT showed substantially greater run-to-run variability and lower held-out performance |
| pre-registered, pre-registration | pre-specified, or defined prior to running the experiment |

**On the last row.** The ablation specification was committed as `d5e8353` on 2026-08-23,
and the first TFGN run started at 2026-08-24T08:36. Prior specification is therefore
evidenced by a time-stamped commit. That commit lives in an internal repository rather than
an external registry, so "pre-specified" is the accurate term, and the commit is cited as
the evidence for it. Deviations recorded as addenda rather than silent edits are described
as such without invoking registration.

**Highest-risk carrier.** `DOCS/flipped/METHODS.md` is the `backed by:` source for §4.5,
§4.6, §6.4, §6.5 and §6.6. It still contains 31 occurrences of "pre-registered" and
"pre-registration", and the phrasings "a crossover, not a defeat" (a section heading),
"the checkpoint learned something real", "Winning config" and "a clean win over the SOTA
competitor". It is a working research record, so it is left as written; prose drawn from
it must be checked against the table above before it enters a chapter.

**Style rules from `.claude/rules/thesis.md`,** which overlap the tone feedback:

- No em dashes anywhere in the thesis.
- No contrastive "X, not Y" section titles or framing. This rules out the reviewed titles
  "a crossover, not a defeat", "S3 is void, not rejected", "S5 is kept, not rejected",
  "S1c's original run is undecidable, not negative", "a design choice, not a given" and
  "reproducible, but not DMN-specific".
- No editorialising or narrative titles. This rules out "Uncovering Baseline Failure Modes",
  "What Reconstruction Pretraining Actually Bought" and "The Actual Research Setting".
- Every abbreviation is defined once in the acronym table and used through `\ac{}`.
- Prefer fewer, longer subsections over fragmentation.

---

<a id="methods-and-results-separation"></a>

## 7. Methods and results separation

Chapter 4 explains what was done and why. Chapter 6 states what happened. Chapter 8
interprets why. The following constructions appear in the current writeups inside
methodological sections and belong in Chapter 6:

| Construction | Belongs in |
|---|---|
| "the answer is not the assumed one" | §6.3 or §6.4, as the relevant result |
| "the winning configuration", "the winner" | §6.4, after the selection rule has been applied |
| a checkpoint "learned something real" | §6.3, as the reconstruction-fidelity and encoder-ablation result |
| "S3 is void", "S5 is kept", "the scaling gate is closed" | §6.4, as ladder outcomes |
| "the failed escalation" | §6.5, as the escalation outcome |

In Chapter 4 the corresponding rungs are described by what they test and what their
keep-or-drop criterion is, with no outcome stated.

---

<a id="contribution-coverage"></a>

## 8. Contribution coverage

Every entry in `DOCS/critical_points/contributions.md`, with its single home.

| Contribution | Method section | Results section |
|---|---|---|
| Inverting the pooling order (TFGN) | §4.5 | §6.4 |
| Capacity accounting against the subject-level denominator | §4.5 | §6.4, discussed in §8.2 |
| GAAE failing at classification, and the pivot it forced | §4.3 | §6.3 |
| VGAE variants and the free-bits objective | §4.3, detail in App. B | §6.3 |
| Reconstruction fidelity as a scale-free Pearson correlation | §4.3 | §6.3 |
| GE-LSTM and GE-GRU time-conditioned trajectory head | §4.4 | §6.4 |
| Disease axis and per-scan disease score | §4.7 | §6.6 |
| Latent steering decoded back to connectivity | §4.7 | §6.6 |
| Fisher discriminant ratio per dimension and per region | §4.7 | §6.6 |
| Pooling comparison and latent visualisation | §4.7 | §6.3 and §6.6 |
| Four perturbation methods by failure mode | §4.7 | §6.7 |
| Edge-perturbation deduplication fix | App. B | pinned by tests, §7.3 |
| Scanner-drift simulation anchored to a measured inventory | §4.7 | §6.7 |
| Cohort stability rate as the robustness metric | §4.7 | §6.7 |
| Classical statistics before any model was fitted | §3.6 | §3.6 |
| Class-imbalance handling and threshold policy | §4.6 | §6.1 |

---

<a id="challenge-coverage"></a>

## 9. Challenge coverage

Every entry in `DOCS/critical_points/challenges.md`.

| Challenge | Where it is set up | Where it is treated |
|---|---|---|
| Label definition and the follow-up window tradeoff | §3.1 | §6.2, §8.5 |
| Irregular longitudinal sampling and variable-length sequences | §2.4.4, §3.2 | §4.4, §6.2 |
| Longitudinal evaluation protocols are not standardised | §2.4.3 | §4.6, §6.1 |
| Site and protocol heterogeneity across ADNI phases against homogeneous DELCODE | §2.4.4, §3.3 | §6.5, §8.4 |

---

<a id="abbreviations"></a>

## 10. Abbreviations

Each needs an `\acro{KEY}[SHORT]{LONG}` entry in the acronym table of `main.tex` before
first use, and must then be referenced through `\ac{}` rather than typed inline.

AD, MCI, fMRI, FC, ROI, DMN, GNN, GAT, GCN, GAAE, VGAE, GEC, GE-LSTM, GE-GRU, TFGN, LSTM,
GRU, MLP, AUC, ROC, OOF, CV, SSL, PCA, UMAP, FDR, PERMANOVA, NBS, ADNI, DELCODE, OASIS,
CSF, PET, MRI.
