# Contributions

## TFGN

### Inverting the pooling order.

Every longitudinal classifier inherited in this project pools a visit's connectome graph
to one vector before the temporal model sees it; TFGN models each of 200 regions' own
trajectory first and pools only afterward. The winning arm — a 68,417-parameter
node-shared LSTM with no graph-propagation stage — beats the matched spatial-first
baseline by 0.186 pooled OOF AUC at 14× fewer parameters, and the pretrained one by
0.030 AUC at 7.6× fewer trainable parameters. The architectural idea and its
implementation are entirely thesis work; the spatial-first stack it is measured against
is the inherited lab framework.

### Capacity accounting against the real denominator

An explicit analysis showing that the effective sample size is subjects, not scans,
because the head fits one binary label per subject. At 133 CV subjects the GELSTM was
carrying 1,806 trainable parameters per subject, with the recurrent core alone larger
than the entire MLP baseline it was compared to. This is the argument that motivated
shrinking the head and testing a GRU.

## GAAE related topics

### GAAE failing at classification

It is currently invisible in the thesis contributions list, and it's the pivot the
whole project turns on. The GAAE-vs-GEC finding — that within-cohort between-patient
variation swamps between-cohort variation, so a reconstruction objective spends capacity
on subject idiosyncrasy — is what moved you from anomaly detection to embed-then-classify.
Everything in Chapter 4 Part A descends from it.

### VGAE

It is a bigger contribution than the thesis currently gives it credit for. Not just
"we tried a VAE" — you built two variants with a non-standard free-bits loss (hard
clamp + quadratic shortfall), and the code comments document why the two obvious
alternatives fail: a plain clamp gives zero gradient below the floor, and a softplus
relaxation converged to a more collapsed posterior with 100% of dims at the floor.
That's a real methods contribution with a negative headline result attached.

### Measuring what the pretext objective actually reconstructs

A single canonical fidelity implementation, shared by both encoder adapters, reporting
a scale-free Pearson r between input and reconstruction because the connectivity
z-score scale is dataset-specific. The GAAE's own reconstruction lands in the "poor"
band, which is a direct, quantitative answer to the question the encoder ablation only
answers indirectly.

### GE-LSTM / GE-GRU time-conditioned trajectory head

A recurrent classifier consuming a sequence of per-visit connectome embeddings with
gate updates conditioned on the inter-visit interval. Designed for sample efficiency
rather than capacity, and it remains the spatial-first comparator that TFGN is tested
against. The GRU variant was added after a capacity analysis, not by preference.

## Latent-space studies

### The disease axis and per-scan disease score

A single linear direction in latent space extracted as the normal to an
L2-regularised logistic decision hyperplane fitted on z-standardised mean-pooled
embeddings. Projecting each scan onto the unit-normalised axis yields a scalar disease
score comparable to a biomarker readout, compressing the classifier's decision function
into one interpretable number per scan.

Residual variance orthogonal to the axis is decomposed by PCA, giving a three-axis space
in which a subject's visit-to-visit trajectory can be drawn against the decision plane,
so progression toward conversion is visible as movement rather than inferred from a
probability.

### Latent steering: decoding the axis back to connectivity

A validation designed to distinguish a meaningful direction from a discriminative
artifact. A probe embedding is traversed along the disease axis in standard-deviation
steps from −3σ to +3σ, each steered point is decoded back to a connectivity matrix, and
the difference maps show which connections the axis actually implies changing. If the
axis were pure classifier geometry, those maps would carry no anatomical structure.

### Fisher discriminant ratio, per latent dimension and per region

A per-dimension separability score computed over the 64-dimensional latent vector, and
the same statistic mapped back onto brain regions. This answers whether conversion
signal is concentrated in a few latent dimensions or smeared across all of them, which
is the practical question behind every claim that the encoder has learned something
disease-relevant.

### Pooling comparison and latent visualisation

Mean pooling versus attention pooling compared head-to-head for the graph encoder
classifier, plus UMAP projections of the latent space with logistic-regression decision
overlays. The pooling comparison is a direct test of a design choice inherited without
justification.

## Perturbation and robustness testing

### Four perturbation methods, each targeting a different failure mode

Not one noise knob but four, separating what could go wrong in a clinical acquisition:
feature noise (Gaussian scaled by the empirical feature spread, graph held fixed);
matrix noise with graph rebuild (the kNN adjacency is recomputed from the corrupted
features, so topology changes as a consequence, the most structurally disruptive case);
edge perturbation (drop and random-add on the topology with features untouched); and
conditioning noise on age and sex, which tests whether the decision is driven by the
graph or by demographics.

The edge method needed a correctness fix that is itself a finding: duplicate edge
columns inflate the dense adjacency above 1 and violate the reconstruction loss's target
range, so deduplication is enforced and pinned by unit tests.

### Scanner-drift simulation anchored to a measured inventory

Before external connectivity matrices existed, cross-cohort degradation was estimated by
perturbing DELCODE graphs to imitate the acquisition drift of ADNI and OASIS-3. The
contribution is the discipline: the perturbation is calibrated to a quantitative
scanner-heterogeneity inventory pulled from those cohorts' own metadata, not to an
assumption like "more vendors means more noise", and the notebooks state explicitly
that this is not a validation claim.

### Cohort stability rate as the reported metric

Robustness is scored not as error drift alone but as the fraction of trials in which the
subject's decision survives, measured against a per-cohort one-vs-rest threshold derived
on validation. Subjects are selected by their margin beyond that threshold, so the test
asks whether the model's most confident calls are also its most stable ones.

Noise studies (D1–D3) are four methods, not one, and they're separated by failure mode
rather than by magnitude — feature noise, noise-plus-graph-rebuild, edge drop/add, and
demographic conditioning noise. The last one is the sharpest: it directly tests whether
the decision rides on the graph or on age and sex. The edge-perturbation dedup fix
(duplicate columns push the dense adjacency above 1 and break the loss's target range)
is a correctness finding with tests pinning it.

## Statistical methods

### Classical statistics before any model was fitted

Converters versus non-converters characterised with PERMANOVA on the connectivity
matrices, the network-based statistic for cluster-level differences, and per-edge effect
sizes with false-discovery-rate control. This establishes whether a group difference
exists at all before a deep model is asked to find one, which is the step most of this
literature skips.

### Class-imbalance and threshold policy

Cost-sensitive weighting was adopted over resampling, on the grounds that under- and
oversampling add no information about the minority class. The decision threshold moved
from a sensitivity-specificity criterion to best F1, always derived out-of-fold, with the
framework raising an error rather than defaulting if a caller omits one.
