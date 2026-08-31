# Removed from THESIS/chapters/06_results.tex §6.6 (\section{Interpretability and Latent-Space Results})
<!-- Original lines: 616–667 -->
<!-- Content: Saliency-map validation (pooled protocol) with Table 6.15 (tab:rq6-interpretability) -->

```latex
\paragraph{Saliency-map validation (pooled protocol).} The topology score produced by S5's
dual-score head and the model-free drift anchor $\tilde{d}$ are validated against the
pre-specified permutation test on \ac{DMN} overlap and against cross-seed reproducibility,
as defined in \autoref{sec:instrumentation}. \autoref{tab:rq6-interpretability} reports both
statistics for $s_{\text{topo}}$ (from S5, the primary interpretability configuration), for
$\tilde{d}$ (the offline, model-free anchor), and for S2's gate map as a supporting, dropped
comparator.

\begin{table}[htb]
\centering
\footnotesize
\caption{Saliency-map validation on the pooled protocol. \ac{DMN} overlap counts are out of
  30 regions (of 200), with percentile rank against 1{,}000 label-spin permutations and the
  associated $p$-value; cross-seed Spearman is the mean [range] over the four seeds.}
\label{tab:rq6-interpretability}
\begin{tabular}{@{}lccc@{}}
\toprule
Statistic & $s_{\text{topo}}$ (S5) & $\tilde{d}$ (offline) & Gate map (S2) \\
\midrule
\ac{DMN} overlap (count) & 8/30 & 6/30 & 0/30 \\
\ac{DMN} overlap (percentile, $p$) & $77.9$, $p=0.351$ & $41.6$, $p=0.739$ & $0.0$, $p=1.000$ \\
Cross-seed Spearman & $0.928\ [0.898,0.968]$ & $1.000\ [1.000,1.000]$ & $0.823\ [0.676,0.923]$ \\
\bottomrule
\end{tabular}
\end{table}

Neither $s_{\text{topo}}$ nor $\tilde{d}$ clears the pre-specified permutation test on
\ac{DMN} overlap: the claim that the learned topology score preferentially targets
\ac{DMN} and hippocampal regions is not supported on the atlas available for this thesis.
What the data do support is that $s_{\text{topo}}$ is stable across seeds (mean Spearman
0.928, both cohorts individually above 0.91 when computed separately) and correlates with
the independent, model-free drift anchor at $r = 0.456$ ($p = 1.2 \times 10^{-11}$) in a
cross-seed-averaged, median-split quadrant analysis (high-high 66 subjects, high-low 34,
low-high 34, low-low 66). $\tilde{d}$'s cross-seed Spearman of exactly 1.000 follows from
its being a deterministic function of each subject's own data; where a seed selects a
different best fold it drops to 0.633, tracking the subject-set change rather than an
instability in the statistic itself. The dropped gate's map is markedly less stable
(0.823) and shows the same absence of \ac{DMN} enrichment in every fold and both cohorts,
reported for completeness rather than relied upon. Since every configuration above S1 was
dropped by the Tier-2 rule in \autoref{sec:rq3}, this is the entire saliency-map
contribution of the pooled ablation sequence: a negative result on the pre-specified enrichment claim
alongside a positive result on cross-seed reproducibility and correlation with an
independent drift measure.

Two limitations of this validation are properties of the design rather than of the result.
Stability is measured cross-seed rather than cross-fold, because only the best fold's map is
persisted per run, yielding four maps per configuration rather than twenty; this reduces the
statistical power of the reproducibility claim relative to a full cross-fold comparison. The
overlap statistic is restricted to the 46 default-mode regions of the Schaefer-200 cortical
atlas, which carries no subcortical regions, so the hippocampal component of the
pre-specified claim is not assessable with this atlas at all.
```
