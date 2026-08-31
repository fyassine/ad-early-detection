# Removed from THESIS/chapters/06_results.tex §6.4 (RQ3)
<!-- Original lines: 446–564 -->
<!-- Reason: subsection uses a different cohort (DELCODE-only), different subject pool,
     and held-out test AUC — closer to §6.5 RQ4 territory.
     Possible future home: opening of \section{RQ4: Cross-Cohort Transfer} -->

```latex
\paragraph{A direct head-to-head on a matched \ac{DELCODE}-only cohort.} A second, more
tightly matched comparison holds the cohort fixed exactly, using the \ac{DELCODE}-only
windowed cohort characterised in \autoref{fig:cohort-design} (95 cross-validation subjects,
25 held-out test subjects), rather than the pooled protocol's own matched-window
restriction above. This is the head-to-head prepared by \autoref{sec:comparability}: the
frozen pretrained \ac{GELSTM}, restricted to the same two-to-three-visit window, against the
stabilised Brain-TokenGT configuration characterised in \autoref{sec:reproducibility-audit}.

\begin{table}[htb]
\centering
\caption{\ac{DELCODE}-only matched-cohort held-out test \ac{AUC}. \ac{GELSTM} is one run per
  seed; Brain-TokenGT pools eighteen repeat runs across the same four seeds, filtered to a
  single commit, four to six repeats per seed.}
\label{tab:rq3-matched-cohort}
\begin{tabular}{@{}lc@{}}
\toprule
Configuration & Held-out test \ac{AUC} \\
\midrule
\ac{GELSTM}, frozen pretrained, matched window & $0.8782 \pm 0.0256$ \\
Brain-TokenGT, stabilised & $0.6163 \pm 0.0679$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htb]
\centering
\caption{Held-out test \ac{AUC} across nineteen repeat runs of the stabilised
  Brain-TokenGT configuration, four to seven runs per seed with no other setting
  changed.}
\label{tab:btgt-seed-variability}
\begin{tabular}{@{}cccc@{}}
\toprule
Seed & Repeats & Mean test \ac{AUC} & Standard deviation \\
\midrule
42 & 7 & 0.5779 & 0.1167 \\
43 & 4 & 0.6802 & 0.1024 \\
44 & 4 & 0.5227 & 0.0849 \\
45 & 4 & 0.6477 & 0.1002 \\
\bottomrule
\end{tabular}
\end{table}

The mean standard deviation within a seed for Brain-TokenGT, 0.1011, exceeds the standard
deviation of the four seed means, 0.0706 (\autoref{tab:btgt-seed-variability}). Since the
spread among repeats sharing one seed is larger than the spread among the seeds themselves,
individual values obtained by running this configuration once per seed cannot be read as pure
seed effects; they are draws from the within-seed run-to-run scatter. Across all nineteen runs
the test \ac{AUC} averages 0.6025 with a standard deviation of 0.1123, and individual runs at
a shared seed range from 0.357 to 0.708.

\begin{figure}[htb]
  \centering
  \includegraphics[width=\textwidth]{fig5_param_efficiency_frontier}
  \caption{Held-out test \ac{AUC} against trainable parameters (left) and a multi-metric
  comparison profile (right) for the matched-cohort head-to-head of
  \autoref{tab:rq3-matched-cohort}.}
  \label{fig:rq3-param-efficiency}
\end{figure}

The margin is $+0.2619$ \ac{AUC}. Because folds within a seed are not independent, the
independent unit is again the seed, of which four are available; a Wilcoxon signed-rank
test on the seed-paired margin floors at $p=0.125$, the smallest attainable value at this
sample size, so it is reported as underpowered rather than as a significance result, with
the descriptive fact that the frozen \ac{GELSTM} exceeds Brain-TokenGT's seed mean at all
four seeds. A seed-cluster bootstrap 95\% confidence interval on the margin, which
propagates Brain-TokenGT's within-seed repeat variability rather than discarding it, is
$[+0.158, +0.375]$ and excludes zero. \autoref{fig:rq3-cv-test} additionally reports the
underlying cross-validation and held-out distributions this margin is drawn from.

\begin{figure}[htb]
  \centering
  \includegraphics[width=\textwidth]{fig1_cv_and_test_distributions}
  \caption{Cross-validation \ac{AUC} (descriptive only, folds not independent), held-out
  test \ac{AUC} and held-out test F1 for the matched-cohort head-to-head of
  \autoref{tab:rq3-matched-cohort}, with the seed-paired Wilcoxon statistic and the
  bootstrap margin confidence interval annotated.}
  \label{fig:rq3-cv-test}
\end{figure}

\begin{figure}[htb]
  \centering
  \includegraphics[width=0.85\textwidth]{fig2_roc_and_pr_curves}
  \caption{Multi-seed Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves
  for the head-to-head comparison on the matched cohort window across four seeds. Solid lines
  denote mean trajectories, shaded ribbons depict $\pm 1$ standard deviation bands, and the
  dotted line on the PR plot marks baseline converter prevalence ($P = 0.44$).}
  \label{fig:rq3-roc-pr}
\end{figure}

\begin{figure}[htb]
  \centering
  \includegraphics[width=\textwidth]{fig3_confusion_matrices}
  \caption{Multi-seed confusion matrices for the held-out test set ($N=25$ subjects) comparing
  \ac{GELSTM} and Brain-TokenGT across the four random seeds at optimal validation thresholds.}
  \label{fig:rq3-confusion-matrices}
\end{figure}

\begin{figure}[htb]
  \centering
  \includegraphics[width=0.85\textwidth]{fig4_probability_calibration}
  \caption{Probability calibration analysis on the held-out test cohort: reliability diagram
  (left) comparing empirical conversion fraction against predicted conversion probability, and
  Brier score loss (right) across the four seeds.}
  \label{fig:rq3-calibration}
\end{figure}

\autoref{fig:rq3-roc-pr} presents the underlying multi-seed \ac{ROC} and Precision-Recall
trajectories, \autoref{fig:rq3-confusion-matrices} shows the detailed test set confusion
matrices across all four seeds, and \autoref{fig:rq3-calibration} illustrates the corresponding
probability calibration and Brier score losses.

This matched-cohort margin and the pooled-protocol matched-window margin of
\autoref{tab:rq3-w3} answer related but distinct questions over different populations and
different held-out sets, consistent with the convention of \autoref{sec:reporting-rules};
neither supersedes the other, and neither is pooled with the other into a single figure.
Following \autoref{sec:comparability}, no conclusion is drawn from either comparison about
which architecture would transfer to a cohort outside its development set; that question is
addressed directly in \autoref{sec:rq4}.
```
