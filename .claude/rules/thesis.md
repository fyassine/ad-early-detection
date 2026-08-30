# Thesis writing (`THESIS/`)

Applies only when writing or editing files under `THESIS/`.

## 1. Voice

- Before drafting or editing prose, read `DOCS/writing_style/thesis_updated.pdf` and match
  its sentence length, tone, and level of hedging. Do not fall back to generic
  LaTeX-thesis boilerplate phrasing.
- Register is firm, scientific, academic. No poetic or rhetorical flourish.
- Never use an em dash (`—`).
- Never use contrastive "X, not Y" filler framing. Banned examples:
  - "a design choice, not a given"
  - "that is a decision with measurable consequences, not a platitude"
- Never use argumentative or narrative phrasing that editorializes a result instead of
  reporting it. State what the data shows, in neutral terms, without framing it as a
  verdict or a story beat. Banned examples:
  - "clean win"
  - "classic small-sample overfitting signature"
  - "a crossover, not a defeat"
  - "the answer is not the assumed one"
  - "winning configuration"
  - a checkpoint "learned something real"
- Never use subjective/evaluative adjectives to characterize the scope, quality, or
  completeness of the work — state what was done and let the reader judge. Examples: not
  "comprehensive ablations," just "ablations"; not calling leakage-prevention/split-
  discipline practice "rigorous" (it's the expected standard, not an achievement — use a
  neutral section title instead).
- Chapter/section titles must name what the section actually contains, not a vague or
  overclaiming label (e.g. not "Methodology: Spatiotemporal Graph Modelling" if the
  chapter is really about baselines — call it "Baselines").
- Never say "this thesis's own pipeline" / "our pipeline" when components predate the
  thesis (see §7) — name the specific model/arm instead, e.g. "the GELSTM none-arm
  implementation evaluated here re-runs byte-identically at a fixed seed."

## 2. Verify before stating

Any number, metric, or concept attributed to the code, a config, or a results file must be
checked against that source before it is written down. Never state a figure or claim from
memory or assumption.

## 3. Avoid redundancy

Do not restate the same claim or point in different words within a section or paragraph.

## 4. Prefer longer subsections over fragmentation

Favor fewer, longer `\subsection`s developed across multiple paragraphs over branching
into many small subsections. If a subsection is about to spawn several
`\subsubsection`s, merge them into a single coherent subsection instead, unless each one
is substantial enough to stand as its own independent unit.

## 5. Abbreviations must be defined before use

Every abbreviation is defined once, in the acronym table
(`\begin{acronym}...\end{acronym}` in `THESIS/main.tex`), and referenced in prose with
`\ac{KEY}`, never as a raw abbreviation typed inline.

- Before writing `\ac{KEY}`, check whether `KEY` already has an `\acro{KEY}[SHORT]{LONG}`
  entry in that table.
- If it does not, add the entry to the table first, then use `\ac{KEY}`.
- Never use an abbreviation, via `\ac{}` or typed directly in text, that has no entry in
  the table.
- Exception: chapter/section/subsection titles use the plain short form typed directly
  (e.g. `\section{External Cohorts: ADNI and OASIS-3}`), never `\ac{KEY}`. `\ac`'s
  first-use expansion would otherwise dump the full long form into that heading, and
  from there into the ToC, PDF bookmarks, and running headers — the one place expansion
  is pure noise. This applies only to headings; the abbreviation must still have an
  entry in the acronym table and still get its one `\ac{KEY}` expansion at its first
  use in body prose.

## 6. Compile after every change

After any edit inside `THESIS/`, compile the PDF: from `THESIS/`, run `make pdf`
(`latexmk`, output at `THESIS/build/main.pdf`). If compilation fails, fix the LaTeX error
before handing the change off. Never leave the tree in a non-compiling state.

## 7. Attribute prior work explicitly

This thesis extends a pre-existing methodological/software framework (fMRI
preprocessing, functional-connectivity analysis, subject-level graph construction,
graph-autoencoder modelling, DELCODE split/leakage-control definitions). Every time a
component built on that framework is introduced, state inherited-vs-thesis-original
*before* describing the configuration or modifications applied during the thesis —
don't let a reader infer a pre-existing component was introduced by this work.

- Cohort-inclusion decisions (ADNI, OASIS-3) belong to "the broader project"; the
  specific integration work performed (dataset processing, `build_pooled_assets.py`,
  cohort-specific preprocessing adaptations) belongs to the thesis.
- Never describe an inherited design choice as if it were a thesis decision (e.g. not
  "rather than an inherited convention" for the spatial-first pipeline — say the design
  predates the thesis and is tested here via ablation).

## 8. Causal claims require an isolating experiment

A probe or decodability result (e.g. cohort identity decodable from an embedding) shows
association/representation, not causation. Never say such a result "explains why",
is the "diagnosed cause", or the "logical consequence" of a separate downstream result
(e.g. poor cross-cohort transfer) unless a specific experiment isolates that causal
path. Use hedged language instead: "is consistent with", "may contribute to", "a
plausible/likely explanation".

Keep model claims scoped to the exact evaluated protocol — do not extrapolate a result
obtained under one protocol into a claim of generalisable representation, and do not
extend one model's instability finding into a generalization claim about a different
model.

## 9. Separate Methods, Results, and Discussion

Methods states what was done and why. Results states what happened. Discussion
interprets why. Do not let Methods anticipate an outcome or interpretation — if a
sentence describes an outcome, it belongs in Results; if it explains an outcome, it
belongs in Discussion.

## 10. "Pre-specified" vs. "pre-registered"

Use "pre-specified" or "defined prior to running the experiment" for decisions or
interpretation tables that were fixed internally before running an experiment. Reserve
"pre-registered" strictly for a decision recorded in a time-stamped external registry
before the experiment ran.

## 11. Detail placement

Push exhaustive implementation detail (exact code paths, experiment/run IDs, registry
entries, config variable names, debugging history) out of the main narrative into an
"Implementation Details" subsection or appendix — keep in the main text only what is
necessary to understand or reproduce the method.
