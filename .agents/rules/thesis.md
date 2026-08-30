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

## 6. Compile after every change

After any edit inside `THESIS/`, compile the PDF: from `THESIS/`, run `make pdf`
(`latexmk`, output at `THESIS/build/main.pdf`). If compilation fails, fix the LaTeX error
before handing the change off. Never leave the tree in a non-compiling state.
