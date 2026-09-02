# Illustrated FC Graph Construction Pipeline Figure (v3)

Standalone preview and assets of the illustrated functional connectivity graph-construction pipeline figure for `THESIS/chapters/04_methodology.tex` (Section 4.2: Problem Formulation and Graph Construction).

## What is this?

This directory contains:

- **`fig_graph_construction_pipeline_v3.tex`** — The redesigned TikZ flowchart:
  - **Two-tier landscape architecture**: Tier 1 (Stages 1–3) handles fMRI parcellation and continuous correlation; Tier 2A (Stages 4a/4b) models the dual representation ($X$ features and $A$ topology) converging into an explicit **Attributed Graph Merge** ($\mathcal{G}^{(t)}$); Tier 2B (Stage 5) models the longitudinal graph sequence.
  - **Strict content budget**: Title + 1 short sentence + 1 key equation per stage.
  - **Unified styling**: Semantic TUM corporate colors with adaptive `\bg`/`\fg` fills for light and dark themes.
- **`assets/` (and root directory)** — Eight active panel PDFs (vector PDF fonttype 42, transparent backgrounds):
  - `fmri_input.pdf` — Axial synthetic brain slice with overlaid Schaefer-200 parcellation ribbon
  - `regional_timeseries.pdf` — 200 × 180 ROI time series heatmap (clean, uncluttered axes)
  - `fc_matrix.pdf` — 200 × 200 Fisher-$z$ matrix with 7-network modular block structure
  - `node_features.pdf` — Matrix row extraction illustrating $\mathbf{x}_j = C_{j,:}$
  - `sparse_graph.pdf` — 16-node representative network with 4 functional communities and 18 visible edges
  - `graph_snapshot1.pdf`, `graph_snapshot2.pdf`, `graph_snapshot3.pdf` — Longitudinal graph sequence with **fixed node coordinates** across all 3 visits, displaying controlled edge rewiring (orange highlighted connections)
- **`preview.tex`** and **`preview_dark.tex`** — Standalone compilation wrappers replicating the thesis preamble and light/dark theme switch.
- **`build/`** — Compiled PDFs and logs.

## How to rebuild

Activate the project venv and recompile:

```bash
cd /mnt/e/fyassine/ad-early-detection
source .venv/bin/activate
cd DOCS/plots_tkiz/fc_pipeline_diagram

# Regenerate the eight panel PDFs into assets/ and ./
python ../../plots_notebooks/generate_fc_pipeline_assets.py

# Compile light theme preview
tectonic -o build preview.tex

# Compile dark theme preview
tectonic -o build preview_dark.tex
```

## Styling and compliance

All assets comply with `.claude/rules/plots.md`:

- **Typography**: Courier New, 7–8 pt.
- **Output**: PDF vector format (`pdf.fonttype=42`), transparent background.
- **Colours**: TUM corporate palette (`TUMBlue`, `TUMSecondaryBlue2`, `TUMAccentOrange`, `TUMAccentGreen`, `TUMAccentLightBlue`), Okabe-Ito diverging map, no red-green pairs.
- **Content**: All miniature panels are **synthetic schematic illustrations**, as explicitly noted in the figure caption.

## Thesis Integration

Wired in `THESIS/chapters/04_methodology.tex`:

```latex
\begin{figure}[p]
  \centering
  \resizebox{\textwidth}{!}{\input{figures/fig_graph_construction_pipeline_v3}}
  \caption{Subject-level functional connectivity graph construction pipeline ...}
  \label{fig:graph-construction-pipeline}
\end{figure}
```

Assets are synchronized in `THESIS/figures/`.

