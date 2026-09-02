# Illustrated FC Graph Construction Pipeline Figure

Standalone preview of the illustrated functional connectivity graph-construction pipeline figure for THESIS/chapters/04_methodology.tex (Section 4.2: Problem Formulation and Graph Construction).

## What is this?

This directory contains:

- **`fig_graph_construction_pipeline_v2.tex`** — The TikZ flowchart fragment (six-card layout with inset panels). Drop-in replacement for `THESIS/figures/fig_graph_construction_pipeline.tex`, but not yet wired in.
- **Nine panel PDFs** (transparent background, TUM-compliant styling):
  - `fmri_input.pdf` — 3 orthogonal synthetic brain slices
  - `regional_timeseries.pdf` — 200 × 180 ROI timeseries heatmap
  - `fc_matrix.pdf` — 200 × 200 Fisher-z matrix with 7-network block structure
  - `node_features.pdf` — one connectivity row as a feature strip
  - `sparse_graph.pdf` — readable k=3 k-NN graph (48 nodes, visible edges)
  - `graph_snapshot1.pdf`, `graph_snapshot2.pdf`, `graph_snapshot3.pdf` — three sequential graph snapshots for the longitudinal sequence (Stage 5)
  - `subject_outcome.pdf` — Stable MCI (blue) / Converter MCI (orange) outcome cards
- **`preview.tex`** and **`preview_dark.tex`** — Standalone compilation targets that replicate the thesis preamble, TUM colors, and light/dark theme switch, so you can preview the figure without touching `THESIS/`.
- **`build/`** — Compiled PDFs and logs.

## How to rebuild

Activate the project venv and recompile both themes:

```bash
cd /mnt/e/fyassine/ad-early-detection
source .venv/bin/activate
cd DOCS/plots_tkiz/fc_pipeline_diagram

# Regenerate the nine panel PDFs (if you modify generate_fc_pipeline_assets.py)
python ../../plots_notebooks/generate_fc_pipeline_assets.py

# Compile light theme preview
tectonic -o build preview.tex

# Compile dark theme preview
tectonic -o build preview_dark.tex

# View the results
open build/preview.pdf      # light theme
open build/preview_dark.pdf # dark theme
```

## Styling and compliance

All assets follow `.claude/rules/plots.md`:

- **Typography**: Courier New, 8–8.5 pt, matching `DOCS/plots_notebooks/generate_figure7_drift_transfer.py`
- **Output**: PDF only (transparent background, `pdf.fonttype=42` for editable text), no PNG or raster fallbacks for these small glyphs
- **Colours**: TUM corporate palette (from `THESIS/settings.tex`), Okabe-Ito for diverging connectivity maps, no red-green categorical pairs
- **Content**: All panels are **synthetic and schematic illustrations**, not measured subject data. The caption explicitly notes this.

## When you're ready to integrate

Once you approve this preview, wiring it into the thesis is a one-line change in `THESIS/chapters/04_methodology.tex:61`:

```latex
% Current (old version, text-only):
\resizebox{!}{0.76\textheight}{\input{figures/fig_graph_construction_pipeline}}

% New (illustrated):
\resizebox{\textwidth}{!}{\input{figures/fig_graph_construction_pipeline_v2}}
```

Then copy `fig_graph_construction_pipeline_v2.tex` and all nine `.pdf` files into `THESIS/figures/`, and recompile the thesis.

## Notes

- The preview's local acronym table (`preview.tex:62–65`) defines only `ROI`, `MCI`, `BOLD`, and `fMRI`. The full thesis acronym table (`THESIS/main.tex:62–103`) carries the complete project vocabulary.
- Build artifacts go to `build/` only; no outputs clutter the repo root or `THESIS/`.
- The `.gitignore` in this directory (if present) excludes `build/`, `*.aux`, and other LaTeX intermediates, so git status stays clean.
