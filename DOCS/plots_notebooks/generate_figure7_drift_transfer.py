#!/usr/bin/env python3
"""
generate_figure7_drift_transfer.py

Generates Figure 7 (Figure 6.7 in thesis):
Simulated Scanner Drift vs. Real Empirical Cross-Cohort Transfer.

Compliant with DOCS/rules/plots.md:
- Typography: Courier New monospace, correct pt sizes
- Palette: Okabe-Ito and BeautifulFigures Two-Hue Teal/Purple (colorblind-safe, no raw red)
- Anti-occlusion: zorder >= 10, text halo stroke or white bbox, physical point offsets
- Vector export: PDF (TrueType 42), high-res PNG (600 DPI)
- Dark mode safe: Opaque white card background to ensure perfect contrast in dark PDF
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# MM -> inches conversion
MM = 1 / 25.4

# Master style matching plots.md
plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.2,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

# Output paths
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRS = [
    REPO_ROOT / 'THESIS' / 'figures',
    REPO_ROOT / 'DOCS' / 'results' / 'figures',
]
for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Color definitions (Two-Hue Teal/Purple canonical scheme)
# -------------------------------------------------------------
# Panel A: Proxy vs Real (Teal vs Purple)
C_SIM_FILL   = '#2BA099'  # Two-Hue Teal Medium (Simulated Drift / Proxy)
C_SIM_EDGE   = '#14605B'  # Two-Hue Teal Dark
C_REAL_FILL  = '#873397'  # Two-Hue Purple Medium (Real Empirical Transfer)
C_REAL_EDGE  = '#50165B'  # Two-Hue Purple Dark
C_CHANCE     = '#555555'  # Neutral Dark Gray

# Panel B: Metrics (Teal vs Purple vs Neutral Slate Blue)
C_SPEC_LINE  = '#14605B'  # Teal Dark
C_SPEC_FILL  = '#2BA099'  # Teal Medium (Specificity / Stable)
C_SENS_LINE  = '#50165B'  # Purple Dark
C_SENS_FILL  = '#873397'  # Purple Medium (Sensitivity / Converter)
C_PROB_LINE  = '#0072B2'  # Okabe-Ito Blue (Mean Output Probability)
C_PROB_DARK  = '#1C3E50'

# Create Figure (180 mm x 72 mm -> IEEE double column / thesis textwidth)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), facecolor='white')
fig.patch.set_facecolor('white')

# -------------------------------------------------------------
# Panel A: Cross-Cohort AUC & Transfer Gap
# -------------------------------------------------------------
categories = ['DELCODE\n(In-Domain)', 'ADNI\n(Target)', 'OASIS-3\n(Target)']
x = np.arange(len(categories))
width = 0.28

sim_auc = [0.7929, 0.6821, 0.7000]
real_auc = [0.7929, 0.4971, 0.4892]
real_err = [0.0, 0.0, 0.0224]

# Chance floor reference
ax1.axhline(0.50, color=C_CHANCE, linestyle='--', linewidth=1.0, alpha=0.85, zorder=1, label='Chance Floor (0.50)')

# Bars
rects1 = ax1.bar(
    x - width/2, sim_auc, width,
    label='Simulated Drift (Proxy)',
    color=C_SIM_FILL, edgecolor=C_SIM_EDGE,
    linewidth=0.8, alpha=0.9, zorder=3
)
rects2 = ax1.bar(
    x + width/2, real_auc, width, yerr=real_err,
    capsize=3.5,
    error_kw={'elinewidth': 1.0, 'ecolor': '#222222', 'capthick': 1.0},
    label='Real Empirical Transfer',
    color=C_REAL_FILL, edgecolor=C_REAL_EDGE,
    linewidth=0.8, alpha=0.9, zorder=3
)

ax1.set_ylabel('Area Under ROC (AUC)')
ax1.set_title('A: Cross-Cohort AUC & Transfer Gap', fontsize=8.5, fontweight='bold', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_ylim(0.36, 0.94)
ax1.legend(loc='upper right', fontsize=6.8)
ax1.grid(axis='y', linestyle='-', alpha=0.15, color='#888888', linewidth=0.5, zorder=0)

# Value callouts with white halo stroke
HALO = [pe.withStroke(linewidth=2.5, foreground='white')]

ax1.text(x[0], sim_auc[0] + 0.015, '0.793', ha='center', va='bottom', fontsize=7.0, color='#222222', fontweight='bold', path_effects=HALO, zorder=10)

ax1.text(x[1] - width/2, sim_auc[1] + 0.015, f'{sim_auc[1]:.3f}', ha='center', va='bottom', fontsize=6.8, color=C_SIM_EDGE, fontweight='bold', path_effects=HALO, zorder=10)
ax1.text(x[1] + width/2 + 0.035, real_auc[1] + 0.015, f'{real_auc[1]:.3f}', ha='center', va='bottom', fontsize=6.8, color=C_REAL_EDGE, fontweight='bold', path_effects=HALO, zorder=10)

ax1.text(x[2] - width/2, sim_auc[2] + 0.015, f'{sim_auc[2]:.3f}', ha='center', va='bottom', fontsize=6.8, color=C_SIM_EDGE, fontweight='bold', path_effects=HALO, zorder=10)
ax1.text(x[2] + width/2 + 0.035, real_auc[2] + real_err[2] + 0.018, f'{real_auc[2]:.3f}', ha='center', va='bottom', fontsize=6.8, color=C_REAL_EDGE, fontweight='bold', path_effects=HALO, zorder=10)

# -------------------------------------------------------------
# Panel B: Sensitivity Collapse Mechanism
# -------------------------------------------------------------
conditions = ['Baseline\n(0% noise)', 'ADNI-sim\n(1.37× drift)', 'OASIS-sim\n(1.35× drift)']
x2 = np.arange(len(conditions))
sens = [0.643, 0.000, 0.071]
spec = [0.600, 1.000, 1.000]
mean_prob = [0.511, 0.268, 0.301]

# Reference line & threshold text
ax2.axhline(0.50, color=C_CHANCE, linestyle='--', linewidth=0.9, alpha=0.7, zorder=1)
ax2.text(2.48, 0.52, 'Threshold ≈ 0.50', fontsize=6.6, color='#444444', ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85), zorder=10)

l1 = ax2.plot(x2, spec, marker='s', color=C_SPEC_LINE, linewidth=1.6, markersize=5.5, label='Specificity (Stable Recall)', zorder=4)
l2 = ax2.plot(x2, sens, marker='o', color=C_SENS_FILL, linewidth=1.6, markersize=5.5, label='Sensitivity (Converter Recall)', zorder=4)
l3 = ax2.plot(x2, mean_prob, marker='^', color=C_PROB_LINE, linestyle=':', linewidth=1.4, markersize=5.5, label='Mean Output Probability', zorder=4)

# Data point callout labels with halo
ax2.text(x2[0] - 0.08, sens[0] + 0.02, f'{sens[0]:.3f}', fontsize=6.8, color=C_SENS_LINE, ha='right', va='bottom', fontweight='bold', path_effects=HALO, zorder=10)
ax2.text(x2[0] - 0.08, spec[0] - 0.02, f'{spec[0]:.3f}', fontsize=6.8, color=C_SPEC_LINE, ha='right', va='top', fontweight='bold', path_effects=HALO, zorder=10)
ax2.text(x2[0] - 0.08, mean_prob[0] - 0.06, f'{mean_prob[0]:.3f}', fontsize=6.6, color=C_PROB_LINE, ha='right', va='top', path_effects=HALO, zorder=10)

ax2.text(x2[1], spec[1] - 0.06, f'{spec[1]:.3f}', fontsize=6.8, color=C_SPEC_LINE, ha='center', va='top', fontweight='bold', path_effects=HALO, zorder=10)
ax2.text(x2[1], mean_prob[1] + 0.04, f'{mean_prob[1]:.3f}', fontsize=6.6, color=C_PROB_LINE, ha='center', va='bottom', fontweight='bold', path_effects=HALO, zorder=10)
ax2.text(x2[1], sens[1] - 0.07, f'{sens[1]:.3f}', fontsize=6.8, color=C_SENS_LINE, ha='center', va='top', fontweight='bold', path_effects=HALO, zorder=10)

ax2.text(x2[2], spec[2] - 0.06, f'{spec[2]:.3f}', fontsize=6.8, color=C_SPEC_LINE, ha='center', va='top', fontweight='bold', path_effects=HALO, zorder=10)
ax2.text(x2[2] + 0.08, mean_prob[2] + 0.02, f'{mean_prob[2]:.3f}', fontsize=6.6, color=C_PROB_LINE, ha='left', va='bottom', fontweight='bold', path_effects=HALO, zorder=10)
ax2.text(x2[2] + 0.08, sens[2] + 0.03, f'{sens[2]:.3f}', fontsize=6.8, color=C_SENS_LINE, ha='left', va='bottom', fontweight='bold', path_effects=HALO, zorder=10)

ax2.set_ylabel('Metric Value / Probability')
ax2.set_title('B: Sensitivity Collapse Mechanism', fontsize=8.5, fontweight='bold', loc='left')
ax2.set_xticks(x2)
ax2.set_xticklabels(conditions)
ax2.set_xlim(-0.45, 2.55)
ax2.set_ylim(-0.15, 1.38)
ax2.legend(loc='upper right', fontsize=6.8)
ax2.grid(linestyle='-', alpha=0.15, color='#888888', linewidth=0.5, zorder=0)

for ax in (ax1, ax2):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_color('#222222')
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.7)

for out_dir in OUTPUT_DIRS:
    pdf_path = out_dir / 'fig_scanner_drift_vs_real_transfer.pdf'
    png_path = out_dir / 'fig_scanner_drift_vs_real_transfer.png'
    fig.savefig(pdf_path, facecolor='white', edgecolor='none')
    fig.savefig(png_path, dpi=600, facecolor='white', edgecolor='none')
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')

print('✓ Figure 6.7 generation complete.')
