"""
generate_protocol_funnel_assets.py

Generates publication-ready vector PDF assets for the Four-Tier Evaluation Protocol
Funnel diagram (fig_protocol_funnel.tex) following .agents/rules/plots.md strictly:
- Visual Reference: DOCS/figures_style/image.png
- Typography: Courier New monospace, body 7-8 pt, axes label 8 pt
- Frame & ticks: Inward ticks (xtick.direction='in', ytick.direction='in'), top/right spines OFF
- Colors: Two-hue Teal / Purple (#2ba099 / #873397) + Okabe-Ito Vermillion (#D55E00)
- NO red-green pairings (colorblind-safe)
- Anti-occlusion: TEXT_BBOX / pe.withStroke on all data-overlapping text, zorder >= 10
- Offsets in points: textcoords='offset points'
- Vector output: pdf.fonttype = 42, svg.fonttype = 'none'
- Sizing: Journal sizing in mm (MM = 1 / 25.4)

Assets generated:
1. tier1_floor_gates.pdf       -- Baseline floor hurdles vs candidate model (horizontal bars)
2. tier2_selection_rule.pdf    -- Paired Delta-AUC forest plot (4 seeds + mean with SE threshold)
(Tier 3 plot removed; replaced with native 2-column checklist in TikZ)
(Tier 4 plot removed; represented by native dual-cohort pills and lock badge)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots_tkiz" / "protocol_funnel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4  # mm to inches conversion

# Publication two-hue Teal / Purple palette (.agents/rules/plots.md Part 5 & rq4 notebook)
COLOR_TEAL_FILL = "#2ba099"
COLOR_TEAL_EDGE = "#14605b"
COLOR_PURPLE_FILL = "#873397"
COLOR_PURPLE_EDGE = "#50165b"
COLOR_GRAY_FILL = "#DAD7CB"
COLOR_GRAY_EDGE = "#808080"
COLOR_VERMILLION = "#D55E00"   # Okabe-Ito for threshold/hurdle lines (colorblind-safe)
COLOR_DARK = "#222222"

TEXT_BBOX = dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88)

# Master matplotlib rcParams (.agents/rules/plots.md Part 3)
plt.rcParams.update({
    "font.family": "Courier New",
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": COLOR_DARK,
})


def save(fig, filename: str):
    """Saves figure to PDF with tight bounds and true vector quality."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {path}")


# -------------------------------------------------------------
# 1. tier1_floor_gates.pdf -- Horizontal bar plot of floor hurdles
# -------------------------------------------------------------
def plot_tier1_floor_gates():
    # Exactly matching dimensions: 40 mm x 23 mm
    fig, ax = plt.subplots(figsize=(40 * MM, 23 * MM))
    
    y_pos = [0, 1, 2]
    vals = [0.530, 0.492, 0.749]
    labels = ["Demo", "Static", "S1"]
    colors = [COLOR_GRAY_FILL, COLOR_GRAY_FILL, COLOR_TEAL_FILL]
    edgecolors = [COLOR_GRAY_EDGE, COLOR_GRAY_EDGE, COLOR_TEAL_EDGE]
    height = 0.55
    
    for y, v, c, ec in zip(y_pos, vals, colors, edgecolors):
        ax.barh(y, v, height=height, color=c, edgecolor=ec, linewidth=0.8, zorder=3)
    
    # Value annotations with anti-occlusion rule
    ax.annotate("0.53", xy=(0.53, 0), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=COLOR_DARK, zorder=10)
    ax.annotate("0.49", xy=(0.49, 1), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=COLOR_DARK, zorder=10)
    ax.annotate("0.75", xy=(0.749, 2), xytext=(-5, 0), textcoords="offset points",
                va="center", ha="right", fontsize=6.8, fontweight="bold", color="white", zorder=10)
    
    # Floor hurdle line at 0.530
    ax.axvline(0.530, color=COLOR_VERMILLION, linestyle="--", linewidth=0.85, zorder=2)
    ax.annotate("hurdle (0.53)", xy=(0.530, 2.50), xytext=(3, 0), textcoords="offset points",
                fontsize=6.2, color=COLOR_VERMILLION, fontweight="bold", va="center",
                bbox=TEXT_BBOX, zorder=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.50, 2.80)
    ax.set_xlabel("OOF AUC", labelpad=1)
    ax.set_xticks([0.0, 0.5, 0.75])
    ax.set_xticklabels(["0", ".50", ".75"])
    
    # Declutter: top/right spines off (Principle 3)
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout(pad=0.25)
    save(fig, "tier1_floor_gates.pdf")


# -------------------------------------------------------------
# 2. tier2_selection_rule.pdf -- Forest plot of Delta-AUC
# -------------------------------------------------------------
def plot_tier2_selection_rule():
    # Exactly matching dimensions: 40 mm x 23 mm
    fig, ax = plt.subplots(figsize=(40 * MM, 23 * MM))
    
    seed_deltas = [0.031, 0.008, 0.024, -0.005]
    seed_y = [3, 2, 1, 0]
    
    # Dotted zero reference
    ax.axvline(0, color=COLOR_GRAY_EDGE, linestyle=":", linewidth=0.65, zorder=1)
    
    se = 0.0074
    mean_val = 0.0145
    
    # Shaded 1-SE decision region (mean > 1 SE)
    ax.axvspan(se, 0.045, color=COLOR_TEAL_FILL, alpha=0.10, zorder=0)
    
    # 1-SE threshold line (SE = 0.0074)
    ax.axvline(se, color=COLOR_VERMILLION, linestyle="--", linewidth=0.85, zorder=2)
    ax.annotate("1-SE threshold", xy=(se, 3.85), xytext=(2, 0), textcoords="offset points",
                fontsize=6.0, color=COLOR_VERMILLION, fontweight="bold", va="center",
                bbox=TEXT_BBOX, zorder=10)
    
    # 4 seed points (Teal scatter with darker edge)
    for y, d in zip(seed_y, seed_deltas):
        ax.scatter(d, y, s=18, color=COLOR_TEAL_FILL, edgecolor=COLOR_TEAL_EDGE,
                   linewidth=0.7, zorder=4, alpha=0.9)
    
    # Summary Mean with SE error bar (Purple diamond)
    y_mean = 4.20
    ax.errorbar(
        mean_val, y_mean, xerr=se,
        fmt="D", markersize=4.2, color=COLOR_PURPLE_FILL,
        ecolor=COLOR_PURPLE_FILL, markeredgecolor=COLOR_PURPLE_EDGE,
        elinewidth=1.2, capsize=2.5, capthick=0.8,
        zorder=5
    )

    ax.set_yticks([0, 1, 2, 3, y_mean])
    ax.set_yticklabels(["s4", "s3", "s2", "s1", "Mean"])
    ax.set_xlim(-0.025, 0.045)
    ax.set_ylim(-0.45, 4.75)
    ax.set_xlabel(r"$\Delta\mathrm{AUC}$", labelpad=1)
    ax.set_xticks([-0.02, 0.0, 0.02, 0.04])
    ax.set_xticklabels(["-.02", "0", "+.02", "+.04"])
    
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout(pad=0.25)
    save(fig, "tier2_selection_rule.pdf")


def main():
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    plot_tier1_floor_gates()
    plot_tier2_selection_rule()
    
    # Remove Tier 3 and Tier 4 plots if present
    for fname in ["tier3_robustness_vetoes.pdf", "tier4_held_out.pdf"]:
        for p in [OUTPUT_DIR / fname, assets_dir / fname]:
            if p.exists():
                p.unlink()
                print(f"removed {p}")
            
    import shutil
    for f in [
        "tier1_floor_gates.pdf",
        "tier2_selection_rule.pdf",
    ]:
        src = OUTPUT_DIR / f
        dst = assets_dir / f
        shutil.copy2(src, dst)
    print(f"Copied all active assets to {assets_dir}")


if __name__ == "__main__":
    main()
