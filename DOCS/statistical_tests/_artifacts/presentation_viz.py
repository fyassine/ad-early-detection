import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'significant': '#27ae60',
    'not_significant': '#c0392b',
    'warning': '#f39c12',
    'primary': '#2980b9',
    'secondary': '#8e44ad',
    'neutral': '#7f8c8d',
    'light_gray': '#ecf0f1',
    'dark': '#2c3e50',
    'white': '#ffffff'
}

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})


def plot_permanova_summary(permanova_results: dict, group_names: tuple = ('Group A', 'Group B'),
                           figsize: tuple = (14, 5), save_path: str = None):
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(1, 4, width_ratios=[1.2, 1.2, 1, 1.2], wspace=0.4)
    
    f_stat = permanova_results['f_statistic']
    r_squared = permanova_results['r_squared']
    p_value = permanova_results['p_value']
    is_sig = p_value < 0.05
    
    ax1 = fig.add_subplot(gs[0])
    colors_pie = [COLORS['significant'] if is_sig else COLORS['not_significant'], COLORS['light_gray']]
    sizes = [r_squared * 100, (1 - r_squared) * 100]
    
    wedges, texts, autotexts = ax1.pie(
        sizes, colors=colors_pie, autopct='%1.2f%%',
        startangle=90, explode=(0.08, 0),
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    autotexts[1].set_color(COLORS['dark'])
    
    ax1.set_title('Variance Explained\nby Group Membership', fontsize=13, fontweight='bold', pad=10)
    
    legend_patches = [
        mpatches.Patch(color=colors_pie[0], label='Between Groups'),
        mpatches.Patch(color=colors_pie[1], label='Within Groups')
    ]
    ax1.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=2, fontsize=10, frameon=False)
    
    ax2 = fig.add_subplot(gs[1])
    bar_color = COLORS['primary']
    bars = ax2.barh(['F-statistic'], [f_stat], color=bar_color, height=0.4, 
                    edgecolor=COLORS['dark'], linewidth=1.5)
    ax2.set_xlim(0, max(f_stat * 1.5, 2.5))
    ax2.set_title('Test Statistic (F)', fontsize=13, fontweight='bold', pad=10)
    ax2.text(f_stat + f_stat*0.08, 0, f'{f_stat:.3f}', va='center', fontsize=14, fontweight='bold',
             color=COLORS['dark'])
    ax2.set_yticks([])
    ax2.spines['left'].set_visible(False)
    
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    sig_color = COLORS['significant'] if is_sig else COLORS['not_significant']
    sig_text = '✓ SIGNIFICANT' if is_sig else '✗ NOT SIGNIFICANT'
    
    result_box = dict(boxstyle='round,pad=0.6', facecolor=sig_color, alpha=0.2, 
                      edgecolor=sig_color, linewidth=3)
    ax3.text(0.5, 0.6, sig_text, transform=ax3.transAxes, fontsize=15, 
             fontweight='bold', ha='center', va='center', bbox=result_box, color=sig_color)
    
    if is_sig:
        sig_level = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
        ax3.text(0.5, 0.25, f'p < 0.05 {sig_level}', transform=ax3.transAxes, fontsize=12,
                ha='center', va='center', color=COLORS['dark'])
    
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    
    stats_box = dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_gray'], 
                     edgecolor=COLORS['neutral'], linewidth=1)
    stats_text = f"  p-value:  {p_value:.4f}\n  R² value: {r_squared:.4f}\n  F-stat:   {f_stat:.3f}"
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             ha='center', va='center', fontfamily='monospace', bbox=stats_box)
    ax4.set_title('Statistics', fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle(f'PERMANOVA: {group_names[0]} vs {group_names[1]}', 
                 fontsize=16, fontweight='bold', y=1.02, color=COLORS['dark'])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return get_permanova_interpretation(permanova_results, group_names)


def plot_nbs_summary(nbs_results: dict, group_names: tuple = ('Group A', 'Group B'),
                     figsize: tuple = (16, 11), save_path: str = None):
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(2, 3, height_ratios=[1, 1.1], hspace=0.35, wspace=0.35)
    
    test_stats = nbs_results['test_statistics']
    null_dist = nbs_results['null_distribution']
    components = nbs_results['all_components']
    threshold = nbs_results['primary_threshold']
    sig_comps = nbs_results['significant_components']
    has_sig = len(sig_comps) > 0
    
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(test_stats, bins=60, color=COLORS['primary'], 
                                 alpha=0.8, edgecolor='white', linewidth=0.5)
    
    for i, patch in enumerate(patches):
        if bins[i] > threshold or bins[i+1] < -threshold:
            patch.set_facecolor(COLORS['not_significant'])
            patch.set_alpha(0.9)
    
    ax1.axvline(threshold, color=COLORS['not_significant'], linestyle='--', 
                linewidth=2.5, label=f'Threshold: t = ±{threshold}')
    ax1.axvline(-threshold, color=COLORS['not_significant'], linestyle='--', linewidth=2.5)
    
    ymax = ax1.get_ylim()[1]
    ax1.fill_betweenx([0, ymax], threshold, max(test_stats)*1.1, 
                      alpha=0.15, color=COLORS['not_significant'])
    ax1.fill_betweenx([0, ymax], min(test_stats)*1.1, -threshold, 
                      alpha=0.15, color=COLORS['not_significant'])
    
    ax1.set_xlabel('t-statistic', fontsize=12)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('Step 1: Edge-wise Statistics', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    n_supra = np.sum(nbs_results['suprathreshold_edges'])
    ax1.text(0.02, 0.98, f'Suprathreshold: {n_supra} edges', transform=ax1.transAxes,
            fontsize=10, va='top', ha='left', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(null_dist, bins=50, color=COLORS['neutral'], alpha=0.7, 
             edgecolor='white', linewidth=0.5, label='Null Distribution')
    
    if len(components) > 0:
        largest_size = components[0]['size']
        largest_p = components[0].get('p_value', 1)
        obs_color = COLORS['significant'] if largest_p < 0.05 else COLORS['warning']
        ax2.axvline(largest_size, color=obs_color, linewidth=3.5, 
                    label=f'Observed: {largest_size} edges', zorder=10)
        
    pct_95 = np.percentile(null_dist, 95)
    ax2.axvline(pct_95, color=COLORS['not_significant'], linestyle=':', linewidth=2.5, 
                label=f'95th %ile: {pct_95:.0f}')
    
    ax2.set_xlabel('Maximum Component Size', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Step 2: Permutation Test', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    ax3 = fig.add_subplot(gs[0, 2])
    if len(components) > 0:
        n_show = min(len(components), 8)
        sizes = [c['size'] for c in components[:n_show]]
        pvals = [c.get('p_value', 1) for c in components[:n_show]]
        colors = [COLORS['significant'] if p < 0.05 else COLORS['not_significant'] for p in pvals]
        
        bars = ax3.bar(range(n_show), sizes, color=colors, edgecolor=COLORS['dark'], 
                       linewidth=1.2, alpha=0.85)
        ax3.axhline(pct_95, color=COLORS['warning'], linestyle='--', linewidth=2, 
                   label='Significance threshold')
        
        for i, (bar, p, s) in enumerate(zip(bars, pvals, sizes)):
            if p < 0.05:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02, 
                        '★', ha='center', fontsize=16, color=COLORS['significant'])
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{s}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax3.set_xlabel('Component', fontsize=12)
        ax3.set_ylabel('Size (# edges)', fontsize=12)
        ax3.set_title('Step 3: Component Analysis', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax3.set_xticks(range(n_show))
        ax3.set_xticklabels([f'C{i+1}' for i in range(n_show)])
    else:
        ax3.text(0.5, 0.5, 'No components\nfound', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14, color=COLORS['neutral'])
        ax3.set_title('Step 3: Component Analysis', fontsize=13, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, :2])
    
    n_edges = len(test_stats)
    n_suprathreshold = np.sum(nbs_results['suprathreshold_edges'])
    n_sig_edges = sum(c['size'] for c in sig_comps) if sig_comps else 0
    
    categories = ['Total Edges\nTested', 'Suprathreshold\nEdges', 'Significant\nNetwork']
    values = [n_edges, n_suprathreshold, n_sig_edges]
    bar_colors = [COLORS['primary'], COLORS['warning'], COLORS['significant']]
    
    x_pos = np.arange(len(categories)) * 1.5
    bars = ax4.bar(x_pos, values, color=bar_colors, edgecolor=COLORS['dark'], 
                   linewidth=1.5, alpha=0.85, width=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n_edges*0.03,
                f'{val:,}', ha='center', fontsize=13, fontweight='bold', color=COLORS['dark'])
        if i > 0:
            pct = 100 * val / n_edges
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{pct:.1f}%', ha='center', va='center', fontsize=11, 
                    color='white', fontweight='bold')
    
    for i in range(len(categories)-1):
        ax4.annotate('', xy=(x_pos[i+1]-0.3, values[i+1]*0.7), 
                    xytext=(x_pos[i]+0.3, values[i]*0.7),
                    arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=2.5))
    
    ax4.set_ylabel('Number of Edges', fontsize=12)
    ax4.set_title('NBS Analysis Pipeline', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, n_edges * 1.18)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories, fontsize=11)
    ax4.spines['bottom'].set_visible(False)
    ax4.tick_params(axis='x', length=0)
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    result_color = COLORS['significant'] if has_sig else COLORS['not_significant']
    result_text = 'SIGNIFICANT\nNETWORK FOUND' if has_sig else 'NO SIGNIFICANT\nNETWORK'
    result_symbol = '✓' if has_sig else '✗'
    
    result_box = dict(boxstyle='round,pad=0.6', facecolor=result_color, alpha=0.2, 
                      edgecolor=result_color, linewidth=3)
    ax5.text(0.5, 0.78, f'{result_symbol}\n{result_text}', transform=ax5.transAxes, fontsize=14,
             fontweight='bold', ha='center', va='center', bbox=result_box, color=result_color,
             linespacing=1.3)
    
    if has_sig:
        comp = sig_comps[0]
        summary = f"━━━━━ Largest Component ━━━━━\n\n"
        summary += f"  Size:      {comp['size']} edges\n"
        summary += f"  p-value:   {comp['p_value']:.4f}\n"
        summary += f"  Mean |t|:  {comp['mean_stat']:.2f}\n"
        summary += f"  Max |t|:   {comp['max_stat']:.2f}"
    else:
        summary = f"━━━━━ Analysis Summary ━━━━━\n\n"
        summary += f"  Threshold:    t = {threshold}\n"
        summary += f"  Permutations: {nbs_results['n_permutations']:,}\n"
        if len(components) > 0:
            summary += f"  Largest:      {components[0]['size']} edges\n"
            summary += f"  p-value:      {components[0].get('p_value', 1):.4f}"
    
    summary_box = dict(boxstyle='round,pad=0.4', facecolor=COLORS['light_gray'], 
                       edgecolor=COLORS['neutral'], linewidth=1)
    ax5.text(0.5, 0.32, summary, transform=ax5.transAxes, fontsize=10,
             ha='center', va='center', fontfamily='monospace', bbox=summary_box,
             linespacing=1.4)
    
    plt.suptitle(f'Network-Based Statistic (NBS): {group_names[0]} vs {group_names[1]}',
                 fontsize=16, fontweight='bold', y=0.98, color=COLORS['dark'])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return get_nbs_interpretation(nbs_results, group_names)


def plot_diagnostic_summary(diagnostic_results: dict, figsize: tuple = (16, 9), save_path: str = None):
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.35)
    
    pair_name = diagnostic_results['pair_name']
    n_a, n_b = diagnostic_results['n_a'], diagnostic_results['n_b']
    min_n = diagnostic_results['min_sample_size']
    pvals = diagnostic_results['pvals']
    cohens_d = diagnostic_results['cohens_d']
    sig_fdr = diagnostic_results['sig_fdr_count']
    n_edges = len(pvals)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    if min_n < 20:
        status, status_color = 'LOW', COLORS['not_significant']
    elif min_n < 30:
        status, status_color = 'MODERATE', COLORS['warning']
    else:
        status, status_color = 'ADEQUATE', COLORS['significant']
    
    bar_colors = [COLORS['primary'], COLORS['secondary']]
    bars = ax1.bar([f'{pair_name.split(" vs ")[0]}\n(Group A)', 
                    f'{pair_name.split(" vs ")[1]}\n(Group B)'], 
                   [n_a, n_b], color=bar_colors, edgecolor=COLORS['dark'], 
                   linewidth=1.5, alpha=0.85)
    
    for bar, val in zip(bars, [n_a, n_b]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_a, n_b)*0.03,
                f'n = {val}', ha='center', fontsize=12, fontweight='bold')
    
    status_box = dict(boxstyle='round,pad=0.3', facecolor=status_color, alpha=0.2)
    ax1.text(0.98, 0.98, status, transform=ax1.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='top', color=status_color, bbox=status_box)
    
    ax1.set_ylabel('Sample Size', fontsize=12)
    ax1.set_title('Sample Sizes', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(n_a, n_b) * 1.15)
    
    ax2 = fig.add_subplot(gs[0, 1])
    n_bins = 40
    n, bins, patches = ax2.hist(pvals, bins=n_bins, color=COLORS['primary'], 
                                 alpha=0.8, edgecolor='white', linewidth=0.5)
    
    expected_per_bin = len(pvals) / n_bins
    ax2.axhline(expected_per_bin, color=COLORS['not_significant'], linestyle='--', 
                linewidth=2.5, label='Expected under null', zorder=5)
    
    for i, patch in enumerate(patches):
        if bins[i] < 0.05:
            patch.set_facecolor(COLORS['significant'])
    
    ax2.fill_betweenx([0, ax2.get_ylim()[1]*1.5], 0, 0.05, 
                      alpha=0.1, color=COLORS['significant'])
    ax2.axvline(0.05, color=COLORS['significant'], linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('P-value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('P-value Distribution', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.set_xlim(0, 1)
    
    excess = np.sum(pvals < 0.05) - (0.05 * len(pvals))
    if excess > 0:
        ax2.text(0.02, 0.98, f'Excess at p<0.05: {int(excess)}', transform=ax2.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    effect_bins = np.linspace(min(cohens_d), max(cohens_d), 50)
    n, bins, patches = ax3.hist(cohens_d, bins=effect_bins, color=COLORS['primary'], 
                                 alpha=0.8, edgecolor='white', linewidth=0.5)
    
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if abs(bin_center) > 0.8:
            patch.set_facecolor(COLORS['not_significant'])
        elif abs(bin_center) > 0.5:
            patch.set_facecolor(COLORS['warning'])
        else:
            patch.set_facecolor(COLORS['primary'])
    
    ax3.axvline(0, color=COLORS['dark'], linestyle='-', linewidth=1.5, alpha=0.5)
    for thresh, style in [(0.5, '--'), (0.8, ':')]:
        ax3.axvline(thresh, color=COLORS['neutral'], linestyle=style, linewidth=1.5, alpha=0.8)
        ax3.axvline(-thresh, color=COLORS['neutral'], linestyle=style, linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel("Cohen's d", fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Effect Size Distribution', fontsize=13, fontweight='bold')
    
    legend_patches = [
        mpatches.Patch(color=COLORS['primary'], label='Small (|d| < 0.5)'),
        mpatches.Patch(color=COLORS['warning'], label='Medium (0.5-0.8)'),
        mpatches.Patch(color=COLORS['not_significant'], label='Large (|d| > 0.8)')
    ]
    ax3.legend(handles=legend_patches, loc='upper right', fontsize=9, framealpha=0.95)
    
    ax4 = fig.add_subplot(gs[1, 0])
    
    sig_unc = diagnostic_results['sig_uncorrected']
    not_sig = n_edges - sig_unc
    fdr_rejected = sig_unc - sig_fdr
    
    sizes = [not_sig, fdr_rejected, sig_fdr]
    colors_pie = [COLORS['light_gray'], COLORS['warning'], COLORS['significant']]
    labels = ['Not Significant', 'FDR Rejected', 'FDR Significant']
    explode = (0, 0, 0.1)
    
    wedges, texts, autotexts = ax4.pie(
        sizes, colors=colors_pie, autopct=lambda p: f'{p:.1f}%' if p > 0.5 else '',
        startangle=90, explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    ax4.set_title('Multiple Testing Correction', fontsize=13, fontweight='bold')
    ax4.legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=9, frameon=False)
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    abs_d = np.abs(cohens_d)
    effect_counts = [
        np.sum(abs_d <= 0.5),
        np.sum((abs_d > 0.5) & (abs_d <= 0.8)),
        np.sum(abs_d > 0.8)
    ]
    effect_labels = ['Small\n(|d| < 0.5)', 'Medium\n(0.5 - 0.8)', 'Large\n(|d| > 0.8)']
    effect_colors = [COLORS['primary'], COLORS['warning'], COLORS['not_significant']]
    
    bars = ax5.bar(effect_labels, effect_counts, color=effect_colors, 
                   edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
    
    for bar, val in zip(bars, effect_counts):
        pct = 100 * val / len(cohens_d)
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(effect_counts)*0.02,
                f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{val:,}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    ax5.set_ylabel('Number of Edges', fontsize=12)
    ax5.set_title('Effect Size Categories', fontsize=13, fontweight='bold')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    sig_fdr_pct = diagnostic_results['sig_fdr_pct']
    med_effect = diagnostic_results['median_effect_size']
    
    if sig_fdr > 0 and min_n >= 20:
        overall_status = 'STRONG SIGNAL'
        overall_color = COLORS['significant']
    elif sig_fdr > 0 or (diagnostic_results['sig_uncorrected'] > 0.05 * n_edges * 1.5):
        overall_status = 'WEAK SIGNAL'
        overall_color = COLORS['warning']
    else:
        overall_status = 'NO SIGNAL'
        overall_color = COLORS['not_significant']
    
    result_box = dict(boxstyle='round,pad=0.6', facecolor=overall_color, alpha=0.2, 
                      edgecolor=overall_color, linewidth=3)
    ax6.text(0.5, 0.82, overall_status, transform=ax6.transAxes, fontsize=16,
             fontweight='bold', ha='center', va='center', bbox=result_box, color=overall_color)
    
    summary = f"━━━━ Summary Statistics ━━━━\n\n"
    summary += f"  Total edges:     {n_edges:,}\n"
    summary += f"  FDR significant: {sig_fdr} ({sig_fdr_pct:.2f}%)\n"
    summary += f"  Median |d|:      {med_effect:.3f}\n"
    summary += f"  Medium effects:  {diagnostic_results['pct_medium_effect']:.1f}%\n"
    summary += f"  Large effects:   {diagnostic_results['pct_large_effect']:.1f}%"
    
    summary_box = dict(boxstyle='round,pad=0.4', facecolor=COLORS['light_gray'], 
                       edgecolor=COLORS['neutral'], linewidth=1)
    ax6.text(0.5, 0.38, summary, transform=ax6.transAxes, fontsize=10,
             ha='center', va='center', fontfamily='monospace', bbox=summary_box, linespacing=1.4)
    
    plt.suptitle(f'Diagnostic Analysis: {pair_name}', fontsize=16, fontweight='bold', 
                 y=0.98, color=COLORS['dark'])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return get_diagnostic_interpretation(diagnostic_results)


def plot_combined_results(permanova_results: dict, nbs_results: dict, 
                          group_names: tuple = ('Group A', 'Group B'),
                          figsize: tuple = (14, 6), save_path: str = None):
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.3)
    
    permanova_sig = permanova_results['p_value'] < 0.05
    nbs_sig = len(nbs_results['significant_components']) > 0
    
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    
    p_color = COLORS['significant'] if permanova_sig else COLORS['not_significant']
    p_symbol = '✓' if permanova_sig else '✗'
    
    ax1.add_patch(plt.Circle((0.5, 0.55), 0.25, facecolor=p_color, alpha=0.2, 
                              edgecolor=p_color, linewidth=3, transform=ax1.transAxes))
    ax1.text(0.5, 0.55, p_symbol, transform=ax1.transAxes, fontsize=50, 
             ha='center', va='center', color=p_color, fontweight='bold')
    ax1.text(0.5, 0.9, 'PERMANOVA', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold', ha='center', color=COLORS['dark'])
    ax1.text(0.5, 0.18, f'p = {permanova_results["p_value"]:.4f}\nR² = {permanova_results["r_squared"]:.4f}',
             transform=ax1.transAxes, fontsize=11, ha='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_gray'], edgecolor=COLORS['neutral']))
    
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    n_color = COLORS['significant'] if nbs_sig else COLORS['not_significant']
    n_symbol = '✓' if nbs_sig else '✗'
    
    ax2.add_patch(plt.Circle((0.5, 0.55), 0.25, facecolor=n_color, alpha=0.2, 
                              edgecolor=n_color, linewidth=3, transform=ax2.transAxes))
    ax2.text(0.5, 0.55, n_symbol, transform=ax2.transAxes, fontsize=50, 
             ha='center', va='center', color=n_color, fontweight='bold')
    ax2.text(0.5, 0.9, 'NBS', transform=ax2.transAxes, fontsize=14, 
             fontweight='bold', ha='center', color=COLORS['dark'])
    
    if nbs_sig:
        comp = nbs_results['significant_components'][0]
        nbs_info = f'p = {comp["p_value"]:.4f}\n{comp["size"]} edges'
    else:
        nbs_info = 'No significant\nnetwork'
    ax2.text(0.5, 0.18, nbs_info, transform=ax2.transAxes, fontsize=11, ha='center', 
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_gray'], edgecolor=COLORS['neutral']))
    
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    if permanova_sig and nbs_sig:
        interp = 'CLEAR NETWORK\nDIFFERENCES'
        interp_color = COLORS['significant']
        detail = 'Groups show significant differences\nin connected brain networks.\nStrong evidence for distinct connectivity patterns.'
    elif permanova_sig and not nbs_sig:
        interp = 'DIFFUSE\nDIFFERENCES'
        interp_color = COLORS['warning']
        detail = 'Groups differ overall, but differences\nare not localized to specific networks.\nChanges may be distributed across many edges.'
    elif not permanova_sig and nbs_sig:
        interp = 'VERIFY\nRESULTS'
        interp_color = COLORS['warning']
        detail = 'Unexpected pattern detected.\nNBS significant without PERMANOVA.\nConsider re-running analysis.'
    else:
        interp = 'NO GROUP\nDIFFERENCES'
        interp_color = COLORS['not_significant']
        detail = 'No significant connectivity differences\ndetected between groups.\nGroups appear similar in brain connectivity.'
    
    result_box = dict(boxstyle='round,pad=0.6', facecolor=interp_color, alpha=0.2, 
                      edgecolor=interp_color, linewidth=3)
    ax3.text(0.5, 0.72, interp, transform=ax3.transAxes, fontsize=15,
             fontweight='bold', ha='center', va='center', bbox=result_box, 
             color=interp_color, linespacing=1.3)
    
    detail_box = dict(boxstyle='round,pad=0.4', facecolor=COLORS['light_gray'], 
                      edgecolor=COLORS['neutral'], linewidth=1)
    ax3.text(0.5, 0.28, detail, transform=ax3.transAxes, fontsize=10, ha='center',
             va='center', bbox=detail_box, linespacing=1.5)
    
    plt.suptitle(f'Combined Analysis: {group_names[0]} vs {group_names[1]}',
                 fontsize=16, fontweight='bold', y=0.98, color=COLORS['dark'])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return get_combined_interpretation(permanova_results, nbs_results, group_names)


def plot_effect_size_volcano(diagnostic_results: dict, figsize: tuple = (11, 9), 
                             fdr_threshold: float = 0.05, save_path: str = None):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    cohens_d = diagnostic_results['cohens_d']
    pvals_fdr = diagnostic_results['pvals_fdr']
    
    neg_log_p = -np.log10(pvals_fdr + 1e-10)
    
    sig_large = (pvals_fdr < fdr_threshold) & (np.abs(cohens_d) > 0.5)
    sig_small = (pvals_fdr < fdr_threshold) & (np.abs(cohens_d) <= 0.5)
    ns_large = (pvals_fdr >= fdr_threshold) & (np.abs(cohens_d) > 0.5)
    ns_small = ~sig_large & ~sig_small & ~ns_large
    
    ax.scatter(cohens_d[ns_small], neg_log_p[ns_small], c=COLORS['light_gray'], 
               alpha=0.4, s=25, edgecolors='none', label=f'N.S. + Small ({np.sum(ns_small):,})', zorder=1)
    ax.scatter(cohens_d[ns_large], neg_log_p[ns_large], c=COLORS['primary'], 
               alpha=0.6, s=35, edgecolors='white', linewidths=0.5, 
               label=f'N.S. + Large effect ({np.sum(ns_large)})', zorder=2)
    ax.scatter(cohens_d[sig_small], neg_log_p[sig_small], c=COLORS['warning'], 
               alpha=0.7, s=40, edgecolors='white', linewidths=0.5, 
               label=f'Sig. + Small effect ({np.sum(sig_small)})', zorder=3)
    ax.scatter(cohens_d[sig_large], neg_log_p[sig_large], c=COLORS['not_significant'], 
               alpha=0.8, s=50, edgecolors='white', linewidths=0.8, 
               label=f'Sig. + Large effect ({np.sum(sig_large)})', zorder=4)
    
    sig_line = -np.log10(fdr_threshold)
    ax.axhline(sig_line, color=COLORS['not_significant'], linestyle='--', linewidth=2,
               label=f'FDR = {fdr_threshold}', alpha=0.8)
    
    for thresh in [0.5, -0.5]:
        ax.axvline(thresh, color=COLORS['primary'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(0, color=COLORS['dark'], linestyle='-', linewidth=1, alpha=0.3)
    
    ax.text(0.52, sig_line + 0.05, 'FDR threshold', fontsize=9, color=COLORS['not_significant'])
    ax.text(0.52, ax.get_ylim()[1]*0.95, '|d| = 0.5', fontsize=9, color=COLORS['primary'])
    
    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=13)
    ax.set_ylabel('-log₁₀(FDR-corrected p-value)', fontsize=13)
    ax.set_title(f'Volcano Plot: {diagnostic_results["pair_name"]}', fontsize=15, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, title='Categories', title_fontsize=11)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return get_volcano_interpretation(diagnostic_results, fdr_threshold)


def plot_connectivity_heatmap(nbs_results: dict, n_nodes: int = None, 
                              figsize: tuple = (14, 6), save_path: str = None):
    test_stats = nbs_results['test_statistics']
    n_edges = len(test_stats)
    
    if n_nodes is None:
        n_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    
    matrix = np.zeros((n_nodes, n_nodes))
    idx = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if idx < n_edges:
                matrix[i, j] = test_stats[idx]
                matrix[j, i] = test_stats[idx]
            idx += 1
    
    sig_comps = nbs_results['significant_components']
    sig_edges = sig_comps[0]['edges'] if len(sig_comps) > 0 else []
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmax = np.percentile(np.abs(test_stats), 99)
    
    ax1 = axes[0]
    im1 = ax1.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    ax1.set_title('All Edges (t-statistics)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Brain Region', fontsize=11)
    ax1.set_ylabel('Brain Region', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, label='t-statistic')
    cbar1.ax.tick_params(labelsize=9)
    
    ax2 = axes[1]
    sig_matrix = np.zeros((n_nodes, n_nodes))
    
    edge_to_nodes = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_to_nodes.append((i, j))
    
    for edge_idx in sig_edges:
        if edge_idx < len(edge_to_nodes):
            i, j = edge_to_nodes[edge_idx]
            sig_matrix[i, j] = test_stats[edge_idx]
            sig_matrix[j, i] = test_stats[edge_idx]
    
    im2 = ax2.imshow(sig_matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    ax2.set_title('Significant Network Only', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Brain Region', fontsize=11)
    ax2.set_ylabel('Brain Region', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, label='t-statistic')
    cbar2.ax.tick_params(labelsize=9)
    
    n_sig = len(sig_edges)
    ax2.text(0.02, 0.98, f'{n_sig} significant edges', transform=ax2.transAxes,
            fontsize=10, va='top', ha='left', color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['significant'], alpha=0.8))
    
    plt.suptitle('Connectivity Difference Matrices', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return get_heatmap_interpretation(nbs_results)


def get_permanova_interpretation(results: dict, group_names: tuple) -> str:
    p = results['p_value']
    r2 = results['r_squared']
    f = results['f_statistic']
    
    interp = f"\n{'='*70}\n"
    interp += "📊 PERMANOVA INTERPRETATION\n"
    interp += f"{'='*70}\n\n"
    
    if p < 0.001:
        interp += f"✓ HIGHLY SIGNIFICANT (p < 0.001)\n\n"
    elif p < 0.01:
        interp += f"✓ VERY SIGNIFICANT (p < 0.01)\n\n"
    elif p < 0.05:
        interp += f"✓ SIGNIFICANT (p < 0.05)\n\n"
    else:
        interp += f"✗ NOT SIGNIFICANT (p ≥ 0.05)\n\n"
    
    interp += f"The PERMANOVA test compares the overall connectivity patterns between\n"
    interp += f"{group_names[0]} and {group_names[1]}.\n\n"
    
    interp += f"• R² = {r2:.4f} ({r2*100:.2f}%): This means {r2*100:.2f}% of the total\n"
    interp += f"  variance in connectivity is explained by group membership.\n\n"
    
    if r2 < 0.01:
        interp += f"  → Effect size: NEGLIGIBLE (R² < 1%)\n"
    elif r2 < 0.06:
        interp += f"  → Effect size: SMALL (R² = 1-6%)\n"
    elif r2 < 0.14:
        interp += f"  → Effect size: MEDIUM (R² = 6-14%)\n"
    else:
        interp += f"  → Effect size: LARGE (R² ≥ 14%)\n"
    
    interp += f"\n• F-statistic = {f:.3f}: Higher values indicate greater between-group\n"
    interp += f"  differences relative to within-group variation.\n"
    
    if p < 0.05:
        interp += f"\n✓ CONCLUSION: Groups have significantly different connectivity patterns.\n"
    else:
        interp += f"\n✗ CONCLUSION: No significant difference in overall connectivity patterns.\n"
    
    interp += f"{'='*70}\n"
    
    print(interp)
    return interp


def get_nbs_interpretation(results: dict, group_names: tuple) -> str:
    sig_comps = results['significant_components']
    all_comps = results['all_components']
    threshold = results['primary_threshold']
    n_perms = results['n_permutations']
    
    interp = f"\n{'='*70}\n"
    interp += "🧠 NBS (NETWORK-BASED STATISTIC) INTERPRETATION\n"
    interp += f"{'='*70}\n\n"
    
    interp += f"The NBS identifies connected subnetworks that differ between groups.\n"
    interp += f"Unlike edge-by-edge testing, NBS has more power to detect distributed\n"
    interp += f"network effects.\n\n"
    
    interp += f"Parameters used:\n"
    interp += f"  • Primary threshold: t = {threshold}\n"
    interp += f"  • Permutations: {n_perms:,}\n\n"
    
    if len(sig_comps) > 0:
        interp += f"✓ SIGNIFICANT NETWORK FOUND\n\n"
        comp = sig_comps[0]
        interp += f"Largest significant component:\n"
        interp += f"  • Size: {comp['size']} edges (connections)\n"
        interp += f"  • p-value: {comp['p_value']:.4f}\n"
        interp += f"  • Mean |t|: {comp['mean_stat']:.2f}\n"
        interp += f"  • Max |t|: {comp['max_stat']:.2f}\n\n"
        interp += f"This means {comp['size']} brain connections form a connected network\n"
        interp += f"that significantly differs between {group_names[0]} and {group_names[1]}.\n"
    else:
        interp += f"✗ NO SIGNIFICANT NETWORK FOUND\n\n"
        if len(all_comps) > 0:
            interp += f"Largest component found: {all_comps[0]['size']} edges\n"
            interp += f"p-value: {all_comps[0].get('p_value', 'N/A')}\n\n"
        interp += f"No connected subnetwork shows significant group differences.\n"
        interp += f"Consider:\n"
        interp += f"  • Lowering the threshold (currently t = {threshold})\n"
        interp += f"  • Differences may be too diffuse to form networks\n"
    
    interp += f"\n{'='*70}\n"
    
    print(interp)
    return interp


def get_diagnostic_interpretation(results: dict) -> str:
    interp = f"\n{'='*70}\n"
    interp += "📈 DIAGNOSTIC ANALYSIS INTERPRETATION\n"
    interp += f"{'='*70}\n\n"
    
    interp += f"Comparison: {results['pair_name']}\n\n"
    
    interp += f"1. SAMPLE SIZE\n"
    interp += f"   • Group A: n = {results['n_a']}, Group B: n = {results['n_b']}\n"
    min_n = results['min_sample_size']
    if min_n < 20:
        interp += f"   ⚠ LOW sample size - results may be unreliable\n"
    elif min_n < 30:
        interp += f"   ~ MODERATE sample size - interpret with caution\n"
    else:
        interp += f"   ✓ ADEQUATE sample size for analysis\n"
    
    interp += f"\n2. STATISTICAL SIGNIFICANCE\n"
    interp += f"   • Uncorrected significant: {results['sig_uncorrected']} edges\n"
    interp += f"   • FDR-corrected significant: {results['sig_fdr_count']} edges ({results['sig_fdr_pct']:.2f}%)\n"
    
    if results['sig_fdr_count'] > 0:
        interp += f"   ✓ Signal detected after multiple testing correction\n"
    else:
        interp += f"   ✗ No edges survive FDR correction\n"
    
    interp += f"\n3. EFFECT SIZES\n"
    interp += f"   • Median |Cohen's d|: {results['median_effect_size']:.3f}\n"
    interp += f"   • Medium effects (0.5-0.8): {results['pct_medium_effect']:.1f}%\n"
    interp += f"   • Large effects (>0.8): {results['pct_large_effect']:.1f}%\n"
    
    if results['median_effect_size'] < 0.2:
        interp += f"   → Most effects are NEGLIGIBLE\n"
    elif results['median_effect_size'] < 0.5:
        interp += f"   → Most effects are SMALL\n"
    elif results['median_effect_size'] < 0.8:
        interp += f"   → Most effects are MEDIUM\n"
    else:
        interp += f"   → Most effects are LARGE\n"
    
    interp += f"\n{'='*70}\n"
    
    print(interp)
    return interp


def get_combined_interpretation(permanova: dict, nbs: dict, group_names: tuple) -> str:
    p_sig = permanova['p_value'] < 0.05
    n_sig = len(nbs['significant_components']) > 0
    
    interp = f"\n{'='*70}\n"
    interp += "🔬 COMBINED ANALYSIS INTERPRETATION\n"
    interp += f"{'='*70}\n\n"
    
    interp += f"Comparing: {group_names[0]} vs {group_names[1]}\n\n"
    
    interp += f"┌─────────────────┬──────────────┬─────────────────────────┐\n"
    interp += f"│  PERMANOVA      │     NBS      │    Interpretation       │\n"
    interp += f"├─────────────────┼──────────────┼─────────────────────────┤\n"
    interp += f"│  Significant    │  Significant │ ✓ Clear network effects │\n"
    interp += f"│  Significant    │  Not sig.    │ ~ Diffuse differences   │\n"
    interp += f"│  Not sig.       │  Significant │ ⚠ Recheck (unusual)     │\n"
    interp += f"│  Not sig.       │  Not sig.    │ ✗ No group differences  │\n"
    interp += f"└─────────────────┴──────────────┴─────────────────────────┘\n\n"
    
    interp += f"Your results:\n"
    interp += f"  • PERMANOVA: {'✓ SIGNIFICANT' if p_sig else '✗ NOT SIGNIFICANT'} (p = {permanova['p_value']:.4f})\n"
    interp += f"  • NBS: {'✓ SIGNIFICANT' if n_sig else '✗ NOT SIGNIFICANT'}"
    if n_sig:
        interp += f" ({nbs['significant_components'][0]['size']} edges, p = {nbs['significant_components'][0]['p_value']:.4f})\n"
    else:
        interp += "\n"
    
    interp += f"\n"
    
    if p_sig and n_sig:
        interp += f"✓ CONCLUSION: Clear evidence of network-level differences between groups.\n"
        interp += f"  The groups differ in their overall connectivity AND these differences\n"
        interp += f"  are localized to identifiable connected brain networks.\n"
    elif p_sig and not n_sig:
        interp += f"~ CONCLUSION: Groups differ overall but differences are diffuse.\n"
        interp += f"  Connectivity differences exist but are spread across many edges\n"
        interp += f"  rather than forming coherent networks. Consider using FDR-corrected\n"
        interp += f"  edge-wise tests to identify specific differences.\n"
    elif not p_sig and n_sig:
        interp += f"⚠ CONCLUSION: Unusual pattern - verify analysis.\n"
        interp += f"  NBS found significant networks but PERMANOVA did not detect\n"
        interp += f"  overall differences. This is uncommon and may indicate:\n"
        interp += f"  - Very localized effects\n"
        interp += f"  - Need for threshold adjustment\n"
    else:
        interp += f"✗ CONCLUSION: No evidence of connectivity differences between groups.\n"
        interp += f"  Neither overall connectivity nor specific networks differ significantly.\n"
    
    interp += f"\n{'='*70}\n"
    
    print(interp)
    return interp


def get_volcano_interpretation(results: dict, fdr_threshold: float) -> str:
    cohens_d = results['cohens_d']
    pvals_fdr = results['pvals_fdr']
    
    sig_large = np.sum((pvals_fdr < fdr_threshold) & (np.abs(cohens_d) > 0.5))
    sig_small = np.sum((pvals_fdr < fdr_threshold) & (np.abs(cohens_d) <= 0.5))
    ns_large = np.sum((pvals_fdr >= fdr_threshold) & (np.abs(cohens_d) > 0.5))
    
    interp = f"\n{'='*70}\n"
    interp += "🌋 VOLCANO PLOT INTERPRETATION\n"
    interp += f"{'='*70}\n\n"
    
    interp += f"The volcano plot shows both statistical significance (y-axis) and\n"
    interp += f"effect size (x-axis) for each brain connection.\n\n"
    
    interp += f"Key findings:\n"
    interp += f"  • Significant + Large effect (top priority): {sig_large} edges\n"
    interp += f"  • Significant + Small effect: {sig_small} edges\n"
    interp += f"  • Not significant but Large effect: {ns_large} edges\n\n"
    
    if sig_large > 0:
        interp += f"✓ {sig_large} edges have both statistical significance AND meaningful\n"
        interp += f"  effect sizes - these are the most reliable findings.\n"
    
    if ns_large > 0:
        interp += f"\n⚠ {ns_large} edges have large effects but don't reach significance.\n"
        interp += f"  These may become significant with larger samples.\n"
    
    interp += f"\n{'='*70}\n"
    
    print(interp)
    return interp


def get_heatmap_interpretation(results: dict) -> str:
    sig_comps = results['significant_components']
    test_stats = results['test_statistics']
    n_edges = len(test_stats)
    
    interp = f"\n{'='*70}\n"
    interp += "🗺️ CONNECTIVITY HEATMAP INTERPRETATION\n"
    interp += f"{'='*70}\n\n"
    
    interp += f"The heatmaps show t-statistics for each brain connection:\n"
    interp += f"  • Red: Higher connectivity in Group A\n"
    interp += f"  • Blue: Higher connectivity in Group B\n"
    interp += f"  • White: No difference\n\n"
    
    interp += f"Left panel: All {n_edges:,} edges tested\n"
    
    if len(sig_comps) > 0:
        n_sig = sig_comps[0]['size']
        interp += f"Right panel: {n_sig} edges in the significant network\n\n"
        interp += f"The sparse pattern in the right panel shows which specific\n"
        interp += f"brain connections drive the group differences.\n"
    else:
        interp += f"Right panel: No significant edges (empty)\n"
    
    interp += f"\n{'='*70}\n"
    
    print(interp)
    return interp
