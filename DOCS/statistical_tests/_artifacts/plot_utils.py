import numpy as np

def plot_pvalue_histogram(ax, pvals, title):
    pvals_plot = pvals[pvals > 0]
    ax.set_title(title)
    ax.set_xlabel('-log10(p-value)')
    ax.set_ylabel('Count')
    if pvals_plot.size > 0:
        ax.hist(-np.log10(pvals_plot), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No p-values > 0 to plot', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes)