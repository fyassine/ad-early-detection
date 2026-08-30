import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Dict, Any, List, Tuple
from skbio.stats.distance import permanova as skbio_permanova
from skbio import DistanceMatrix
import pingouin as pg


def analyze_group_differences(group1_data, group2_data, alpha=0.05):
    n_edges = group1_data.shape[1]
    pvals = []
    stats = []

    for i in range(n_edges):
        stat, pval = mannwhitneyu(group1_data[:, i], group2_data[:, i], alternative='two-sided')
        pvals.append(pval)
        stats.append(stat)

    pvals = np.array(pvals)
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    sig_count = np.sum(pvals_fdr < alpha)

    return pvals_fdr, stats, sig_count


def compute_effect_sizes(group1_data, group2_data):
    n_edges = group1_data.shape[1]
    cohens_d = np.zeros(n_edges)
    
    for i in range(n_edges):
        d = pg.compute_effsize(group1_data[:, i], group2_data[:, i], eftype='cohen')
        cohens_d[i] = d
    
    return cohens_d


def run_diagnostic_tests(cohorts: dict, group_a: str = 'healthy', group_b: str = 'ad', plot: bool = True) -> Dict[str, Any]:
    if group_a not in cohorts or group_b not in cohorts:
        return {'valid': False}

    name_a, name_b = cohorts[group_a]['name'], cohorts[group_b]['name']
    edges_a, edges_b = cohorts[group_a]['edges'], cohorts[group_b]['edges']
    n_a, n_b = len(cohorts[group_a]['ids']), len(cohorts[group_b]['ids'])
    n_edges = edges_a.shape[1]

    avg_sd_a = np.mean(np.std(edges_a, axis=0))
    avg_sd_b = np.mean(np.std(edges_b, axis=0))

    pvals_fdr, _, sig_fdr = analyze_group_differences(edges_a, edges_b)
    pvals = []
    for i in range(n_edges):
        _, p = mannwhitneyu(edges_a[:, i], edges_b[:, i], alternative='two-sided')
        pvals.append(p)
    pvals = np.array(pvals)
    sig_unc = np.sum(pvals < 0.05)

    cohens_d = compute_effect_sizes(edges_a, edges_b)
    abs_d = np.abs(cohens_d)
    med_d = np.median(abs_d)
    pct_medium = np.sum(abs_d > 0.5) / n_edges * 100
    pct_large = np.sum(abs_d > 0.8) / n_edges * 100

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(pvals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axhline(y=n_edges / 50, color='r', linestyle='--', label='Null (Uniform)')
        axes[0].set_title('P-value Distribution')
        axes[0].set_xlabel('Uncorrected P-value')
        axes[0].legend()

        axes[1].hist(cohens_d, bins=50, edgecolor='black', color='orange', alpha=0.7)
        axes[1].axvline(x=0, color='k', linestyle='--')
        axes[1].set_title("Effect Size Distribution (Cohen's d)")
        axes[1].set_xlabel("d (Negative = Group B higher)")
        plt.tight_layout()
        plt.show()

    return {
        'valid': True,
        'pair_name': f"{name_a} vs {name_b}",
        'n_a': n_a,
        'n_b': n_b,
        'min_sample_size': min(n_a, n_b),
        'avg_sd_a': avg_sd_a,
        'avg_sd_b': avg_sd_b,
        'sig_uncorrected': sig_unc,
        'sig_fdr_count': sig_fdr,
        'sig_fdr_pct': (sig_fdr / n_edges) * 100,
        'median_effect_size': med_d,
        'pct_medium_effect': pct_medium,
        'pct_large_effect': pct_large,
        'pvals': pvals,
        'pvals_fdr': pvals_fdr,
        'cohens_d': cohens_d
    }


def run_batch_diagnostics(cohorts: dict, comparisons: List[Tuple[str, str]], plot: bool = False) -> List[Dict]:
    results = []

    for group_a, group_b in comparisons:
        metrics = run_diagnostic_tests(cohorts, group_a, group_b, plot=plot)
        if not metrics['valid']:
            continue

        flags = []
        status = "SAFE"

        if metrics['min_sample_size'] < 20:
            flags.append("Low Sample Size (<20)")
            status = "HIGH RISK"
        elif metrics['min_sample_size'] < 30:
            flags.append("Small Sample Size (<30)")
            if status == "SAFE":
                status = "CAUTION"

        if metrics['sig_fdr_count'] == 0:
            flags.append("No Signal (0 FDR edges)")
            status = "HIGH RISK"
        elif metrics['sig_fdr_pct'] < 1.0:
            flags.append("Weak Signal (<1% FDR edges)")
            if status == "SAFE":
                status = "CAUTION"

        if metrics['pct_medium_effect'] < 0.5:
            flags.append("Tiny Effects (Few edges > 0.5)")
            if status == "SAFE":
                status = "CAUTION"

        results.append({
            'pair': metrics['pair_name'],
            'status': status,
            'flags': flags,
            'fdr_edges': metrics['sig_fdr_count'],
            'median_d': metrics['median_effect_size'],
            'metrics': metrics
        })

    return results


def permanova(cohorts: dict, n_permutations: int = 5000, distance_metric: str = 'euclidean') -> dict:
    group_names = list(cohorts.keys())
    all_data = []
    group_labels = []
    subject_ids = []

    for group_name in group_names:
        edges = cohorts[group_name]['edges']
        ids = cohorts[group_name]['ids']
        all_data.append(edges)
        group_labels.extend([group_name] * edges.shape[0])
        subject_ids.extend([f"{group_name}_{i}" for i in range(edges.shape[0])])

    all_data = np.vstack(all_data)
    group_labels = np.array(group_labels)

    dist_condensed = pdist(all_data, metric=distance_metric)
    dist_matrix = squareform(dist_condensed)
    dm = DistanceMatrix(dist_matrix, ids=subject_ids)

    import pandas as pd
    grouping = pd.Series(group_labels, index=subject_ids, name='group')

    results = skbio_permanova(dm, grouping, permutations=n_permutations)

    return {
        'f_statistic': results['test statistic'],
        'r_squared': results['test statistic'] * (len(group_names) - 1) / (len(group_labels) - 1) / (1 + results['test statistic'] * (len(group_names) - 1) / (len(group_labels) - 1)),
        'p_value': results['p-value'],
        'n_permutations': n_permutations,
        'distance_metric': distance_metric,
        'sample_size': results['sample size'],
        'num_groups': results['number of groups']
    }


def network_based_statistic(cohorts: dict,
                            group_a: str = 'healthy',
                            group_b: str = 'ad',
                            primary_threshold: float = 3.0,
                            n_permutations: int = 5000,
                            alpha: float = 0.05) -> dict:
    if group_a not in cohorts or group_b not in cohorts:
        raise ValueError(f"Groups '{group_a}' or '{group_b}' not found")

    edges_a = cohorts[group_a]['edges']
    edges_b = cohorts[group_b]['edges']
    n_edges = edges_a.shape[1]

    n_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    triu_indices = np.triu_indices(n_nodes, k=1)

    def fast_ttest(d1, d2):
        mean1, mean2 = np.mean(d1, axis=0), np.mean(d2, axis=0)
        var1, var2 = np.var(d1, axis=0, ddof=1), np.var(d2, axis=0, ddof=1)
        n1, n2 = d1.shape[0], d2.shape[0]
        denom = np.sqrt(var1 / n1 + var2 / n2)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (mean1 - mean2) / denom
        t[np.isnan(t)] = 0
        return t

    test_stats = fast_ttest(edges_a, edges_b)
    suprathreshold_edges = np.abs(test_stats) > primary_threshold

    if np.sum(suprathreshold_edges) == 0:
        return {'significant_components': [], 'null_distribution': np.array([0])}

    observed_components = _find_components(suprathreshold_edges, test_stats, n_nodes, triu_indices)

    if len(observed_components) == 0:
        return {'significant_components': [], 'null_distribution': np.array([0])}

    all_edges = np.vstack([edges_a, edges_b])
    n_a = edges_a.shape[0]
    n_total = all_edges.shape[0]
    max_component_sizes = np.zeros(n_permutations)

    for perm_idx in range(n_permutations):
        perm_indices = np.random.permutation(n_total)
        perm_a = all_edges[perm_indices[:n_a], :]
        perm_b = all_edges[perm_indices[n_a:], :]
        perm_stats = fast_ttest(perm_a, perm_b)
        perm_suprathreshold = np.abs(perm_stats) > primary_threshold
        max_size = _get_max_component_size(perm_suprathreshold, n_nodes, triu_indices)
        max_component_sizes[perm_idx] = max_size

    significant_components = []
    for comp in observed_components:
        p_value = np.sum(max_component_sizes >= comp['size']) / n_permutations
        comp['p_value'] = p_value
        comp['significant'] = p_value < alpha
        if comp['significant']:
            significant_components.append(comp)

    return {
        'test_statistics': test_stats,
        'suprathreshold_edges': suprathreshold_edges,
        'all_components': observed_components,
        'significant_components': significant_components,
        'null_distribution': max_component_sizes,
        'primary_threshold': primary_threshold,
        'n_permutations': n_permutations
    }


def _find_components(suprathreshold_mask, test_stats, n_nodes, triu_indices):
    if not np.any(suprathreshold_mask):
        return []

    rows = triu_indices[0][suprathreshold_mask]
    cols = triu_indices[1][suprathreshold_mask]
    data = np.ones(len(rows), dtype=int)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    n_comps, labels = connected_components(adj_matrix, directed=False, return_labels=True)

    if n_comps == n_nodes:
        return []

    row_labels = labels[rows]
    unique_ids, counts = np.unique(row_labels, return_counts=True)
    original_indices = np.where(suprathreshold_mask)[0]

    components = []
    for comp_id, size in zip(unique_ids, counts):
        mask_in_subset = (row_labels == comp_id)
        component_edges = original_indices[mask_in_subset]
        comp_stats = test_stats[component_edges]

        components.append({
            'edges': component_edges,
            'size': size,
            'mean_stat': np.mean(np.abs(comp_stats)),
            'max_stat': np.max(np.abs(comp_stats))
        })

    components.sort(key=lambda x: x['size'], reverse=True)
    return components


def _get_max_component_size(suprathreshold_mask, n_nodes, triu_indices):
    if not np.any(suprathreshold_mask):
        return 0

    rows = triu_indices[0][suprathreshold_mask]
    cols = triu_indices[1][suprathreshold_mask]
    data = np.ones(len(rows), dtype=int)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    n_comps, labels = connected_components(adj_matrix, directed=False, return_labels=True)

    if n_comps == n_nodes:
        return 0

    edge_labels = labels[rows]
    _, counts = np.unique(edge_labels, return_counts=True)
    return np.max(counts)


def run_network_analysis(cohorts: dict,
                         group_a: str = None,
                         group_b: str = None,
                         nbs_threshold: float = 3.0,
                         n_perms: int = 1000,
                         run_nbs: bool = True,
                         run_permanova: bool = True) -> dict:
    group_names = list(cohorts.keys())

    if group_a is not None and group_b is not None:
        if group_a not in cohorts or group_b not in cohorts:
            return None
        pairwise_cohorts = {group_a: cohorts[group_a], group_b: cohorts[group_b]}
    else:
        pairwise_cohorts = cohorts
        if run_nbs and len(group_names) >= 2:
            group_a, group_b = group_names[0], group_names[1]

    results = {}

    if run_permanova:
        results['permanova'] = permanova(pairwise_cohorts, n_permutations=n_perms)

    if run_nbs and group_a and group_b:
        results['nbs'] = network_based_statistic(
            pairwise_cohorts,
            group_a=group_a,
            group_b=group_b,
            primary_threshold=nbs_threshold,
            n_permutations=n_perms
        )

    return results


def visualize_nbs_results(nbs_results: dict, alpha: float = 0.05):
    test_stats = nbs_results['test_statistics']
    suprathreshold = nbs_results['suprathreshold_edges']
    null_dist = nbs_results['null_distribution']
    components = nbs_results['all_components']
    threshold = nbs_results['primary_threshold']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(test_stats, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(threshold, color='r', linestyle='--', label=f'Threshold: ±{threshold}')
    axes[0, 0].axvline(-threshold, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Test Statistic (t)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Test Statistics')
    axes[0, 0].legend()

    axes[0, 1].hist(null_dist, bins=50, alpha=0.7, color='gray', edgecolor='black')
    if len(components) > 0:
        for comp in components[:3]:
            color = 'green' if comp.get('significant', False) else 'orange'
            axes[0, 1].axvline(comp['size'], color=color, linestyle='--',
                               label=f"Obs: {comp['size']} (p={comp.get('p_value', 0):.3f})")
    axes[0, 1].set_xlabel('Maximum Component Size')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Null Distribution (Permutation Test)')
    axes[0, 1].legend()

    if len(components) > 0:
        comp_sizes = [c['size'] for c in components]
        comp_pvals = [c.get('p_value', 1) for c in components]
        colors = ['green' if p < alpha else 'red' for p in comp_pvals]
        axes[1, 0].bar(range(len(comp_sizes)), comp_sizes, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(np.percentile(null_dist, 95), color='orange', linestyle='--', label='95th percentile')
        axes[1, 0].set_xlabel('Component Index')
        axes[1, 0].set_ylabel('Component Size')
        axes[1, 0].set_title('Observed Component Sizes')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No components found', ha='center', va='center', transform=axes[1, 0].transAxes)

    axes[1, 1].axis('off')
    summary = f"Suprathreshold: {np.sum(suprathreshold)}\nComponents: {len(components)}\n"
    summary += f"Significant: {sum(1 for c in components if c.get('significant', False))}\n"
    summary += f"Threshold: {threshold}\nPermutations: {len(null_dist)}"
    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes, fontfamily='monospace', fontsize=12, va='top')

    plt.tight_layout()
    plt.show()


def visualize_permanova(permanova_results: dict):
    if 'null_distribution' not in permanova_results:
        print("No null distribution available for visualization")
        return
        
    null_dist = permanova_results.get('null_distribution', np.array([]))
    obs_f = permanova_results['f_statistic']
    p_value = permanova_results['p_value']

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(null_dist, bins=50, alpha=0.7, color='gray', edgecolor='black', density=True)
    ax.axvline(obs_f, color='red', linestyle='--', linewidth=2, label=f'Observed F={obs_f:.3f}')
    ax.set_xlabel('F-statistic')
    ax.set_ylabel('Density')
    ax.set_title(f'PERMANOVA (p = {p_value:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.show()
