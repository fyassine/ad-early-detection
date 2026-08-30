import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
from collections import deque
import warnings
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def network_based_statistic(cohorts: dict,
                            group_a: str = 'healthy',
                            group_b: str = 'ad',
                            primary_threshold: float = 3.0,
                            n_permutations: int = 5000,
                            alpha: float = 0.05,
                            stat_type: str = 't') -> dict:
    """
    Network-Based Statistic (NBS) for brain connectivity data.
    (Optimized for speed using vectorization and sparse graph theory)
    """

    # Validation
    if group_a not in cohorts or group_b not in cohorts:
        raise ValueError(f"Groups '{group_a}' or '{group_b}' not found")

    edges_a = cohorts[group_a]['edges']
    edges_b = cohorts[group_b]['edges']
    n_edges = edges_a.shape[1]
    n_subs_a = edges_a.shape[0]
    n_subs_b = edges_b.shape[0]

    # --- OPTIMIZATION PRE-CALCULATION ---
    # Calculate number of nodes and pre-compute indices for fast graph building
    # n_edges = n_nodes * (n_nodes - 1) / 2
    n_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    triu_indices = np.triu_indices(n_nodes, k=1)  # Row and Col indices for every edge 0..n_edges
    # ------------------------------------

    print("=" * 60)
    print("NETWORK-BASED STATISTIC (NBS) - FAST VERSION")
    print("=" * 60)
    print("\nWHAT THIS TEST DOES:")
    print("  The NBS identifies CONNECTED SUBNETWORKS (not individual edges)")
    print("  that differ between groups. It's like cluster-based thresholding")
    print("  but for network data instead of brain images.")
    print()
    print("  How it works:")
    print("    1. Tests each edge independently (t-test)")
    print("    2. Keeps edges above threshold (forms suprathreshold network)")
    print("    3. Finds connected components in this network")
    print("    4. Uses permutation testing to assess significance")
    print()
    print("  Key Advantage: More powerful than edge-by-edge FDR correction")
    print("                 when brain changes form connected networks")
    print()
    print("  Key Limitation: Only provides WEAK FWE control")
    print("                  (can't declare individual edges significant,")
    print("                  only entire components)")
    print()
    print("THRESHOLD SELECTION GUIDE:")
    print("  t = 2.5 → Liberal (exploratory, finds more but riskier)")
    print("  t = 3.0 → Moderate (RECOMMENDED: balanced sensitivity/specificity)")
    print("  t = 3.5 → Conservative (fewer false positives, may miss weak effects)")
    print()
    print("  Lower threshold = More edges → Larger components → Higher power")
    print("  Higher threshold = Fewer edges → Smaller components → Less noise")
    print()
    print("-" * 60)
    print(f"Group A ({cohorts[group_a]['name']}): {edges_a.shape[0]} subjects")
    print(f"Group B ({cohorts[group_b]['name']}): {edges_b.shape[0]} subjects")
    print(f"Total edges tested: {n_edges}")
    print(f"Primary threshold: {primary_threshold}")
    print(f"Permutations: {n_permutations}")

    # Step 1: Compute test statistic for each edge
    print("\n[1/4] Computing test statistics for each edge (Vectorized)...")

    # --- OPTIMIZATION 1: Vectorized T-Test (No loop) ---
    # We define a fast function here to use later in permutations
    def fast_ttest(d1, d2):
        mean1 = np.mean(d1, axis=0)
        mean2 = np.mean(d2, axis=0)
        var1 = np.var(d1, axis=0, ddof=1)
        var2 = np.var(d2, axis=0, ddof=1)
        n1 = d1.shape[0]
        n2 = d2.shape[0]
        denom = np.sqrt(var1 / n1 + var2 / n2)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (mean1 - mean2) / denom
        t[np.isnan(t)] = 0
        return t

    test_stats = fast_ttest(edges_a, edges_b)
    # ----------------------------------------------------

    # Step 2: Identify suprathreshold edges and find components
    print("[2/4] Identifying connected components in suprathreshold network...")
    suprathreshold_edges = np.abs(test_stats) > primary_threshold
    n_suprathreshold = np.sum(suprathreshold_edges)

    print(f"    Suprathreshold edges: {n_suprathreshold} / {n_edges} ({100 * n_suprathreshold / n_edges:.2f}%)")

    if n_suprathreshold == 0:
        print("    WARNING: No edges exceed threshold. Consider lowering threshold.")
        return {'significant_components': [], 'null_distribution': np.array([0])}

    # Find connected components (Using optimized function)
    observed_components = _find_components_fast(suprathreshold_edges, test_stats, n_nodes, triu_indices)

    if len(observed_components) == 0:
        print("    No connected components found.")
        return {'significant_components': [], 'null_distribution': np.array([0])}

    print(f"    Found {len(observed_components)} component(s)")

    # Step 3: Permutation testing
    print(f"[3/4] Running {n_permutations} permutations...")

    all_edges = np.vstack([edges_a, edges_b])
    n_a = edges_a.shape[0]
    n_total = all_edges.shape[0]

    max_component_sizes = np.zeros(n_permutations)

    # Progress tracking
    progress_points = [int(n_permutations * p) for p in [0.25, 0.5, 0.75]]

    for perm_idx in range(n_permutations):
        # Randomly permute group labels
        # Note: We permute indices on the stacked array to avoid memory copies
        perm_indices = np.random.permutation(n_total)
        perm_a = all_edges[perm_indices[:n_a], :]
        perm_b = all_edges[perm_indices[n_a:], :]

        # --- OPTIMIZATION 1 REUSED: Vectorized Stats ---
        perm_stats = fast_ttest(perm_a, perm_b)

        # --- OPTIMIZATION 2 REUSED: Fast Component Finding ---
        perm_suprathreshold = np.abs(perm_stats) > primary_threshold

        # We only need the MAX size for the null distribution, so we use a lighter version
        # of the component finder that doesn't store all edge details
        max_size = _get_max_component_size_fast(perm_suprathreshold, n_nodes, triu_indices)
        max_component_sizes[perm_idx] = max_size

    print("    Permutation testing complete!")

    # Step 4: Calculate p-values for observed components
    print("[4/4] Computing corrected p-values...")

    significant_components = []
    for comp in observed_components:
        # P-value: proportion of permutations with component >= observed size
        p_value = np.sum(max_component_sizes >= comp['size']) / n_permutations
        comp['p_value'] = p_value
        comp['significant'] = p_value < alpha

        if comp['significant']:
            significant_components.append(comp)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if len(significant_components) > 0:
        print(f"✓ Found {len(significant_components)} SIGNIFICANT component(s):\n")
        for idx, comp in enumerate(significant_components):
            print(f"  Component {idx + 1}:")
            print(f"    Size: {comp['size']} edges")
            print(f"    P-value: {comp['p_value']:.4f} (corrected)")
            print(f"    Mean test statistic: {comp['mean_stat']:.3f}")
            print()
        print("INTERPRETATION:")
        print(f"  ✓ These {len(significant_components)} connected subnetwork(s) show")
        print(f"    significant differences between {cohorts[group_a]['name']}")
        print(f"    and {cohorts[group_b]['name']}.")
        print()
        print("  NOTE: Individual edges within these components cannot be")
        print("        declared significant (weak FWE control). The ENTIRE")
        print("        component is what's statistically significant.")
        print()
        print("  NEXT STEPS:")
        print("    • Examine which brain regions these components connect")
        print("    • Consider biological plausibility of these networks")
        print("    • Validate in independent dataset if possible")
    else:
        print("✗ No significant components found at α = {:.3f}".format(alpha))
        print(f"  Largest component: {observed_components[0]['size']} edges")
        print(f"  P-value: {observed_components[0]['p_value']:.4f}")
        print()
        print("INTERPRETATION:")
        print(f"  The connectivity differences between {cohorts[group_a]['name']}")
        print(f"  and {cohorts[group_b]['name']} do not form statistically")
        print("  significant connected networks at this threshold.")
        print()
        print("  POSSIBLE REASONS:")
        print("    • True null: No network-level differences exist")
        print("    • Threshold too high: Try t=2.5 (more liberal)")
        print("    • Sample size too small: Need more statistical power")
        print("    • Differences are isolated (not networked): Use FDR instead")
        print()
        print("  RECOMMENDATIONS:")
        print("    • Run PERMANOVA to test for ANY multivariate differences")
        print("    • Try lower threshold (exploratory analysis)")
        print("    • Consider effect sizes of individual edges")

    # Visualization
    _visualize_nbs_results(test_stats, suprathreshold_edges, max_component_sizes,
                           observed_components, primary_threshold, alpha)

    return {
        'test_statistics': test_stats,
        'suprathreshold_edges': suprathreshold_edges,
        'all_components': observed_components,
        'significant_components': significant_components,
        'null_distribution': max_component_sizes,
        'primary_threshold': primary_threshold,
        'n_permutations': n_permutations
    }


def _find_components_fast(suprathreshold_mask, test_stats, n_nodes, triu_indices):
    """
    Optimized component finding using scipy.sparse.csgraph.
    Replaces the manual BFS deque implementation.
    """
    if not np.any(suprathreshold_mask):
        return []

    # Map 1D edge indices back to 2D node pairs (i, j)
    # We only care about suprathreshold edges
    rows = triu_indices[0][suprathreshold_mask]
    cols = triu_indices[1][suprathreshold_mask]

    # Create sparse adjacency matrix (Nodes x Nodes)
    # 1 indicates a suprathreshold connection exists
    data = np.ones(len(rows), dtype=int)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Find connected components on the NODE graph
    # (Fast C-implementation)
    n_comps, labels = connected_components(adj_matrix, directed=False, return_labels=True)

    if n_comps == n_nodes:  # No connections
        return []

    # Now we need to map back to edges.
    # An edge is part of a component if both its nodes belong to that component.
    # Get component label for every node in our suprathreshold list
    row_labels = labels[rows]
    col_labels = labels[cols]

    # Identify which component each EDGE belongs to
    # Since we built the graph from these edges, row_labels must equal col_labels
    # for all valid edges in a component.
    edge_component_ids = row_labels

    # Get unique components and their sizes (number of edges)
    unique_ids, counts = np.unique(edge_component_ids, return_counts=True)

    # Store indices of suprathreshold mask to map back to original test_stats array
    original_indices = np.where(suprathreshold_mask)[0]

    components = []
    for comp_id, size in zip(unique_ids, counts):
        # Find which edges belong to this component
        mask_in_subset = (edge_component_ids == comp_id)

        # Map back to original edge indices (0...N_edges)
        component_edges = original_indices[mask_in_subset]

        # Calculate stats
        comp_stats = test_stats[component_edges]

        components.append({
            'edges': component_edges,
            'size': size,
            'mean_stat': np.mean(np.abs(comp_stats)),
            'max_stat': np.max(np.abs(comp_stats))
        })

    # Sort by size (largest first)
    components.sort(key=lambda x: x['size'], reverse=True)
    return components


def _get_max_component_size_fast(suprathreshold_mask, n_nodes, triu_indices):
    """
    Ultra-light version of component finder just for permutation loop.
    Returns only the size of the largest component.
    """
    if not np.any(suprathreshold_mask):
        return 0

    rows = triu_indices[0][suprathreshold_mask]
    cols = triu_indices[1][suprathreshold_mask]
    data = np.ones(len(rows), dtype=int)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    n_comps, labels = connected_components(adj_matrix, directed=False, return_labels=True)

    if n_comps == n_nodes:
        return 0

    # Count edges per component label
    # Each edge connects two nodes with the same label
    edge_labels = labels[rows]
    _, counts = np.unique(edge_labels, return_counts=True)

    return np.max(counts)


# Keep the original helper for compatibility (though not used in the optimized path)
def _create_edge_adjacency_fast(n_nodes, n_edges=None):
    """
    (Legacy) Create edge adjacency for a brain connectivity matrix.
    Kept to prevent ImportErrors if imported elsewhere.
    """
    edge_to_nodes = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_to_nodes.append((i, j))

    node_to_edges = {node: [] for node in range(n_nodes)}
    for edge_idx, (i, j) in enumerate(edge_to_nodes):
        node_to_edges[i].append(edge_idx)
        node_to_edges[j].append(edge_idx)

    adjacency = {}
    for edge_idx, (i, j) in enumerate(edge_to_nodes):
        neighbors = set()
        neighbors.update(node_to_edges[i])
        neighbors.update(node_to_edges[j])
        neighbors.discard(edge_idx)
        adjacency[edge_idx] = list(neighbors)

    return adjacency


# Keep visualization as is
def _visualize_nbs_results(test_stats, suprathreshold, null_dist,
                           components, threshold, alpha):
    """Visualize NBS results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Test statistics distribution
    axes[0, 0].hist(test_stats, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(threshold, color='r', linestyle='--', label=f'Threshold: ±{threshold}')
    axes[0, 0].axvline(-threshold, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Test Statistic (t)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Test Statistics')
    axes[0, 0].legend()

    # 2. Null distribution of max component sizes
    axes[0, 1].hist(null_dist, bins=50, alpha=0.7, color='gray', edgecolor='black')
    if len(components) > 0:
        for comp in components[:3]:  # Show top 3
            color = 'green' if comp.get('significant', False) else 'orange'
            axes[0, 1].axvline(comp['size'], color=color, linestyle='--',
                               label=f"Obs: {comp['size']} (p={comp.get('p_value', 0):.3f})")
    axes[0, 1].set_xlabel('Maximum Component Size')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Null Distribution (Permutation Test)')
    axes[0, 1].legend()

    # 3. Component sizes
    if len(components) > 0:
        comp_sizes = [c['size'] for c in components]
        comp_pvals = [c.get('p_value', 1) for c in components]
        colors = ['green' if p < alpha else 'red' for p in comp_pvals]

        axes[1, 0].bar(range(len(comp_sizes)), comp_sizes, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(np.percentile(null_dist, 95), color='orange',
                           linestyle='--', label='95th percentile (null)')
        axes[1, 0].set_xlabel('Component Index')
        axes[1, 0].set_ylabel('Component Size (# edges)')
        axes[1, 0].set_title('Observed Component Sizes')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No components found',
                        ha='center', va='center', transform=axes[1, 0].transAxes)

    # 4. Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    NBS SUMMARY
    {'=' * 30}

    Suprathreshold edges: {np.sum(suprathreshold)}
    Components found: {len(components)}
    Significant (α={alpha}): {sum(1 for c in components if c.get('significant', False))}

    Primary threshold: {threshold}
    Permutations: {len(null_dist)}

    Null distribution:
      Mean max size: {np.mean(null_dist):.1f}
      95th percentile: {np.percentile(null_dist, 95):.1f}
    """

    if len(components) > 0:
        summary_text += f"\n    Largest component: {components[0]['size']}"
        summary_text += f"\n    P-value: {components[0].get('p_value', 1):.4f}"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def permanova(cohorts: dict,
              n_permutations: int = 5000,
              distance_metric: str = 'euclidean') -> dict:
    """
    PERMANOVA (Permutational Multivariate Analysis of Variance)

    Tests if group centroids differ in high-dimensional connectivity space
    without assuming multivariate normality. Does NOT assume edge independence.

    Parameters:
    -----------
    cohorts : dict
        Dictionary with group data
    n_permutations : int
        Number of permutations
    distance_metric : str
        Distance metric ('euclidean', 'correlation', 'cityblock', etc.)

    Returns:
    --------
    dict with F-statistic, p-value, and effect size (R²)
    """

    print("\n" + "=" * 60)
    print("PERMANOVA (Permutational MANOVA)")
    print("=" * 60)
    print("\nWHAT THIS TEST DOES:")
    print("  PERMANOVA tests whether groups differ in their OVERALL")
    print("  connectivity patterns using multivariate distances.")
    print()
    print("  How it works:")
    print("    1. Treats each subject as a point in high-dimensional space")
    print("    2. Calculates distances between all subject pairs")
    print("    3. Tests if within-group distances < between-group distances")
    print("    4. Uses permutation to assess significance")
    print()
    print("  KEY ADVANTAGES:")
    print("    ✓ Does NOT assume edges are independent")
    print("      (Perfect for brain networks with correlated connections!)")
    print("    ✓ Does NOT assume multivariate normality")
    print("    ✓ Captures overall group differences (not just univariate)")
    print("    ✓ Provides R² effect size (% variance explained)")
    print()
    print("  WHAT IT TELLS YOU:")
    print("    Significant result → Groups have different connectivity patterns")
    print("    Non-significant → Groups are similar in connectivity space")
    print()
    print("  COMPLEMENTARY TO NBS:")
    print("    • PERMANOVA: 'Do groups differ at all?' (omnibus test)")
    print("    • NBS: 'WHERE do groups differ?' (localization)")
    print()
    print("-" * 60)

    # Combine all groups
    group_names = list(cohorts.keys())
    all_data = []
    group_labels = []

    for group_name in group_names:
        edges = cohorts[group_name]['edges']
        all_data.append(edges)
        group_labels.extend([group_name] * edges.shape[0])
        print(f"{cohorts[group_name]['name']}: {edges.shape[0]} subjects")

    all_data = np.vstack(all_data)
    group_labels = np.array(group_labels)
    n_total = all_data.shape[0]

    print(f"Total subjects: {n_total}")
    print(f"Distance metric: {distance_metric}")

    # Compute distance matrix
    print("\n[1/3] Computing distance matrix...")
    dist_matrix = squareform(pdist(all_data, metric=distance_metric))

    # Compute observed F-statistic
    print("[2/3] Computing observed F-statistic...")
    obs_f, obs_r2 = _compute_permanova_f(dist_matrix, group_labels, group_names)

    print(f"    Observed F-statistic: {obs_f:.4f}")
    print(f"    Observed R²: {obs_r2:.4f}")

    # Permutation test
    print(f"[3/3] Running {n_permutations} permutations...")

    perm_f_stats = np.zeros(n_permutations)

    for perm_idx in range(n_permutations):
        # Permute labels
        perm_labels = np.random.permutation(group_labels)
        perm_f, _ = _compute_permanova_f(dist_matrix, perm_labels, group_names)
        perm_f_stats[perm_idx] = perm_f

    # Calculate p-value
    p_value = np.sum(perm_f_stats >= obs_f) / n_permutations

    print("\n" + "=" * 60)
    print("PERMANOVA RESULTS")
    print("=" * 60)
    print(f"F-statistic: {obs_f:.4f}")
    print(f"R² (effect size): {obs_r2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print()

    if p_value < 0.001:
        print("✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)")
        interpretation = "STRONG"
    elif p_value < 0.01:
        print("✓✓ VERY SIGNIFICANT (p < 0.01)")
        interpretation = "CLEAR"
    elif p_value < 0.05:
        print("✓ SIGNIFICANT (p < 0.05)")
        interpretation = "MEANINGFUL"
    else:
        print("✗ NOT SIGNIFICANT (p >= 0.05)")
        interpretation = "NO"

    print()
    print("INTERPRETATION:")
    print(f"  R² = {obs_r2:.1%} of variance in connectivity is explained")
    print(f"  by group membership.")
    print()

    if obs_r2 < 0.01:
        effect_label = "Negligible"
        print(f"  Effect size: {effect_label} (R² < 1%)")
        print("  → Groups are nearly identical in connectivity space")
    elif obs_r2 < 0.06:
        effect_label = "Small"
        print(f"  Effect size: {effect_label} (R² = 1-6%)")
        print("  → Groups differ slightly in connectivity patterns")
    elif obs_r2 < 0.14:
        effect_label = "Medium"
        print(f"  Effect size: {effect_label} (R² = 6-14%)")
        print("  → Groups show noticeable differences in connectivity")
    else:
        effect_label = "Large"
        print(f"  Effect size: {effect_label} (R² ≥ 14%)")
        print("  → Groups are substantially different in connectivity space")

    print()
    if p_value < 0.05:
        print(f"  CONCLUSION: There is {interpretation} evidence that")
        print(f"              connectivity patterns differ between groups.")
        print()
        print("  WHAT THIS MEANS:")
        print("    • Groups occupy different regions in connectivity space")
        print("    • The overall 'connectivity fingerprint' differs")
        print("    • Differences may be distributed across many edges")
        print()
        print("  NEXT STEPS:")
        print("    • Use NBS to identify WHICH networks differ")
        print("    • Examine individual edges with highest contributions")
        print("    • Consider machine learning to classify based on connectivity")
    else:
        print(f"  CONCLUSION: No significant evidence that connectivity")
        print(f"              patterns differ between groups overall.")
        print()
        print("  POSSIBLE REASONS:")
        print("    • True null: Groups have similar connectivity")
        print("    • Small effect size (need larger sample)")
        print("    • High within-group variability")
        print("    • Preprocessing issues (check data quality)")
        print()
        print("  RECOMMENDATIONS:")
        print("    • Check sample size and power")
        print("    • Examine data quality and outliers")
        print("    • Consider more homogeneous subgroups")

    # Visualization
    _visualize_permanova(perm_f_stats, obs_f, p_value)

    return {
        'f_statistic': obs_f,
        'r_squared': obs_r2,
        'p_value': p_value,
        'null_distribution': perm_f_stats,
        'distance_metric': distance_metric
    }


def _compute_permanova_f(dist_matrix, labels, group_names):
    """Compute PERMANOVA F-statistic and R²"""
    n = len(labels)

    # Total sum of squares
    ss_total = np.sum(dist_matrix ** 2) / n

    # Within-group sum of squares
    ss_within = 0
    n_groups = len(group_names)

    for group in group_names:
        group_mask = labels == group
        n_group = np.sum(group_mask)

        if n_group > 1:
            group_dist = dist_matrix[np.ix_(group_mask, group_mask)]
            ss_within += np.sum(group_dist ** 2) / n_group

    # Between-group sum of squares
    ss_between = ss_total - ss_within

    # Degrees of freedom
    df_between = n_groups - 1
    df_within = n - n_groups

    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1

    # F-statistic
    f_stat = ms_between / ms_within if ms_within > 0 else 0

    # R² (effect size)
    r_squared = ss_between / ss_total if ss_total > 0 else 0

    return f_stat, r_squared


def _visualize_permanova(null_dist, obs_f, p_value):
    """Visualize PERMANOVA results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Null distribution
    axes[0].hist(null_dist, bins=50, alpha=0.7, color='gray', edgecolor='black', density=True)
    axes[0].axvline(obs_f, color='red', linestyle='--', linewidth=2,
                    label=f'Observed F={obs_f:.3f}')
    axes[0].set_xlabel('F-statistic')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'PERMANOVA Null Distribution\n(p = {p_value:.4f})')
    axes[0].legend()

    # Q-Q plot
    expected = np.sort(null_dist)
    theoretical_quantiles = np.linspace(0, 1, len(expected))

    axes[1].scatter(expected, theoretical_quantiles, alpha=0.3, s=10)
    axes[1].axvline(obs_f, color='red', linestyle='--', label='Observed')
    axes[1].set_xlabel('Permuted F-statistic')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Empirical CDF')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage function
def run_network_analysis(cohorts: dict,
                         group_a: str = None,
                         group_b: str = None,
                         nbs_threshold: float = 3.0,
                         n_perms: int = 1000,
                         run_nbs: bool = True,
                         run_permanova: bool = True):
    """
    Run NBS and/or PERMANOVA analyses on brain connectivity data

    RECOMMENDED WORKFLOW:
    1. Run PERMANOVA first → Answers: "Do groups differ AT ALL?"
    2. If significant, run NBS → Answers: "WHERE do they differ?"

    Parameters:
    -----------
    cohorts : dict
        Your cohort data dictionary
    group_a, group_b : str or None
        Groups to compare for NBS (required if run_nbs=True)
        If None, will use first two groups in cohorts
    nbs_threshold : float
        Primary threshold for NBS
        • 2.5 = Liberal (exploratory, finds more)
        • 3.0 = Moderate (RECOMMENDED, balanced)
        • 3.5 = Conservative (fewer false positives)
    n_perms : int
        Number of permutations
        • 1000 = Quick testing
        • 5000+ = Publication-quality results
    run_nbs : bool
        Whether to run NBS (only for 2-group comparisons)
    run_permanova : bool
        Whether to run PERMANOVA (works with 2+ groups)

    Returns:
    --------
    dict with 'nbs' and/or 'permanova' results
    """

    group_names = list(cohorts.keys())

    # Determine if this is a pairwise comparison or multi-group analysis
    if group_a is not None and group_b is not None:
        # PAIRWISE mode: User specified two groups
        if group_a not in cohorts or group_b not in cohorts:
            print(f"ERROR: Groups '{group_a}' and/or '{group_b}' not found in cohorts")
            print(f"Available groups: {group_names}")
            return None

        # Create subset with only the two groups of interest
        pairwise_cohorts = {group_a: cohorts[group_a], group_b: cohorts[group_b]}
        analysis_mode = 'pairwise'
        n_groups = 2
    else:
        # MULTI-GROUP mode: Analyze all groups
        pairwise_cohorts = cohorts
        analysis_mode = 'multigroup'
        n_groups = len(cohorts)

        # Auto-select groups for NBS if requested
        if run_nbs:
            if n_groups >= 2:
                group_a = group_names[0]
                group_b = group_names[1]
                print(f"NOTE: Auto-selected groups for NBS: '{group_a}' vs '{group_b}'")
            else:
                print("ERROR: Need at least 2 groups for analysis")
                return None

    print("=" * 70)
    print("COMPREHENSIVE NETWORK ANALYSIS FOR BRAIN CONNECTIVITY")
    print("=" * 70)

    if analysis_mode == 'pairwise':
        if analysis_mode == 'pairwise':
            print(f"\nPAIRWISE COMPARISON MODE")
            print(f"Analyzing 2 groups:")
            print(f"  • {pairwise_cohorts[group_a]['name']} (n={len(pairwise_cohorts[group_a]['ids'])})")
            print(f"  • {pairwise_cohorts[group_b]['name']} (n={len(pairwise_cohorts[group_b]['ids'])})")
            print()
            print("This analysis includes TWO complementary tests:")
            print()
            print("  1. PERMANOVA (Permutational MANOVA)")
            print("     → Tests if groups differ in OVERALL connectivity")
            print("     → Does NOT assume edge independence")
            print("     → Omnibus test: 'Are groups different?'")
            print()
            print("  2. Network-Based Statistic (NBS)")
            print("     → Identifies WHICH subnetworks differ")
            print("     → More powerful than edge-by-edge correction")
            print("     → Localization: 'WHERE are they different?'")
            print()
            print("  INTERPRETATION GUIDE:")
            print("  ┌─────────────────┬──────────────┬─────────────────────────┐")
            print("  │  PERMANOVA      │     NBS      │    Interpretation       │")
            print("  ├─────────────────┼──────────────┼─────────────────────────┤")
            print("  │  Significant    │  Significant │ ✓ Clear network effects │")
            print("  │  Significant    │  Not sig.    │ ~ Diffuse differences   │")
            print("  │  Not sig.       │  Significant │ ⚠ Recheck (unlikely)    │")
            print("  │  Not sig.       │  Not sig.    │ ✗ No group differences  │")
            print("  └─────────────────┴──────────────┴─────────────────────────┘")
        else:
            print(f"\nMULTI-GROUP ANALYSIS MODE")
            print(f"Analyzing {n_groups} groups:")
            for name in group_names:
                print(f"  • {cohorts[name]['name']} (n={len(cohorts[name]['ids'])})")
            print()
            print("With 3+ groups, running PERMANOVA for omnibus test:")
            print()
            print("  PERMANOVA (Permutational MANOVA)")
            print("  → Tests if ANY groups differ in connectivity")
            print("  → Does NOT assume edge independence")
            print("  → Omnibus test across all groups")
            print()
            print("  NOTE: NBS is only for 2-group comparisons.")
            print("        If PERMANOVA is significant, run NBS on specific pairs.")
            run_nbs = False  # Can't run NBS with 3+ groups

    print()
    print("=" * 70)

    results = {}

    # Run PERMANOVA first (works with any number of groups)
    if run_permanova:
        print("\n" + "▶" * 35)
        print("STEP 1: PERMANOVA (Omnibus Test)")
        print("▶" * 35)
        permanova_results = permanova(
            pairwise_cohorts,  # Use the correct cohort subset
            n_permutations=n_perms
        )
        results['permanova'] = permanova_results

    # Run NBS second (only for 2 groups)
    if run_nbs and analysis_mode == 'pairwise':
        print("\n" + "▶" * 35)
        print("STEP 2: Network-Based Statistic (Localization)")
        print("▶" * 35)
        nbs_results = network_based_statistic(
            pairwise_cohorts,  # Use the correct cohort subset
            group_a=group_a,
            group_b=group_b,
            primary_threshold=nbs_threshold,
            n_permutations=n_perms
        )
        results['nbs'] = nbs_results

    # Combined interpretation
    if analysis_mode == 'pairwise' and run_permanova and run_nbs:
        print("\n" + "=" * 70)
        print("COMBINED INTERPRETATION")
        print("=" * 70)

        permanova_sig = permanova_results['p_value'] < 0.05
        nbs_sig = len(nbs_results['significant_components']) > 0

        print(f"\nPERMANOVA: {'✓ SIGNIFICANT' if permanova_sig else '✗ NOT SIGNIFICANT'}")
        print(f"           (p = {permanova_results['p_value']:.4f}, R² = {permanova_results['r_squared']:.3f})")
        print(f"\nNBS:       {'✓ SIGNIFICANT' if nbs_sig else '✗ NOT SIGNIFICANT'}")
        if nbs_sig:
            print(f"           ({len(nbs_results['significant_components'])} component(s) found)")
        print()
        print("-" * 70)

        if permanova_sig and nbs_sig:
            print("✓✓ STRONG CONCLUSION:")
            print("   Groups differ in connectivity, and these differences form")
            print("   identifiable connected subnetworks. This is ideal for NBS!")
            print()
            print("   → Examine the significant NBS components for biological insight")
            print("   → These networks likely reflect disease-related changes")

        elif permanova_sig and not nbs_sig:
            print("~ MODERATE CONCLUSION:")
            print("   Groups differ in OVERALL connectivity (PERMANOVA significant),")
            print("   but differences don't form large connected networks (NBS null).")
            print()
            print("   Possible reasons:")
            print("   • Differences are diffuse (many small changes)")
            print("   • NBS threshold too high (try t=2.5)")
            print("   • Effects are weak but widespread")
            print()
            print("   → Consider edge-by-edge analysis with FDR correction")
            print("   → Try lower NBS threshold for exploratory analysis")

        elif not permanova_sig and nbs_sig:
            print("⚠ UNEXPECTED RESULT:")
            print("   NBS found components but PERMANOVA is not significant.")
            print("   This is unusual and warrants careful inspection.")
            print()
            print("   Possible issues:")
            print("   • False positive in NBS (check p-values carefully)")
            print("   • Small sample size causing instability")
            print("   • Consider increasing permutations")
            print()
            print("   → Interpret with caution")
            print("   → Validate in independent dataset")

        else:
            print("✗ NULL RESULT:")
            print("   No significant connectivity differences detected by either test.")
            print()
            print("   Before concluding 'no effect', consider:")
            print("   • Sample size: Do you have enough power?")
            print("   • Data quality: Check preprocessing and outliers")
            print("   • Effect size: May be too small to detect")
            print("   • Threshold: Try more liberal NBS threshold (exploratory)")
            print()
            print("   → Check statistical power")
            print("   → Review data quality control metrics")

        print("\n" + "=" * 70)

    elif analysis_mode == 'multigroup' and run_permanova:
        print("\n" + "=" * 70)
        print("NEXT STEPS FOR MULTI-GROUP ANALYSIS")
        print("=" * 70)

        permanova_sig = permanova_results['p_value'] < 0.05

        if permanova_sig:
            print("\n✓ PERMANOVA is SIGNIFICANT")
            print(f"  (p = {permanova_results['p_value']:.4f}, R² = {permanova_results['r_squared']:.3f})")
            print()
            print("  At least one group differs from the others.")
            print()
            print("  RECOMMENDED FOLLOW-UP:")
            print("  → Run PAIRWISE comparisons to identify which groups differ")
            print()
            print("  Example code:")
            print("  ```python")
            for i, g1 in enumerate(group_names):
                for g2 in group_names[i + 1:]:
                    print(f"  # Compare {g1} vs {g2}")
                    print(f"  results = run_network_analysis(")
                    print(f"      cohorts,")
                    print(f"      group_a='{g1}',")
                    print(f"      group_b='{g2}',")
                    print(f"      n_perms=5000")
                    print(f"  )")
                    print()
            print("  ```")
        else:
            print("\n✗ PERMANOVA is NOT SIGNIFICANT")
            print(f"  (p = {permanova_results['p_value']:.4f}, R² = {permanova_results['r_squared']:.3f})")
            print()
            print("  No significant connectivity differences across all groups.")
            print()
            print("  Before concluding 'no effect':")
            print("  • Check sample sizes and statistical power")
            print("  • Review data quality and preprocessing")
            print("  • Consider effect sizes are too small to detect")

    return results