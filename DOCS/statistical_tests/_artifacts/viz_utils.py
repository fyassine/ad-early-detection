import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


def visualize_nbs_network(nbs_results, cohorts, group_a='healthy', group_b='ad',
                          component_idx=0, node_names=None, figsize=(20, 12)):
    """
    Visualize the significant network component found by NBS.

    Parameters:
    -----------
    nbs_results : dict
        Results from network_based_statistic()
    cohorts : dict
        Original cohort data
    group_a, group_b : str
        Group names for comparison
    component_idx : int
        Which component to visualize (default: largest = 0)
    node_names : list or None
        Names of brain regions (if available)
    figsize : tuple
        Figure size
    """

    if len(nbs_results['significant_components']) == 0:
        print("No significant components to visualize!")
        return

    # Get the component
    component = nbs_results['significant_components'][component_idx]
    edge_indices = component['edges']
    test_stats = nbs_results['test_statistics'][edge_indices]

    print("=" * 70)
    print(f"VISUALIZING COMPONENT {component_idx + 1}")
    print("=" * 70)
    print(f"Size: {component['size']} edges")
    print(f"P-value: {component['p_value']:.4f}")
    print(f"Mean |t|: {component['mean_stat']:.3f}")
    print()

    # Reconstruct which edges these are
    # Assuming edges are numbered as: (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ...
    edges_a = cohorts[group_a]['edges']
    n_edges_total = edges_a.shape[1]
    n_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges_total)) / 2))

    print(f"Graph structure: {n_nodes} nodes (brain regions)")
    print()

    # Create mapping: edge_index -> (node_i, node_j)
    edge_to_nodes = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_to_nodes.append((i, j))

    # Get the actual node pairs in this component
    component_edges = [edge_to_nodes[idx] for idx in edge_indices]

    # Analyze node involvement
    node_degrees = Counter()
    for i, j in component_edges:
        node_degrees[i] += 1
        node_degrees[j] += 1

    # Get top hubs
    top_hubs = node_degrees.most_common(10)

    print("TOP 10 HUB REGIONS (most connected in this component):")
    print("-" * 70)
    for node_id, degree in top_hubs:
        name = node_names[node_id] if node_names else f"Region {node_id}"
        print(f"  {name:30s} : {degree} connections")
    print()

    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main Network Graph (top, spanning 2 columns)
    ax_network = fig.add_subplot(gs[0:2, 0:2])
    _plot_network_graph(ax_network, component_edges, test_stats, node_degrees,
                        node_names, n_nodes, component_idx)

    # 2. Hub degree distribution (top right)
    ax_degree = fig.add_subplot(gs[0, 2])
    _plot_degree_distribution(ax_degree, node_degrees)

    # 3. Edge strength distribution (middle right)
    ax_strength = fig.add_subplot(gs[1, 2])
    _plot_edge_strength(ax_strength, test_stats, cohorts, group_a, group_b, edge_indices)

    # 4. Connectivity matrix (bottom left)
    ax_matrix = fig.add_subplot(gs[2, 0])
    _plot_connectivity_matrix(ax_matrix, component_edges, n_nodes, test_stats)

    # 5. Hub bar chart (bottom middle)
    ax_hubs = fig.add_subplot(gs[2, 1])
    _plot_hub_barchart(ax_hubs, top_hubs, node_names)

    # 6. Summary stats (bottom right)
    ax_summary = fig.add_subplot(gs[2, 2])
    _plot_summary_stats(ax_summary, component, node_degrees, n_nodes)

    plt.suptitle(f'NBS Network Component {component_idx + 1}: {group_a.upper()} vs {group_b.upper()}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.show()

    return {
        'component_edges': component_edges,
        'node_degrees': node_degrees,
        'top_hubs': top_hubs
    }


def _plot_network_graph(ax, edges, test_stats, node_degrees, node_names, n_nodes, component_idx):
    """Plot the main network graph using networkx"""

    # Create graph
    G = nx.Graph()

    # Get nodes that are actually in this component
    active_nodes = set()
    for i, j in edges:
        active_nodes.add(i)
        active_nodes.add(j)

    G.add_nodes_from(active_nodes)

    # Add edges with weights (absolute t-statistics)
    edge_weights = []
    for (i, j), t_stat in zip(edges, test_stats):
        G.add_edge(i, j, weight=abs(t_stat))
        edge_weights.append(abs(t_stat))

    # Layout
    if len(active_nodes) > 100:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=1.0, iterations=100, seed=42)

    # Node sizes based on degree
    node_sizes = [node_degrees.get(node, 1) * 50 for node in G.nodes()]

    # Node colors based on degree (hub = red, peripheral = blue)
    max_degree = max(node_degrees.values()) if node_degrees else 1
    node_colors = [node_degrees.get(node, 0) / max_degree for node in G.nodes()]

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap='RdYlBu_r',
                                   alpha=0.8,
                                   edgecolors='black',
                                   linewidths=1)

    # Draw edges with varying thickness and color based on t-statistic
    edge_colors = [abs(t) for t in test_stats]
    edges_drawn = nx.draw_networkx_edges(G, pos, ax=ax,
                                         width=[abs(t) / 2 for t in test_stats],
                                         edge_color=edge_colors,
                                         edge_cmap=plt.cm.Reds,
                                         alpha=0.6)

    # Add labels for top hubs only (to avoid clutter)
    top_hub_nodes = sorted(node_degrees.keys(), key=node_degrees.get, reverse=True)[:5]
    labels = {}
    for node in top_hub_nodes:
        if node in active_nodes:
            labels[node] = node_names[node] if node_names else str(node)

    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')

    ax.set_title(f'Network Graph: {len(active_nodes)} regions, {len(edges)} connections',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Add colorbar for node degree
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                               norm=plt.Normalize(vmin=0, vmax=max_degree))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Node Degree\n(# connections)', rotation=270, labelpad=20)


def _plot_degree_distribution(ax, node_degrees):
    """Plot distribution of node degrees"""
    degrees = list(node_degrees.values())

    ax.hist(degrees, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(degrees), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(degrees):.1f}')
    ax.axvline(np.median(degrees), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(degrees):.1f}')

    ax.set_xlabel('Degree (# connections)')
    ax.set_ylabel('# Regions')
    ax.set_title('Hub Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_edge_strength(ax, test_stats, cohorts, group_a, group_b, edge_indices):
    """Plot edge strength differences"""

    # Get actual connectivity values
    edges_a = cohorts[group_a]['edges'][:, edge_indices]
    edges_b = cohorts[group_b]['edges'][:, edge_indices]

    mean_a = np.mean(edges_a, axis=0)
    mean_b = np.mean(edges_b, axis=0)

    # Scatter plot
    ax.scatter(mean_a, mean_b, c=np.abs(test_stats), cmap='Reds',
               alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

    # Diagonal line (no difference)
    lims = [min(mean_a.min(), mean_b.min()), max(mean_a.max(), mean_b.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='No difference')

    ax.set_xlabel(f'{group_a.capitalize()} (mean connectivity)')
    ax.set_ylabel(f'{group_b.capitalize()} (mean connectivity)')
    ax.set_title('Edge Strength Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds',
                               norm=plt.Normalize(vmin=0, vmax=np.abs(test_stats).max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='|t-statistic|')


def _plot_connectivity_matrix(ax, edges, n_nodes, test_stats):
    """Plot connectivity matrix of the component"""

    # Create adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))

    for (i, j), t_stat in zip(edges, test_stats):
        adj_matrix[i, j] = t_stat
        adj_matrix[j, i] = t_stat

    # Only show nodes that are in the component
    active_nodes = sorted(set([i for i, j in edges] + [j for i, j in edges]))

    # Extract submatrix
    sub_matrix = adj_matrix[np.ix_(active_nodes, active_nodes)]

    # Plot
    im = ax.imshow(sub_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(test_stats).max(), vmax=np.abs(test_stats).max())

    ax.set_xlabel('Region Index')
    ax.set_ylabel('Region Index')
    ax.set_title('Connectivity Matrix', fontweight='bold')

    # Colorbar
    plt.colorbar(im, ax=ax, label='t-statistic')


def _plot_hub_barchart(ax, top_hubs, node_names):
    """Plot bar chart of top hubs"""

    nodes = [node_names[node] if node_names else f"R{node}" for node, _ in top_hubs]
    degrees = [degree for _, degree in top_hubs]

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(nodes)))

    bars = ax.barh(range(len(nodes)), degrees, color=colors, edgecolor='black')
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_xlabel('Degree')
    ax.set_title('Top 10 Hub Regions', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)


def _plot_summary_stats(ax, component, node_degrees, n_nodes):
    """Plot summary statistics"""

    ax.axis('off')

    summary_text = f"""
COMPONENT STATISTICS
{'=' * 30}

Network Properties:
  • Total edges: {component['size']}
  • Total nodes: {len(node_degrees)}
  • Network density: {2 * component['size'] / (len(node_degrees) * (len(node_degrees) - 1)):.3f}

Statistical Significance:
  • P-value: {component['p_value']:.4f}
  • Mean |t|: {component['mean_stat']:.3f}
  • Max |t|: {component['max_stat']:.3f}

Hub Statistics:
  • Max degree: {max(node_degrees.values())}
  • Mean degree: {np.mean(list(node_degrees.values())):.1f}
  • Median degree: {np.median(list(node_degrees.values())):.1f}

Network Coverage:
  • % of all nodes: {100 * len(node_degrees) / n_nodes:.1f}%
  • % of edges: {100 * component['size'] / (n_nodes * (n_nodes - 1) / 2):.1f}%
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


# Simpler circular visualization for large networks
def visualize_nbs_circular(nbs_results, cohorts, group_a='healthy', group_b='ad',
                           component_idx=0, node_names=None, figsize=(15, 15)):
    """
    Circular/chord diagram visualization for large networks.
    Better for networks with many nodes.
    """

    if len(nbs_results['significant_components']) == 0:
        print("No significant components to visualize!")
        return

    component = nbs_results['significant_components'][component_idx]
    edge_indices = component['edges']
    test_stats = nbs_results['test_statistics'][edge_indices]

    # Reconstruct edges
    edges_a = cohorts[group_a]['edges']
    n_edges_total = edges_a.shape[1]
    n_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges_total)) / 2))

    edge_to_nodes = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_to_nodes.append((i, j))

    component_edges = [edge_to_nodes[idx] for idx in edge_indices]

    # Get active nodes
    active_nodes = sorted(set([i for i, j in component_edges] + [j for i, j in component_edges]))

    # Count degrees
    node_degrees = Counter()
    for i, j in component_edges:
        node_degrees[i] += 1
        node_degrees[j] += 1

    # Create circular plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Position nodes in circle
    n_active = len(active_nodes)
    angles = np.linspace(0, 2 * np.pi, n_active, endpoint=False)
    node_to_angle = {node: angle for node, angle in zip(active_nodes, angles)}

    # Draw edges
    for (i, j), t_stat in zip(component_edges, test_stats):
        if i in node_to_angle and j in node_to_angle:
            angle_i = node_to_angle[i]
            angle_j = node_to_angle[j]

            # Draw arc
            color_intensity = abs(t_stat) / max(abs(test_stats))
            ax.plot([angle_i, angle_j], [1, 1],
                    color=plt.cm.Reds(color_intensity),
                    alpha=0.3, linewidth=0.5)

    # Draw nodes
    for node in active_nodes:
        angle = node_to_angle[node]
        size = node_degrees[node] * 50
        color_val = node_degrees[node] / max(node_degrees.values())

        ax.scatter(angle, 1, s=size, c=[color_val], cmap='RdYlBu_r',
                   edgecolors='black', linewidths=1, alpha=0.8, zorder=10)

        # Add label for top hubs
        if node_degrees[node] >= sorted(node_degrees.values())[-10]:
            label = node_names[node] if node_names else f"R{node}"
            ax.text(angle, 1.15, label, rotation=np.degrees(angle) - 90,
                    ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_ylim(0, 1.3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    plt.title(f'NBS Network Component {component_idx + 1}: Circular View\n' +
              f'{len(active_nodes)} regions, {len(component_edges)} connections',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()


# Quick summary function
def nbs_network_summary(nbs_results, component_idx=0):
    """Print a quick text summary of the network"""

    if len(nbs_results['significant_components']) == 0:
        print("No significant components found!")
        return

    component = nbs_results['significant_components'][component_idx]
    edge_indices = component['edges']

    print("=" * 70)
    print(f"NBS COMPONENT {component_idx + 1} SUMMARY")
    print("=" * 70)
    print(f"Size: {component['size']} edges")
    print(f"P-value: {component['p_value']:.6f}")
    print(f"Mean |t-statistic|: {component['mean_stat']:.3f}")
    print(f"Max |t-statistic|: {component['max_stat']:.3f}")
    print()
    print("This component represents a connected subnetwork of brain regions")
    print("showing significant connectivity differences between groups.")
    print("=" * 70)