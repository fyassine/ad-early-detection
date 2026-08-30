import pandas as pd
import numpy as np

def view_significant_edges(results, roi_names=None, component_idx=0):
    """
    Decodes NBS results to show exactly which brain regions are connected.

    Parameters:
    -----------
    results : dict
        The output from run_network_analysis()
    roi_names : list of str (Optional)
        List of region names corresponding to your nodes.
        If None, will use "Node 0", "Node 1", etc.
    component_idx : int
        Which significant component to view (default 0 = the largest/most significant)
    """

    if 'nbs' not in results or not results['nbs']['significant_components']:
        print("No significant NBS components found to view.")
        return None

    nbs_data = results['nbs']
    component = nbs_data['significant_components'][component_idx]

    edge_indices = component['edges']
    test_stats = nbs_data['test_statistics'][edge_indices]


    n_edges_total = len(nbs_data['test_statistics'])
    n_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges_total)) / 2))

    rows, cols = np.triu_indices(n_nodes, k=1)

    comp_rows = rows[edge_indices]
    comp_cols = cols[edge_indices]

    edge_list = []

    for i in range(len(edge_indices)):
        idx_a = comp_rows[i]
        idx_b = comp_cols[i]
        t_stat = test_stats[i]

        name_a = roi_names[idx_a] if roi_names is not None else f"Node {idx_a}"
        name_b = roi_names[idx_b] if roi_names is not None else f"Node {idx_b}"

        edge_list.append({
            'Region A': name_a,
            'Region B': name_b,
            'T-Statistic': t_stat,
            'Abs(T-Stat)': abs(t_stat)
        })

    df = pd.DataFrame(edge_list)
    df = df.sort_values('Abs(T-Stat)', ascending=False).reset_index(drop=True)

    print(f"--- ANATOMY OF COMPONENT {component_idx + 1} ({len(df)} edges) ---")
    print(f"P-value: {component['p_value']:.4f}")
    return df

roi_labels = [
    # --- LEFT HEMISPHERE (55 Regions) ---
    # Cortical (1-48 in H-O Atlas)
    'L_Frontal_Pole', 'L_Insula', 'L_Superior_Frontal_Gyrus', 'L_Middle_Frontal_Gyrus', 'L_Inferior_Frontal_Gyrus_Pars_Triangularis',
    'L_Inferior_Frontal_Gyrus_Pars_Opercularis', 'L_Precentral_Gyrus', 'L_Temporal_Pole', 'L_Superior_Temporal_Gyrus_anterior', 'L_Superior_Temporal_Gyrus_posterior',
    'L_Middle_Temporal_Gyrus_anterior', 'L_Middle_Temporal_Gyrus_posterior', 'L_Middle_Temporal_Gyrus_temporooccipital', 'L_Inferior_Temporal_Gyrus_anterior', 'L_Inferior_Temporal_Gyrus_posterior',
    'L_Inferior_Temporal_Gyrus_temporooccipital', 'L_Postcentral_Gyrus', 'L_Superior_Parietal_Lobule', 'L_Supramarginal_Gyrus_anterior', 'L_Supramarginal_Gyrus_posterior',
    'L_Angular_Gyrus', 'L_Lateral_Occipital_Cortex_superior', 'L_Lateral_Occipital_Cortex_inferior', 'L_Intracalcarine_Cortex', 'L_Frontal_Medial_Cortex',
    'L_Juxtapositional_Lobule_SMA', 'L_Subcallosal_Cortex', 'L_Paracingulate_Gyrus', 'L_Cingulate_Gyrus_anterior', 'L_Cingulate_Gyrus_posterior',
    'L_Precuneus', 'L_Cuneal_Cortex', 'L_Frontal_Orbital_Cortex', 'L_Parahippocampal_Gyrus_anterior', 'L_Parahippocampal_Gyrus_posterior',
    'L_Lingual_Gyrus', 'L_Temporal_Fusiform_Cortex_anterior', 'L_Temporal_Fusiform_Cortex_posterior', 'L_Temporal_Occipital_Fusiform_Cortex', 'L_Occipital_Fusiform_Gyrus',
    'L_Frontal_Operculum_Cortex', 'L_Central_Opercular_Cortex', 'L_Parietal_Operculum_Cortex', 'L_Planum_Polare', 'L_Heschls_Gyrus',
    'L_Planum_Temporale', 'L_Supracalcarine_Cortex', 'L_Occipital_Pole',
    # Subcortical (49-55 in H-O Atlas)
    'L_Thalamus', 'L_Caudate', 'L_Putamen', 'L_Pallidum', 'L_Hippocampus', 'L_Amygdala', 'L_Accumbens',

    # --- RIGHT HEMISPHERE (55 Regions) ---
    # Cortical
    'R_Frontal_Pole', 'R_Insula', 'R_Superior_Frontal_Gyrus', 'R_Middle_Frontal_Gyrus', 'R_Inferior_Frontal_Gyrus_Pars_Triangularis',
    'R_Inferior_Frontal_Gyrus_Pars_Opercularis', 'R_Precentral_Gyrus', 'R_Temporal_Pole', 'R_Superior_Temporal_Gyrus_anterior', 'R_Superior_Temporal_Gyrus_posterior',
    'R_Middle_Temporal_Gyrus_anterior', 'R_Middle_Temporal_Gyrus_posterior', 'R_Middle_Temporal_Gyrus_temporooccipital', 'R_Inferior_Temporal_Gyrus_anterior', 'R_Inferior_Temporal_Gyrus_posterior',
    'R_Inferior_Temporal_Gyrus_temporooccipital', 'R_Postcentral_Gyrus', 'R_Superior_Parietal_Lobule', 'R_Supramarginal_Gyrus_anterior', 'R_Supramarginal_Gyrus_posterior',
    'R_Angular_Gyrus', 'R_Lateral_Occipital_Cortex_superior', 'R_Lateral_Occipital_Cortex_inferior', 'R_Intracalcarine_Cortex', 'R_Frontal_Medial_Cortex',
    'R_Juxtapositional_Lobule_SMA', 'R_Subcallosal_Cortex', 'R_Paracingulate_Gyrus', 'R_Cingulate_Gyrus_anterior', 'R_Cingulate_Gyrus_posterior',
    'R_Precuneus', 'R_Cuneal_Cortex', 'R_Frontal_Orbital_Cortex', 'R_Parahippocampal_Gyrus_anterior', 'R_Parahippocampal_Gyrus_posterior',
    'R_Lingual_Gyrus', 'R_Temporal_Fusiform_Cortex_anterior', 'R_Temporal_Fusiform_Cortex_posterior', 'R_Temporal_Occipital_Fusiform_Cortex', 'R_Occipital_Fusiform_Gyrus',
    'R_Frontal_Operculum_Cortex', 'R_Central_Opercular_Cortex', 'R_Parietal_Operculum_Cortex', 'R_Planum_Polare', 'R_Heschls_Gyrus',
    'R_Planum_Temporale', 'R_Supracalcarine_Cortex', 'R_Occipital_Pole',
    # Subcortical
    'R_Thalamus', 'R_Caudate', 'R_Putamen', 'R_Pallidum', 'R_Hippocampus', 'R_Amygdala', 'R_Accumbens'
]