import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
from typing import Dict, Any, List, Tuple

def analyze_group_differences(group1_data, group2_data, label, alpha=0.05):
    n_edges = group1_data.shape[1]
    pvals = []
    stats = []

    print(f"Running Mann-Whitney U tests for {label}...")

    for i in range(n_edges):
        stat, pval = mannwhitneyu(
            group1_data[:, i],
            group2_data[:, i],
            alternative='two-sided',
        )
        pvals.append(pval)
        stats.append(stat)

    pvals = np.array(pvals)

    _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')

    sig_count = np.sum(pvals_fdr < alpha)
    percentage = 100 * sig_count / n_edges

    print(f"Significant edges (FDR < {alpha}) {label}: {sig_count} / {n_edges} ({percentage:.2f}%)")

    return pvals_fdr, stats, sig_count


def run_diagnostic_tests(cohorts: dict, group_a: str = 'healthy', group_b: str = 'ad') -> None:
    """
    Performs robust diagnostic statistical tests comparing two specific cohorts.
    Includes FDR correction and improved variability metrics.
    """

    # --- VALIDATION ---
    if group_a not in cohorts or group_b not in cohorts:
        print(f"Error: One or both groups ('{group_a}', '{group_b}') not found.")
        return

    name_a, name_b = cohorts[group_a]['name'], cohorts[group_b]['name']
    edges_a, edges_b = cohorts[group_a]['edges'], cohorts[group_b]['edges']

    # Basic params
    n_a, n_b = len(cohorts[group_a]['ids']), len(cohorts[group_b]['ids'])
    n_edges = edges_a.shape[1]

    print(f"=== DIAGNOSTIC REPORT: {name_a} vs {name_b} ===")

    # --- 1. SAMPLE SIZE & POWER ---
    print("\n=== 1. SAMPLE SIZE & POWER CHECK ===")
    print(f"{name_a}: {n_a} subjects")
    print(f"{name_b}: {n_b} subjects")

    # Simple power approximation (Lehr's formula for 80% power, alpha=0.05)
    # n_per_group approx 16 / (effect_size^2)
    required_n_medium = 16 / (0.5 ** 2)  # ~64 total (32 per group)
    required_n_large = 16 / (0.8 ** 2)  # ~25 total (12.5 per group)

    print("-" * 40)
    print("Power Estimation (Rule of Thumb):")
    if min(n_a, n_b) < 12:
        print("  [!] CRITICAL: Sample size likely too small even for large effects.")
    elif min(n_a, n_b) < 32:
        print("  [~] CAUTION: Powered only for LARGE effects (d > 0.8).")
    else:
        print("  [✓] GOOD: Powered for medium effects (d > 0.5).")

    # --- 2. VARIABILITY (Standard Deviation) ---
    print("\n=== 2. WITHIN-GROUP VARIABILITY (Standard Deviation) ===")
    print("Note: Using SD instead of CV because correlations can be near zero.")
    print("-" * 40)

    for name, data in [(name_a, edges_a), (name_b, edges_b)]:
        avg_std = np.mean(np.std(data, axis=0))
        print(f"{name}: Avg Edge SD = {avg_std:.3f}")

    print("Interpretation: Lower SD is better. Typical fMRI correlation SD is 0.10-0.20.")

    # --- 3. STATISTICAL TESTS & CORRECTION ---
    print(f"\n=== 3. SIGNAL DETECTION ({n_edges} edges) ===")

    # Calculate uncorrected p-values
    pvals = []
    for i in range(n_edges):
        _, p = mannwhitneyu(edges_a[:, i], edges_b[:, i], alternative='two-sided')
        pvals.append(p)
    pvals = np.array(pvals)

    # FDR Correction (Benjamini-Hochberg)
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    sig_unc = np.sum(pvals < 0.05)
    sig_fdr = np.sum(pvals_fdr < 0.05)

    print(f"Uncorrected (p < 0.05): {sig_unc} edges ({100 * sig_unc / n_edges:.1f}%)")
    print(f"FDR Corrected (q < 0.05): {sig_fdr} edges ({100 * sig_fdr / n_edges:.1f}%)")

    print("-" * 40)
    if sig_fdr > 0:
        print(f"  [✓] STRONG SIGNAL: {sig_fdr} edges survive multiple comparison correction.")
    elif sig_unc > (0.05 * n_edges):
        print("  [~] WEAK SIGNAL: Excess of low p-values, but none survive FDR.")
    else:
        print("  [!] NO SIGNAL: Results consistent with random noise.")

    # --- 4. EFFECT SIZE DISTRIBUTION ---
    print("\n=== 4. EFFECT SIZE DISTRIBUTION (Cohen's d) ===")

    # Vectorized Cohen's d calculation
    mean_a, mean_b = np.mean(edges_a, axis=0), np.mean(edges_b, axis=0)
    var_a, var_b = np.var(edges_a, axis=0, ddof=1), np.var(edges_b, axis=0, ddof=1)
    pooled_sd = np.sqrt((var_a + var_b) / 2)
    cohens_d = (mean_a - mean_b) / pooled_sd
    abs_d = np.abs(cohens_d)

    med_d = np.median(abs_d)
    pct_medium = np.sum(abs_d > 0.5) / n_edges * 100
    pct_large = np.sum(abs_d > 0.8) / n_edges * 100

    print(f"Median Effect Size: |d| = {med_d:.3f}")
    print(f"Edges with Medium Effect (>0.5): {pct_medium:.1f}%")
    print(f"Edges with Large Effect  (>0.8): {pct_large:.1f}%")

    # --- 5. VISUALIZATION ---
    print("\n=== 5. VISUALIZATION ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # P-value Histogram
    axes[0].hist(pvals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axhline(y=n_edges / 50, color='r', linestyle='--', label='Null (Uniform)')
    axes[0].set_title('P-value Distribution')
    axes[0].set_xlabel('Uncorrected P-value')
    axes[0].legend()

    # Effect Size Histogram
    axes[1].hist(cohens_d, bins=50, edgecolor='black', color='orange', alpha=0.7)
    axes[1].axvline(x=0, color='k', linestyle='--')
    axes[1].set_title("Effect Size Distribution (Cohen's d)")
    axes[1].set_xlabel("d (Negative = Group B higher)")

    plt.tight_layout()
    plt.show()

    print("\nNOTE on Independence: Mann-Whitney assumes edges are independent, which")
    print("brain networks are NOT. P-values may be slightly optimistic.")


def _run_diagnostic_tests_pairwise(cohorts: dict, group_a: str, group_b: str, plot: bool = True) -> Dict[str, Any]:
    """
    Performs diagnostic tests and returns a dictionary of metrics for automated evaluation.
    """
    # Validation
    if group_a not in cohorts or group_b not in cohorts:
        print(f"Error: Missing groups {group_a} or {group_b}")
        return {'valid': False}

    name_a, name_b = cohorts[group_a]['name'], cohorts[group_b]['name']
    edges_a, edges_b = cohorts[group_a]['edges'], cohorts[group_b]['edges']
    n_a, n_b = len(cohorts[group_a]['ids']), len(cohorts[group_b]['ids'])
    n_edges = edges_a.shape[1]

    print(f"\n{'=' * 60}")
    print(f"=== DIAGNOSTIC REPORT: {name_a} vs {name_b} ===")
    print(f"{'=' * 60}")

    # 1. Sample Size
    print(f"Sample Sizes: {name_a}={n_a}, {name_b}={n_b}")
    min_n = min(n_a, n_b)

    # 2. Variability (SD)
    avg_sd_a = np.mean(np.std(edges_a, axis=0))
    avg_sd_b = np.mean(np.std(edges_b, axis=0))
    print(f"Avg Edge SD: {name_a}={avg_sd_a:.3f}, {name_b}={avg_sd_b:.3f}")

    # 3. Statistical Tests (Mann-Whitney U)
    pvals = []
    for i in range(n_edges):
        _, p = mannwhitneyu(edges_a[:, i], edges_b[:, i], alternative='two-sided')
        pvals.append(p)
    pvals = np.array(pvals)

    # FDR Correction
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    sig_unc = np.sum(pvals < 0.05)
    sig_fdr = np.sum(pvals_fdr < 0.05)

    print(f"Signal: {sig_fdr} edges survive FDR ({(sig_fdr / n_edges) * 100:.1f}%)")

    # 4. Effect Size (Cohen's d)
    mean_a, mean_b = np.mean(edges_a, axis=0), np.mean(edges_b, axis=0)
    var_a, var_b = np.var(edges_a, axis=0, ddof=1), np.var(edges_b, axis=0, ddof=1)
    pooled_sd = np.sqrt((var_a + var_b) / 2)
    pooled_sd[pooled_sd == 0] = 1e-9  # Avoid div/0

    cohens_d = (mean_a - mean_b) / pooled_sd
    abs_d = np.abs(cohens_d)

    med_d = np.median(abs_d)
    pct_medium = np.sum(abs_d > 0.5) / n_edges * 100

    print(f"Effect Size: Median |d|={med_d:.3f}, Edges >0.5={pct_medium:.1f}%")

    # 5. Visualization
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(pvals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'P-values ({name_a} vs {name_b})')
        axes[1].hist(cohens_d, bins=50, edgecolor='black', color='orange', alpha=0.7)
        axes[1].set_title("Effect Sizes (Cohen's d)")
        plt.tight_layout()
        plt.show()

    # Return metrics for the batch function to check
    return {
        'valid': True,
        'pair_name': f"{name_a} vs {name_b}",
        'min_sample_size': min_n,
        'sig_fdr_count': sig_fdr,
        'sig_fdr_pct': (sig_fdr / n_edges) * 100,
        'median_effect_size': med_d,
        'pct_medium_effect': pct_medium
    }


def run_batch_diagnostics(cohorts: dict, comparisons: List[Tuple[str, str]]):
    """
    Runs diagnostic tests for all pairs and flags potential ML problems.
    """
    print("Starting Batch Diagnostic Tests...\n")

    results = []

    for group_a, group_b in comparisons:
        # Run the test (disable plot if you want it faster/cleaner logs)
        metrics = _run_diagnostic_tests_pairwise(cohorts, group_a, group_b, plot=True)

        if not metrics['valid']:
            continue

        # --- AUTO-FLAGGING LOGIC ---
        flags = []
        status = "SAFE"

        # Flag 1: Low Sample Size
        if metrics['min_sample_size'] < 20:
            flags.append("CRITICAL: Low Sample Size (<20)")
            status = "HIGH RISK"
        elif metrics['min_sample_size'] < 30:
            flags.append("Warning: Small Sample Size (<30)")
            if status == "SAFE": status = "CAUTION"

        # Flag 2: No Signal (FDR)
        if metrics['sig_fdr_count'] == 0:
            flags.append("CRITICAL: No Signal (0 FDR edges)")
            status = "HIGH RISK"
        elif metrics['sig_fdr_pct'] < 1.0:
            flags.append("Warning: Weak Signal (<1% FDR edges)")
            if status == "SAFE": status = "CAUTION"

        # Flag 3: Weak Effect Sizes
        if metrics['pct_medium_effect'] < 0.5:
            flags.append("Warning: Tiny Effects (Few edges > 0.5)")
            if status == "SAFE": status = "CAUTION"

        results.append({
            'Pair': metrics['pair_name'],
            'Status': status,
            'Flags': "; ".join(flags) if flags else "None",
            'FDR_Edges': metrics['sig_fdr_count'],
            'Med_d': metrics['median_effect_size']
        })

    # --- PRINT SUMMARY TABLE ---
    print("\n" + "=" * 80)
    print("FINAL MACHINE LEARNING RISK ASSESSMENT")
    print("=" * 80)
    print(f"{'PAIR':<25} | {'STATUS':<10} | {'FDR EDGES':<10} | {'FLAGS'}")
    print("-" * 80)

    for r in results:
        # Color coding (optional/rudimentary)
        marker = "[!]" if r['Status'] == "HIGH RISK" else "[~]" if r['Status'] == "CAUTION" else "[✓]"
        print(f"{marker} {r['Pair']:<21} | {r['Status']:<10} | {r['FDR_Edges']:<10} | {r['Flags']}")
    print("-" * 80)
    print("Guide:")
    print("  HIGH RISK: Model will likely overfit or fail completely.")
    print("  CAUTION:   Model needs careful feature selection (e.g., top 1-5% features).")
    print("  SAFE:      Data has strong signal; standard ML approaches should work.")
