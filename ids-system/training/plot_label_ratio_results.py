"""
plot_label_ratio_results.py — Generate publication-quality charts
for the label ratio sensitivity analysis.

Usage:
    python plot_label_ratio_results.py
    python plot_label_ratio_results.py --results_csv logs/label_ratio_experiments/label_ratio_results.csv
"""
import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication-quality dark theme
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

# Color palette
COLORS = {
    'dos': '#00d4ff',
    'intrusion': '#ff6b8a',
    'port_scan': '#4ade80',
}

MARKERS = {
    'dos': 'o',
    'intrusion': 's',
    'port_scan': 'D',
}


def load_results_csv(csv_path):
    """Load results from CSV file."""
    import csv
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'Dataset': row['Dataset'],
                'Task': row['Task'],
                'LabelRatio': int(row['LabelRatio']),
                'Accuracy': float(row['Accuracy']),
                'F1_Weighted': float(row['F1_Weighted']),
                'ROC_AUC': float(row['ROC_AUC']),
                'PR_AUC': float(row['PR_AUC']),
            })
    return results


def load_results_json(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_metric_vs_ratio(results, dataset, metric_key, metric_label, save_path):
    """Plot a single metric vs label ratio for one dataset."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    tasks = sorted(set(r['Task'] for r in results if r['Dataset'] == dataset))

    for task in tasks:
        task_data = [r for r in results if r['Dataset'] == dataset and r['Task'] == task]
        task_data.sort(key=lambda x: x['LabelRatio'])

        ratios = [r['LabelRatio'] for r in task_data]
        values = [r[metric_key] for r in task_data]

        ax.plot(ratios, values,
                color=COLORS.get(task, '#ffffff'),
                marker=MARKERS.get(task, 'o'),
                linewidth=2.5, markersize=8,
                label=task.replace('_', ' ').title(),
                zorder=5)

        # Add value annotations
        for x, y in zip(ratios, values):
            ax.annotate(f'{y:.3f}', (x, y),
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=8, color='#8b949e')

    ax.set_xlabel('Labeled Data Ratio (%)', fontweight='bold')
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_title(f'{metric_label} vs. Labeled Data Ratio — {dataset.upper()}',
                fontweight='bold', fontsize=14)
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(['5%', '10%', '20%', '50%', '100%'])
    ax.legend(loc='lower right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05), top=min(1.05, ax.get_ylim()[1] + 0.05))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_dashboard(results, dataset, save_path):
    """Plot a 2x2 dashboard with all metrics for one dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Label Ratio Sensitivity Analysis — {dataset.upper()}',
                fontsize=16, fontweight='bold', y=0.98)

    metrics = [
        ('Accuracy', 'Accuracy'),
        ('F1_Weighted', 'F1 Score (Weighted)'),
        ('ROC_AUC', 'ROC-AUC'),
        ('PR_AUC', 'PR-AUC'),
    ]

    tasks = sorted(set(r['Task'] for r in results if r['Dataset'] == dataset))

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]

        for task in tasks:
            task_data = [r for r in results if r['Dataset'] == dataset and r['Task'] == task]
            task_data.sort(key=lambda x: x['LabelRatio'])

            ratios = [r['LabelRatio'] for r in task_data]
            values = [r[metric_key] for r in task_data]

            ax.plot(ratios, values,
                    color=COLORS.get(task, '#ffffff'),
                    marker=MARKERS.get(task, 'o'),
                    linewidth=2.5, markersize=7,
                    label=task.replace('_', ' ').title(),
                    zorder=5)

            # Value annotations
            for x, y in zip(ratios, values):
                ax.annotate(f'{y:.2f}', (x, y),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=7, color='#8b949e')

        ax.set_xlabel('Labeled Data (%)')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label, fontweight='bold')
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(['5%', '10%', '20%', '50%', '100%'])
        ax.legend(loc='lower right', framealpha=0.8, fontsize=8,
                 facecolor='#161b22', edgecolor='#30363d')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_latex_table(results, save_path):
    """Generate a LaTeX table for the paper."""
    datasets = sorted(set(r['Dataset'] for r in results))
    ratios = sorted(set(r['LabelRatio'] for r in results))
    tasks = sorted(set(r['Task'] for r in results))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance comparison across different labeled data ratios.}")
    lines.append(r"\label{tab:label_ratio}")

    col_spec = "|l|l|" + "|".join(["cc"] * len(ratios)) + "|"
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")

    # Header row
    header = r"Dataset & Task"
    for r in ratios:
        header += f" & \\multicolumn{{2}}{{c|}}{{{r}\%}}"
    header += r" \\"
    lines.append(header)

    sub_header = r" & "
    for _ in ratios:
        sub_header += r" & F1 & AUC"
    sub_header += r" \\"
    lines.append(sub_header)
    lines.append(r"\hline")

    for dataset in datasets:
        for task in tasks:
            row = f"{dataset.upper()} & {task.replace('_', ' ').title()}"
            for ratio in ratios:
                match = [r2 for r2 in results
                        if r2['Dataset'] == dataset and r2['Task'] == task and r2['LabelRatio'] == ratio]
                if match:
                    m = match[0]
                    row += f" & {m['F1_Weighted']:.3f} & {m['ROC_AUC']:.3f}"
                else:
                    row += r" & - & -"
            row += r" \\"
            lines.append(row)
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot label ratio experiment results")
    parser.add_argument("--results_csv", type=str,
                       default="logs/label_ratio_experiments/label_ratio_results.csv",
                       help="Path to the results CSV file")
    parser.add_argument("--output_dir", type=str,
                       default="logs/label_ratio_experiments/plots",
                       help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"Results file not found: {args.results_csv}")
        print("Run the experiments first: .\\run_label_ratio_experiments.ps1")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results_csv(args.results_csv)

    print(f"Loaded {len(results)} result entries.")
    datasets = sorted(set(r['Dataset'] for r in results))

    for dataset in datasets:
        print(f"\nGenerating plots for {dataset}...")

        # Individual metric plots
        for metric_key, metric_label in [
            ('F1_Weighted', 'F1 Score (Weighted)'),
            ('ROC_AUC', 'ROC-AUC'),
            ('PR_AUC', 'PR-AUC'),
            ('Accuracy', 'Accuracy'),
        ]:
            save_path = os.path.join(args.output_dir,
                                    f"{dataset}_{metric_key.lower()}_vs_ratio.png")
            plot_metric_vs_ratio(results, dataset, metric_key, metric_label, save_path)

        # Combined dashboard
        dashboard_path = os.path.join(args.output_dir, f"{dataset}_ratio_dashboard.png")
        plot_combined_dashboard(results, dataset, dashboard_path)

    # LaTeX table
    latex_path = os.path.join(args.output_dir, "label_ratio_table.tex")
    generate_latex_table(results, latex_path)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
