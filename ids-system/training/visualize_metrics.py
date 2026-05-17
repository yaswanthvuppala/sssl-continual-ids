"""
visualize_metrics.py — Generate publication-quality visualization plots for the IDS pipeline.

Usage:
    python training/visualize_metrics.py
    python training/visualize_metrics.py --task intrusion
"""
import os
import sys
import json
import argparse
import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Style Configuration ---
PLOT_DIR = "./logs/plots"
STYLE = {
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#3a3a5c",
    "axes.labelcolor": "#e0e0ff",
    "text.color": "#e0e0ff",
    "xtick.color": "#b0b0d0",
    "ytick.color": "#b0b0d0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
}
PALETTE = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8", "#ff922b"]


def _apply_style():
    plt.rcParams.update(STYLE)
    sns.set_palette(PALETTE)


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════
# 1. CL Metrics Dashboard
# ═══════════════════════════════════════════════════════════
def plot_cl_metrics_dashboard(task_names=None):
    """Training loss curves + mask rate + per-task evaluation bar chart."""
    print("\n[1/4] Generating CL Metrics Dashboard...")
    _apply_style()

    if task_names is None:
        task_names = []
        for d in sorted(os.listdir("./logs")):
            if d.startswith("task_") and os.path.isdir(f"./logs/{d}"):
                task_names.append(d.replace("task_", ""))
    if not task_names:
        print("  No task training logs found. Skipping.")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Training Loss Curves ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(task_names):
        hp = f"./logs/task_{t}/training_history.json"
        if not os.path.exists(hp):
            continue
        h = json.load(open(hp))
        epochs = range(1, len(h["total_loss"]) + 1)
        ax1.plot(epochs, h["total_loss"], '-o', color=PALETTE[i % len(PALETTE)],
                 label=f"{t} (total)", markersize=4, linewidth=2)
        ax1.plot(epochs, h["loss_s"], '--', color=PALETTE[i % len(PALETTE)],
                 alpha=0.6, linewidth=1.2, label=f"{t} (supervised)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curves", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.3)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Panel 2: Mask Rate ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i, t in enumerate(task_names):
        hp = f"./logs/task_{t}/training_history.json"
        if not os.path.exists(hp):
            continue
        h = json.load(open(hp))
        epochs = range(1, len(h["mask_rate"]) + 1)
        ax2.plot(epochs, h["mask_rate"], '-s', color=PALETTE[i % len(PALETTE)],
                 label=t, markersize=4, linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mask Rate")
    ax2.set_title("Pseudo-Label Acceptance Rate", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # --- Panel 3: Per-task Evaluation Metrics (bar chart) ---
    ax3 = fig.add_subplot(gs[1, 0])
    metric_keys = ["accuracy", "recall_weighted", "f1_weighted", "precision_weighted"]
    metric_labels = ["Accuracy", "Recall", "F1", "Precision"]
    bar_data = {}
    for t in task_names:
        mp = f"./logs/eval/metrics_{t}.json"
        if os.path.exists(mp):
            bar_data[t] = json.load(open(mp))
    if bar_data:
        x = np.arange(len(metric_labels))
        width = 0.8 / max(1, len(bar_data))
        for i, (t, m) in enumerate(bar_data.items()):
            vals = [m.get(k, 0) for k in metric_keys]
            bars = ax3.bar(x + i * width, vals, width, color=PALETTE[i % len(PALETTE)],
                           label=t, alpha=0.85, edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{v:.2f}", ha='center', va='bottom', fontsize=8, color='#e0e0ff')
        ax3.set_xticks(x + width * (len(bar_data) - 1) / 2)
        ax3.set_xticklabels(metric_labels)
        ax3.set_ylim(0, 1.15)
        ax3.set_title("Per-Task Evaluation Metrics", fontsize=13, fontweight="bold")
        ax3.legend(fontsize=9, framealpha=0.3)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

    # --- Panel 4: Per-class Recall Comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (t, m) in enumerate(bar_data.items()):
        rpc = m.get("recall_per_class", [])
        if rpc:
            classes = [f"Class {j}" for j in range(len(rpc))]
            x = np.arange(len(classes))
            ax4.bar(x + i * 0.35, rpc, 0.3, color=PALETTE[i % len(PALETTE)],
                    label=t, alpha=0.85, edgecolor='white', linewidth=0.5)
            for j, v in enumerate(rpc):
                ax4.text(x[j] + i * 0.35, v + 0.01, f"{v:.2f}",
                         ha='center', fontsize=9, color='#e0e0ff')
    ax4.set_xticks(np.arange(2) + 0.15)
    ax4.set_xticklabels(["Class 0 (Normal)", "Class 1 (Attack)"])
    ax4.set_ylim(0, 1.15)
    ax4.set_title("Per-Class Recall", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9, framealpha=0.3)
    ax4.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.suptitle("Continual Learning — Metrics Dashboard", fontsize=16,
                 fontweight="bold", color="#00d4ff", y=0.98)
    _save(fig, "cl_metrics_dashboard.png")


# ═══════════════════════════════════════════════════════════
# 2. Memory Level Hierarchy
# ═══════════════════════════════════════════════════════════
def plot_memory_hierarchy():
    """GPM basis dimensionality, cumulative coverage, SVD spectrum."""
    print("\n[2/4] Generating Memory Hierarchy Plot...")
    _apply_style()

    bank_path = "./checkpoints/gpm/memory_bank.pkl"
    if not os.path.exists(bank_path):
        # If no GPM bank exists, create an informational summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.6, "GPM Memory Bank Not Found", ha='center', va='center',
                fontsize=18, color='#ff6b6b', fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.4, "Train multiple tasks (dos, port_scan) to populate the\n"
                "Gradient Projection Memory hierarchy.", ha='center', va='center',
                fontsize=12, color='#b0b0d0', transform=ax.transAxes)
        ax.text(0.5, 0.2, "Single-task (intrusion) training does not use GPM.\n"
                "Showing placeholder for future multi-task continual learning.",
                ha='center', va='center', fontsize=10, color='#808090', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.suptitle("GPM Memory Level Hierarchy", fontsize=16,
                     fontweight="bold", color="#00d4ff")
        _save(fig, "memory_hierarchy.png")
        return

    with open(bank_path, "rb") as f:
        bases = pickle.load(f)

    if not bases:
        print("  Memory bank is empty. Skipping.")
        return

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # Panel 1: Basis Dimensionality per Task
    ax1 = fig.add_subplot(gs[0, 0])
    dims = [b.shape[1] for b in bases]
    task_labels = [f"Task {i+1}" for i in range(len(bases))]
    bars = ax1.bar(task_labels, dims, color=PALETTE[:len(dims)], alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    for bar, d in zip(bars, dims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(d), ha='center', fontsize=11, color='#e0e0ff', fontweight='bold')
    ax1.set_ylabel("Basis Components")
    ax1.set_title("GPM Basis Dimensionality", fontsize=13, fontweight="bold")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Panel 2: Cumulative Subspace Coverage
    ax2 = fig.add_subplot(gs[0, 1])
    param_dim = bases[0].shape[0]
    cumulative = np.cumsum(dims)
    coverage_pct = cumulative / param_dim * 100
    ax2.plot(range(1, len(bases)+1), coverage_pct, '-o', color=PALETTE[0],
             markersize=8, linewidth=2.5)
    ax2.fill_between(range(1, len(bases)+1), coverage_pct, alpha=0.15, color=PALETTE[0])
    for i, (c, p) in enumerate(zip(cumulative, coverage_pct)):
        ax2.annotate(f"{p:.1f}%", (i+1, p), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10, color='#ffd43b')
    ax2.set_xlabel("Tasks Trained")
    ax2.set_ylabel("Subspace Coverage (%)")
    ax2.set_title("Cumulative Gradient Subspace", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Panel 3: SVD Singular Value Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    for i, b in enumerate(bases):
        _, S, _ = np.linalg.svd(b, full_matrices=False)
        S_norm = S / S.max() if S.max() > 0 else S
        ax3.plot(range(1, len(S_norm)+1), S_norm, '-', color=PALETTE[i % len(PALETTE)],
                 label=f"Task {i+1}", linewidth=2)
    ax3.set_xlabel("Component Index")
    ax3.set_ylabel("Normalized Singular Value")
    ax3.set_title("SVD Spectrum per Task", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, framealpha=0.3)
    ax3.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle("GPM Memory Level Hierarchy", fontsize=16,
                 fontweight="bold", color="#00d4ff", y=1.02)
    _save(fig, "memory_hierarchy.png")


# ═══════════════════════════════════════════════════════════
# 3. Evaluation Metrics Suite (ROC, PR, CM, Bar)
# ═══════════════════════════════════════════════════════════
def plot_evaluation_metrics(task_name="intrusion"):
    """ROC curve, PR curve, confusion matrix heatmap, per-class bar chart."""
    print(f"\n[3/4] Generating Evaluation Metrics for '{task_name}'...")
    _apply_style()

    mp = f"./logs/eval/metrics_{task_name}.json"
    if not os.path.exists(mp):
        print(f"  No metrics file found at {mp}. Run evaluation first.")
        return

    m = json.load(open(mp))
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # --- Panel 1: ROC Curve ---
    ax1 = fig.add_subplot(gs[0, 0])
    if "roc_curve" in m:
        fpr = m["roc_curve"]["fpr"]
        tpr = m["roc_curve"]["tpr"]
        auc_val = m.get("roc_auc", 0)
        ax1.plot(fpr, tpr, color=PALETTE[0], linewidth=2.5,
                 label=f"ROC (AUC = {auc_val:.4f})")
        ax1.fill_between(fpr, tpr, alpha=0.1, color=PALETTE[0])
        ax1.plot([0, 1], [0, 1], '--', color='#555577', linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.3)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Panel 2: Precision-Recall Curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    if "pr_curve" in m:
        prec_c = m["pr_curve"]["precision"]
        rec_c = m["pr_curve"]["recall"]
        pr_auc = m.get("pr_auc", 0)
        ax2.plot(rec_c, prec_c, color=PALETTE[1], linewidth=2.5,
                 label=f"PR (AUC = {pr_auc:.4f})")
        ax2.fill_between(rec_c, prec_c, alpha=0.1, color=PALETTE[1])
    if "optimal_threshold" in m:
        opt_t = m["optimal_threshold"]
        opt_p = m.get("optimal_precision_per_class", [0, 0])
        opt_r = m.get("optimal_recall_per_class", [0, 0])
        ax2.axhline(y=opt_p[1] if len(opt_p) > 1 else 0, color=PALETTE[3],
                    linestyle=':', alpha=0.7, label=f"Optimal θ={opt_t:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10, framealpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # --- Panel 3: Confusion Matrix Heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    if "confusion_matrix" in m:
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax3,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                    linewidths=1, linecolor='#2a2a4a', cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 14, "fontweight": "bold"})
    ax3.set_xlabel("Predicted", fontsize=11)
    ax3.set_ylabel("True", fontsize=11)
    ax3.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # --- Panel 4: Per-Class Metric Comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    classes = ["Normal (0)", "Attack (1)"]
    x = np.arange(len(classes))
    width = 0.25
    rpc = m.get("recall_per_class", [0, 0])
    ppc = m.get("precision_per_class", [0, 0])
    fpc = m.get("f1_per_class", [0, 0])
    ax4.bar(x - width, ppc, width, color=PALETTE[0], label="Precision", alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax4.bar(x, rpc, width, color=PALETTE[1], label="Recall", alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax4.bar(x + width, fpc, width, color=PALETTE[2], label="F1", alpha=0.85,
            edgecolor='white', linewidth=0.5)
    for j in range(len(classes)):
        ax4.text(x[j] - width, ppc[j] + 0.02, f"{ppc[j]:.2f}", ha='center', fontsize=9, color='#e0e0ff')
        ax4.text(x[j], rpc[j] + 0.02, f"{rpc[j]:.2f}", ha='center', fontsize=9, color='#e0e0ff')
        ax4.text(x[j] + width, fpc[j] + 0.02, f"{fpc[j]:.2f}", ha='center', fontsize=9, color='#e0e0ff')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes)
    ax4.set_ylim(0, 1.15)
    ax4.set_title("Per-Class Metrics", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9, framealpha=0.3)
    ax4.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.suptitle(f"Evaluation Metrics — {task_name}", fontsize=16,
                 fontweight="bold", color="#00d4ff", y=0.98)
    _save(fig, f"evaluation_metrics_{task_name}.png")


# ═══════════════════════════════════════════════════════════
# 4. Threshold Analysis
# ═══════════════════════════════════════════════════════════
def plot_threshold_analysis(task_name="intrusion"):
    """Precision vs Recall vs F1 as a function of decision threshold."""
    print(f"\n[4/4] Generating Threshold Analysis for '{task_name}'...")
    _apply_style()

    mp = f"./logs/eval/metrics_{task_name}.json"
    if not os.path.exists(mp):
        print(f"  No metrics file found. Skipping.")
        return
    m = json.load(open(mp))
    if "threshold_sweep" not in m:
        print("  No threshold sweep data. Skipping.")
        return

    ts = m["threshold_sweep"]
    thresholds = ts["thresholds"]
    opt_t = m.get("optimal_threshold", 0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, ts["precision"], '-', color=PALETTE[0], linewidth=2.5, label="Precision")
    ax.plot(thresholds, ts["recall"], '-', color=PALETTE[1], linewidth=2.5, label="Recall")
    ax.plot(thresholds, ts["f1"], '-', color=PALETTE[2], linewidth=2.5, label="F1")

    # Mark optimal threshold
    ax.axvline(x=opt_t, color=PALETTE[3], linestyle='--', linewidth=2, alpha=0.8,
               label=f"Optimal θ = {opt_t:.3f}")
    ax.axvline(x=0.5, color='#555577', linestyle=':', linewidth=1.5, alpha=0.6,
               label="Default θ = 0.500")

    # Annotate
    best_f1_idx = np.argmax(ts["f1"])
    ax.annotate(f"Best F1: {ts['f1'][best_f1_idx]:.3f}",
                xy=(thresholds[best_f1_idx], ts['f1'][best_f1_idx]),
                xytext=(20, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=PALETTE[2], lw=1.5),
                fontsize=11, color=PALETTE[2], fontweight='bold')

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis — {task_name}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, framealpha=0.3, loc="center left")
    ax.grid(True, linestyle='--', alpha=0.3)

    _save(fig, f"threshold_analysis_{task_name}.png")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def generate_all_plots(task_name=None):
    """Generate all visualization plots."""
    print("=" * 60)
    print("  SSSL-IDS — Generating Visualization Plots")
    print("=" * 60)

    task = task_name or "intrusion"
    plot_cl_metrics_dashboard()
    plot_memory_hierarchy()
    plot_evaluation_metrics(task)
    plot_threshold_analysis(task)

    print(f"\n{'='*60}")
    print(f"  All plots saved to {os.path.abspath(PLOT_DIR)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate IDS Visualization Plots")
    parser.add_argument("--task", type=str, default="intrusion", help="Task to visualize")
    args = parser.parse_args()
    generate_all_plots(task_name=args.task)


if __name__ == "__main__":
    main()
