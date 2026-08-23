"""
compute_transfer.py — Compute Backward Transfer (BWT) and Forward Transfer (FWT)
for the SSSL-Based Continual IDS, with publication-quality visualization plots.

Supports two modes:
  1. Frozen encoder (default): reads F1 from existing metrics_{task}.json files.
     R[i][j] = R[j][j] for all i >= j (no forgetting by construction).
  2. Unfrozen encoder (--unfrozen): evaluates each (snapshot_i, task_j) pair.
     Loads encoder from checkpoints/{dataset}/snapshots/after_{task_i}/encoder/
     and each task's head from checkpoints/{dataset}/{task_j}/best/.

Generates publication-quality plots:
  - Transfer Matrix Heatmap (annotated R[i][j] matrix)
  - Task Performance Over Time (forgetting curves)
  - BWT Comparison Bar Chart (across datasets)
  - Combined Transfer Dashboard (multi-panel figure)

Usage:
    python training/compute_transfer.py --all
    python training/compute_transfer.py --dataset_name kddcup99
    python training/compute_transfer.py --dataset_name kddcup99 --unfrozen
    python training/compute_transfer.py --all --unfrozen
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Style Configuration for Plots ---
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

def _save_fig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PLOT] Saved: {path}")

TASK_ORDER = ["intrusion", "dos", "port_scan"]
TASK_DISPLAY_NAMES = {
    "intrusion": "Intrusion",
    "dos": "DoS",
    "port_scan": "Port Scan"
}

DATASET_CONFIG = {
    "kddcup99": {
        "type": "dataset",
        "dataset": "kddcup99",
        "data_path": "../KDDCUP99",
        "label_cols": {
            "intrusion": "Label",
            "dos": "AttackCategory",
            "port_scan": "AttackCategory",
        },
    },
    "cicids2017": {
        "type": "dataset",
        "dataset": "cicids2017",
        "data_path": "../CICIDS2017",
        "label_cols": {
            "intrusion": "Label",
            "dos": "AttackCategory",
            "port_scan": "AttackCategory",
        },
    },
    "unsw": {
        "type": "csv",
        "test_csv": "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv",
        "label_cols": {
            "intrusion": "label",
            "dos": "attack_cat",
            "port_scan": "attack_cat",
        },
    },
}


def read_task_f1(eval_dir: str, task_name: str) -> "float | None":
    """Read the weighted F1 score from the canonical metrics_{task}.json file."""
    metrics_path = os.path.join(eval_dir, f"metrics_{task_name}.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics.get("f1_weighted")


def get_preprocessor_path(task: str, ckpt_base: str) -> str:
    """Resolve the preprocessor file path for a task."""
    if task == "intrusion":
        p = os.path.join(ckpt_base, "preprocessor.pkl")
    else:
        p = os.path.join(ckpt_base, f"preprocessor_{task}.pkl")
    if not os.path.exists(p):
        fallback = os.path.join(ckpt_base, "preprocessor.pkl")
        if os.path.exists(fallback):
            return fallback
    return p


def batch_forward(model, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Run model forward pass in minibatches to prevent GPU out-of-memory errors."""
    import tensorflow as tf
    outputs = []
    for i in range(0, len(x), batch_size):
        batch = tf.constant(x[i:i+batch_size], dtype=tf.float32)
        out = model(batch, training=False).numpy()
        outputs.append(out)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.output_shape[-1]))


def load_test_dataframe(dataset_name: str, data_path: str = None, test_csv: str = None):
    """Load test set dataframe for a given dataset with auto-fallbacks and memory-safe sizing."""
    from data.dataset_loader import FlowDatasetLoader
    config = DATASET_CONFIG[dataset_name]
    if config["type"] == "dataset":
        dp = data_path or config["data_path"]
        if not os.path.exists(dp):
            for candidate in [
                f"/content/{dataset_name.upper()}",
                f"/content/{config['dataset'].upper()}",
                f"/content/{dataset_name}",
                f"/content/{config['dataset']}",
                "/content",
                f"../{dataset_name.upper()}",
                f"../{config['dataset'].upper()}",
            ]:
                if os.path.exists(candidate):
                    dp = candidate
                    break
        loader = FlowDatasetLoader(data_path=dp)
        df = loader.load_dataset(config["dataset"], split="test")
        if len(df) > 50000:
            print(f"  [INFO] Test set has {len(df):,} samples. Subsampling to 50,000 for evaluation speed and memory efficiency.")
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        return df
    elif config["type"] == "csv":
        tc = test_csv or config["test_csv"]
        if not os.path.exists(tc):
            for candidate in [
                "/content/IDS-UNSW_NB/UNSW_NB15_testing-set.csv",
                "/content/UNSW_NB15_testing-set.csv",
                "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv",
            ]:
                if os.path.exists(candidate):
                    tc = candidate
                    break
        loader = FlowDatasetLoader(data_path=".")
        df = loader.load_csv(tc)
        if len(df) > 50000:
            print(f"  [INFO] Test set has {len(df):,} samples. Subsampling to 50,000 for evaluation speed.")
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        return df
    else:
        raise ValueError(f"Unknown dataset type for {dataset_name}")


def load_encoder_from_snapshot(ckpt_base: str, snapshot_task: str):
    """Load encoder architecture and restore weights from a task snapshot."""
    import tensorflow as tf
    from training.train_task import load_frozen_encoder
    encoder = load_frozen_encoder(f"{ckpt_base}/encoder_frozen.keras")
    snapshot_dir = f"{ckpt_base}/snapshots/after_{snapshot_task}/encoder"
    ckpt = tf.train.latest_checkpoint(snapshot_dir)
    if ckpt:
        tf.train.Checkpoint(encoder=encoder).restore(ckpt).expect_partial()
        print(f"  [SNAPSHOT] Loaded encoder weights from {ckpt}")
    else:
        print(f"  [WARN] No snapshot found for after_{snapshot_task} at {snapshot_dir}, using SSL weights")
    encoder.trainable = False
    return encoder


def evaluate_task_with_encoder(encoder, task_name: str, ckpt_base: str, test_df, dataset_name: str):
    """Evaluate a task head using representations from a specific encoder state."""
    import tensorflow as tf
    from training.train_task import build_task_head, make_task_labels
    from data.preprocessing import FlowPreprocessor
    from sklearn.metrics import f1_score

    embed_dim = encoder.output_shape[-1]
    label_col = DATASET_CONFIG[dataset_name]["label_cols"][task_name]
    
    head = build_task_head(task_name, embed_dim=embed_dim)
    ckpt_path = (
        tf.train.latest_checkpoint(f"{ckpt_base}/{task_name}/best")
        or tf.train.latest_checkpoint(f"{ckpt_base}/{task_name}")
    )
    if not ckpt_path:
        print(f"    [SKIP] No checkpoint found for task '{task_name}'")
        return None
    tf.train.Checkpoint(head=head).restore(ckpt_path).expect_partial()
    
    prep_path = get_preprocessor_path(task_name, ckpt_base)
    if not os.path.exists(prep_path):
        print(f"    [SKIP] Preprocessor not found: {prep_path}")
        return None
    preprocessor = FlowPreprocessor.load(prep_path)
    
    features, labels_raw = preprocessor.transform(test_df, label_col=label_col)
    binary_labels = make_task_labels(task_name, labels_raw, preprocessor.get_classes())
    
    embeddings = batch_forward(encoder, features, batch_size=2048)
    logits = batch_forward(head, embeddings, batch_size=2048)
    preds = np.argmax(logits, axis=-1)
    
    return float(f1_score(binary_labels, preds, average="weighted", zero_division=0))


def compute_bwt(matrix: dict, c: int = 3) -> float:
    """
    Compute Backward Transfer (BWT) from results matrix:
    BWT = (2 / (c*(c-1))) * sum_{i=2}^{c} sum_{j=1}^{i-1} (R[i][j] - R[j][j])
    """
    if c <= 1:
        return 0.0
    s = 0.0
    count = 0
    for i in range(1, c):  # i from 1 to c-1 (stages 2 to c)
        for j in range(i):  # j from 0 to i-1 (tasks 1 to i-1)
            idx_i = str(i + 1)
            idx_j = str(j + 1)
            if (idx_i in matrix and idx_j in matrix[idx_i] and 
                idx_j in matrix and idx_j in matrix[idx_j]):
                r_ij = matrix[idx_i][idx_j]
                r_jj = matrix[idx_j][idx_j]
                if r_ij is not None and r_jj is not None:
                    s += (r_ij - r_jj)
                    count += 1
    if count > 0:
        return float((2.0 / (c * (c - 1))) * s)
    return 0.0


# ═══════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def plot_transfer_matrix_heatmap(matrix: dict, bwt: float, dataset_name: str, 
                                 task_order=None, mode: str = "frozen", 
                                 output_dirs=None):
    """
    Generate an annotated heatmap of the Transfer Matrix R[i][j].
    Rows: Training Stage i (After Task i)
    Cols: Target Task j (Evaluated on Task j)
    """
    _apply_style()
    task_order = task_order or TASK_ORDER
    c = len(task_order)
    task_names = [TASK_DISPLAY_NAMES.get(t, t.capitalize()) for t in task_order]

    # Build numpy matrix (c x c)
    r_mat = np.full((c, c), np.nan)
    for i in range(c):
        idx_i = str(i + 1)
        if idx_i in matrix:
            for j in range(i + 1):
                idx_j = str(j + 1)
                if idx_j in matrix[idx_i] and matrix[idx_i][idx_j] is not None:
                    r_mat[i, j] = matrix[idx_i][idx_j]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Mask upper triangle
    mask = np.triu(np.ones_like(r_mat, dtype=bool), k=1)

    # Custom colormap
    cmap = sns.color_palette("mako", as_cmap=True)

    row_labels = [f"After {name} (T{i+1})" for i, name in enumerate(task_names)]
    col_labels = [f"{name} (T{j+1})" for j, name in enumerate(task_names)]

    sns.heatmap(
        r_mat,
        mask=mask,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        vmin=max(0.0, float(np.nanmin(r_mat) - 0.05) if not np.isnan(np.nanmin(r_mat)) else 0.5),
        vmax=min(1.0, float(np.nanmax(r_mat) + 0.02) if not np.isnan(np.nanmax(r_mat)) else 1.0),
        cbar_kws={"label": "Weighted F1 Score", "shrink": 0.8},
        linewidths=2,
        linecolor="#1a1a2e",
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot_kws={"size": 13, "fontweight": "bold", "color": "#ffffff"}
    )

    # Highlight diagonal (initial task learning performance)
    for i in range(c):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='#ffd43b', lw=2.5, linestyle='--'))

    ax.set_title(
        f"Continual Learning Transfer Matrix $R[i][j]$ — {dataset_name.upper()}\n"
        f"BWT = {bwt:+.6f} | Mode: {mode.capitalize()} | Diagonal: Initial $R[j][j]$",
        fontsize=13, fontweight="bold", pad=15, color="#00d4ff"
    )
    ax.set_xlabel("Evaluated Task $j$", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Training Stage $i$", fontsize=12, fontweight="bold", labelpad=10)

    for out_dir in (output_dirs or []):
        _save_fig(fig, os.path.join(out_dir, "transfer_matrix_heatmap.png"))
    plt.close(fig)


def plot_task_performance_over_time(matrix: dict, dataset_name: str,
                                    task_order=None, mode: str = "frozen",
                                    output_dirs=None):
    """
    Generate the task retention / forgetting curve line plot:
    X-axis: Training Stage (After Task 1, After Task 2, After Task 3)
    Y-axis: Weighted F1 Score for each task j over time
    """
    _apply_style()
    task_order = task_order or TASK_ORDER
    c = len(task_order)
    task_names = [TASK_DISPLAY_NAMES.get(t, t.capitalize()) for t in task_order]

    stages = [f"Stage {i+1}\n(After {name})" for i, name in enumerate(task_names)]
    stage_indices = list(range(1, c + 1))

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v']

    for j, task_name in enumerate(task_names):
        xs = []
        ys = []
        # Evaluated at stages i >= j
        for i in range(j, c):
            idx_i = str(i + 1)
            idx_j = str(j + 1)
            if idx_i in matrix and idx_j in matrix[idx_i] and matrix[idx_i][idx_j] is not None:
                xs.append(i + 1)
                ys.append(matrix[idx_i][idx_j])

        if ys:
            color = PALETTE[j % len(PALETTE)]
            marker = markers[j % len(markers)]
            ax.plot(xs, ys, marker=marker, markersize=8, linewidth=2.5,
                    label=f"Task {j+1}: {task_name}", color=color)
            
            # Annotate points
            for x_val, y_val in zip(xs, ys):
                ax.annotate(
                    f"{y_val:.4f}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 10 if j % 2 == 0 else -18),
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=color
                )

    ax.set_xticks(stage_indices)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_xlabel("Sequential Training Progression", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Weighted F1 Score ($R[i][j]$)", fontsize=12, fontweight="bold", labelpad=10)
    
    # Set y-limits with padding
    all_y = [
        matrix[str(i+1)][str(j+1)]
        for i in range(c) for j in range(i+1)
        if str(i+1) in matrix and str(j+1) in matrix[str(i+1)] and matrix[str(i+1)][str(j+1)] is not None
    ]
    if all_y:
        y_min = max(0.0, min(all_y) - 0.08)
        y_max = min(1.03, max(all_y) + 0.08)
        ax.set_ylim(y_min, y_max)

    ax.set_title(
        f"Continual Learning — Task Performance Retention Over Time ({dataset_name.upper()})\n"
        f"Flat lines = Perfect Retention (GPM Protected) | Declining = Catastrophic Forgetting",
        fontsize=12, fontweight="bold", pad=15, color="#00d4ff"
    )
    ax.legend(fontsize=10, loc="lower left", framealpha=0.4)
    ax.grid(True, linestyle='--', alpha=0.3)

    for out_dir in (output_dirs or []):
        _save_fig(fig, os.path.join(out_dir, "task_performance_over_time.png"))
    plt.close(fig)


def plot_transfer_dashboard(matrix: dict, bwt: float, dataset_name: str,
                            task_order=None, mode: str = "frozen",
                            output_dirs=None):
    """
    Generate a 2-panel comprehensive Transfer Dashboard:
    Left: R[i][j] Transfer Heatmap
    Right: Task Performance Over Time Forgetting Curves
    """
    _apply_style()
    task_order = task_order or TASK_ORDER
    c = len(task_order)
    task_names = [TASK_DISPLAY_NAMES.get(t, t.capitalize()) for t in task_order]

    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 2, wspace=0.32)

    # --- Panel 1: Transfer Heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    r_mat = np.full((c, c), np.nan)
    for i in range(c):
        idx_i = str(i + 1)
        if idx_i in matrix:
            for j in range(i + 1):
                idx_j = str(j + 1)
                if idx_j in matrix[idx_i] and matrix[idx_i][idx_j] is not None:
                    r_mat[i, j] = matrix[idx_i][idx_j]

    mask = np.triu(np.ones_like(r_mat, dtype=bool), k=1)
    cmap = sns.color_palette("mako", as_cmap=True)
    row_labels = [f"After {name}" for name in task_names]
    col_labels = [f"{name}" for name in task_names]

    sns.heatmap(
        r_mat,
        mask=mask,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        vmin=max(0.0, float(np.nanmin(r_mat) - 0.05) if not np.isnan(np.nanmin(r_mat)) else 0.5),
        vmax=min(1.0, float(np.nanmax(r_mat) + 0.02) if not np.isnan(np.nanmax(r_mat)) else 1.0),
        cbar_kws={"label": "F1 Score", "shrink": 0.8},
        linewidths=1.5,
        linecolor="#1a1a2e",
        ax=ax1,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot_kws={"size": 12, "fontweight": "bold", "color": "#ffffff"}
    )
    for i in range(c):
        ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='#ffd43b', lw=2, linestyle='--'))

    ax1.set_title("Transfer Matrix $R[i][j]$", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Evaluated Task $j$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Training Stage $i$", fontsize=11, fontweight="bold")

    # --- Panel 2: Performance Over Time ---
    ax2 = fig.add_subplot(gs[0, 1])
    stages = [f"Stage {i+1}\n({name})" for i, name in enumerate(task_names)]
    markers = ['o', 's', '^', 'D']

    for j, task_name in enumerate(task_names):
        xs, ys = [], []
        for i in range(j, c):
            idx_i, idx_j = str(i + 1), str(j + 1)
            if idx_i in matrix and idx_j in matrix[idx_i] and matrix[idx_i][idx_j] is not None:
                xs.append(i + 1)
                ys.append(matrix[idx_i][idx_j])
        if ys:
            color = PALETTE[j % len(PALETTE)]
            ax2.plot(xs, ys, marker=markers[j % len(markers)], markersize=8, linewidth=2.5,
                     label=f"T{j+1}: {task_name}", color=color)
            for x_val, y_val in zip(xs, ys):
                ax2.annotate(f"{y_val:.4f}", (x_val, y_val), textcoords="offset points",
                             xytext=(0, 10 if j % 2 == 0 else -18), ha='center',
                             fontsize=9, fontweight='bold', color=color)

    ax2.set_xticks(list(range(1, c + 1)))
    ax2.set_xticklabels(stages, fontsize=10)
    ax2.set_xlabel("Sequential Training Progression", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Weighted F1 Score ($R[i][j]$)", fontsize=11, fontweight="bold")
    ax2.set_title("Task Retention / Forgetting Curves", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.4)
    ax2.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle(
        f"Continual Learning Evaluation Dashboard — {dataset_name.upper()} "
        f"(BWT = {bwt:+.6f} | Mode: {mode.capitalize()})",
        fontsize=15, fontweight="bold", color="#00d4ff", y=0.98
    )

    for out_dir in (output_dirs or []):
        _save_fig(fig, os.path.join(out_dir, "transfer_dashboard.png"))
    plt.close(fig)


def plot_bwt_comparison(all_results: dict, output_dirs=None):
    """
    Generate a comparison bar chart of Backward Transfer (BWT) across all datasets.
    """
    _apply_style()
    valid_results = {k: v for k, v in all_results.items() if v is not None and "bwt" in v}
    if not valid_results:
        return

    datasets = list(valid_results.keys())
    bwt_values = [valid_results[d]["bwt"] for d in datasets]
    modes = [valid_results[d].get("mode", "frozen") for d in datasets]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(datasets))
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(datasets))]

    bars = ax.bar(x, bwt_values, width=0.45, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=1.2)

    # Add zero baseline reference
    ax.axhline(0.0, color='#ffd43b', linestyle='--', linewidth=1.8, label="Zero Forgetting Baseline (BWT = 0.0)")

    # Annotate bars
    for bar, val, mode in zip(bars, bwt_values, modes):
        height = bar.get_height()
        y_offset = 0.002 if val >= 0 else -0.005
        va = 'bottom' if val >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"BWT = {val:+.6f}\n({mode})",
            ha='center', va=va, fontsize=10, fontweight='bold', color='#e0e0ff'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11, fontweight='bold')
    ax.set_ylabel("Backward Transfer (BWT)", fontsize=12, fontweight='bold')
    ax.set_title(
        "Continual Learning — Backward Transfer (BWT) Across Datasets\n"
        r"$\text{BWT} = \frac{2}{C(C-1)}\sum_{i=2}^C \sum_{j=1}^{i-1} (R[i][j] - R[j][j])$",
        fontsize=13, fontweight='bold', pad=15, color="#00d4ff"
    )

    y_abs_max = max(abs(min(bwt_values, default=0.0)), abs(max(bwt_values, default=0.0)), 0.02)
    ax.set_ylim(-y_abs_max * 1.5 - 0.01, y_abs_max * 1.5 + 0.01)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.4)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    for out_dir in (output_dirs or []):
        _save_fig(fig, os.path.join(out_dir, "bwt_comparison_bar_chart.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# ANALYSIS ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

def run_transfer_analysis(dataset_name: str, unfrozen: bool = False, data_path: str = None, test_csv: str = None) -> dict:
    """
    Run transfer analysis for a dataset, build results matrix, compute BWT/FWT,
    save results to JSON, and render all visualization plots.
    """
    ckpt_base = f"./checkpoints/{dataset_name}"
    log_base = f"./logs/{dataset_name}"
    eval_dir = f"{log_base}/eval"
    plot_dir = f"{log_base}/plots"
    global_plot_dir = "./logs/plots"

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(global_plot_dir, exist_ok=True)

    mode_str = "unfrozen" if unfrozen else "frozen"

    print(f"\n{'=' * 60}")
    print(f"  Computing Transfer Matrix for: {dataset_name.upper()} (Mode: {mode_str})")
    print(f"{'=' * 60}")

    c = len(TASK_ORDER)
    matrix = {}

    if not unfrozen:
        # Frozen mode: read canonical F1 from metrics_{task}.json
        for i, task_i in enumerate(TASK_ORDER):
            idx_i = str(i + 1)
            matrix[idx_i] = {}
            for j in range(i + 1):
                idx_j = str(j + 1)
                task_j = TASK_ORDER[j]
                f1 = read_task_f1(eval_dir, task_j)
                if f1 is not None:
                    print(f"  Stage {idx_i} ({task_i}) -> Task {idx_j} ({task_j}): F1 = {f1:.6f}")
                    matrix[idx_i][idx_j] = round(f1, 6)
                else:
                    print(f"  Stage {idx_i} ({task_i}) -> Task {idx_j} ({task_j}): SKIPPED (no metrics)")
    else:
        # Unfrozen mode: evaluate each snapshot
        import tensorflow as tf
        try:
            test_df = load_test_dataframe(dataset_name, data_path=data_path, test_csv=test_csv)
        except Exception as e:
            print(f"[ERROR] Failed to load test dataframe for {dataset_name}: {e}")
            return None

        for i, task_i in enumerate(TASK_ORDER):
            idx_i = str(i + 1)
            matrix[idx_i] = {}

            snapshot_dir = f"{ckpt_base}/snapshots/after_{task_i}/encoder"
            if not os.path.exists(snapshot_dir) or not tf.train.latest_checkpoint(snapshot_dir):
                print(f"  [SKIP] No snapshot found for after_{task_i} at {snapshot_dir}")
                # Fallback to reading existing metric if snapshot is missing
                for j in range(i + 1):
                    f1 = read_task_f1(eval_dir, TASK_ORDER[j])
                    if f1 is not None:
                        matrix[idx_i][str(j + 1)] = round(f1, 6)
                continue

            encoder = load_encoder_from_snapshot(ckpt_base, task_i)

            for j in range(i + 1):
                idx_j = str(j + 1)
                task_j = TASK_ORDER[j]
                print(f"  Evaluating encoder after '{task_i}' on task '{task_j}'...")
                f1 = evaluate_task_with_encoder(encoder, task_j, ckpt_base, test_df, dataset_name)
                if f1 is not None:
                    matrix[idx_i][idx_j] = round(f1, 6)
                    print(f"    R[{idx_i}][{idx_j}] = {f1:.6f}")

    bwt = compute_bwt(matrix, c=c)
    fwt = None

    results = {
        "results_matrix": matrix,
        "bwt": round(bwt, 6),
        "fwt": fwt,
        "fwt_note": (
            "FWT is undefined for this architecture: each task uses an independent "
            "binary classification head that does not exist before that task is trained. "
            "Zero-shot forward transfer requires a shared output space (e.g., "
            "class-incremental learning with a single growing output layer)."
        ),
        "bwt_note": (
            "In frozen mode, BWT is 0.0 by construction. In unfrozen mode with GPM, "
            "BWT measures retention of earlier task representations after fine-tuning "
            "on subsequent tasks with gradient projection."
        ),
        "task_order": TASK_ORDER,
        "mode": mode_str,
    }

    # Save to canonical location
    out_file = f"{eval_dir}/continual_transfer.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    # Generate Plots
    print("\n  Generating transfer visualization plots...")
    plot_targets = [plot_dir, global_plot_dir]
    plot_transfer_matrix_heatmap(matrix, bwt, dataset_name, TASK_ORDER, mode_str, plot_targets)
    plot_task_performance_over_time(matrix, dataset_name, TASK_ORDER, mode_str, plot_targets)
    plot_transfer_dashboard(matrix, bwt, dataset_name, TASK_ORDER, mode_str, plot_targets)

    # Print ASCII Matrix
    print(f"\n  Results Matrix R[i][j] — {dataset_name.upper()} ({mode_str})")
    print(f"  {'':>18}", end="")
    for t in TASK_ORDER:
        print(f"{t:>12}", end="")
    print()
    for i in range(c):
        idx_i = str(i + 1)
        print(f"  After task {idx_i:>2}  ", end="")
        for j in range(c):
            idx_j = str(j + 1)
            if idx_i in matrix and idx_j in matrix[idx_i] and matrix[idx_i][idx_j] is not None:
                print(f"{matrix[idx_i][idx_j]:>12.4f}", end="")
            else:
                print(f"{'—':>12}", end="")
        print()

    print(f"\n  BWT = {bwt:+.6f}")
    print(f"  FWT = null (undefined for independent-head architecture)")

    return results


def print_summary_table(all_results: dict):
    """Print formatted summary table and generate cross-dataset BWT bar chart."""
    print(f"\n{'=' * 60}")
    print(f"  BWT / FWT Summary — All Datasets")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<15} {'Mode':<10} {'BWT':>12} {'FWT':>10}")
    print(f"  {'-' * 52}")
    for ds_name, result in all_results.items():
        if result is not None:
            bwt_str = f"{result['bwt']:+.6f}"
            fwt_str = "null" if result["fwt"] is None else f"{result['fwt']:.6f}"
            mode_str = result.get("mode", "frozen")
            print(f"  {ds_name:<15} {mode_str:<10} {bwt_str:>12} {fwt_str:>10}")
    print(f"  {'-' * 52}\n")

    # Generate cross-dataset comparison plots
    plot_dirs = ["./logs/plots"] + [f"./logs/{ds}/plots" for ds in all_results.keys()]
    plot_bwt_comparison(all_results, plot_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Compute BWT and FWT for Continual IDS with Visualization Plots"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["kddcup99", "cicids2017", "unsw"],
        default=None,
        help="Dataset to compute transfer metrics for",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset directory path",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Test CSV path for tabular datasets",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all three datasets",
    )
    parser.add_argument(
        "--unfrozen",
        action="store_true",
        help="Evaluate from snapshots (unfrozen encoder mode)",
    )
    args = parser.parse_args()

    if not args.all and not args.dataset_name:
        parser.error("Either --dataset_name or --all is required")

    datasets = (
        ["kddcup99", "cicids2017", "unsw"] if args.all else [args.dataset_name]
    )

    all_results = {}
    for ds in datasets:
        result = run_transfer_analysis(ds, unfrozen=args.unfrozen, data_path=args.data_path, test_csv=args.test_csv)
        all_results[ds] = result

    if len(all_results) > 1 or args.all:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
