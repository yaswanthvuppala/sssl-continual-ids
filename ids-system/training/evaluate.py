"""
evaluate.py — Compute classification and continual-learning metrics.

Usage:
    python training/evaluate.py
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from training.train_task import build_task_head, make_task_labels


def find_optimal_threshold(labels, probs_positive, strategy="f1"):
    """
    Find the optimal decision threshold for the positive class.
    
    strategy:
        - 'f1': maximize F1 score
        - 'recall_90': find the highest threshold that achieves >= 0.90 recall
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, probs_positive)
    
    if strategy == "f1":
        # F1 = 2 * (P * R) / (P + R)
        f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return best_threshold, {
            "threshold": float(best_threshold),
            "precision": float(precision_arr[best_idx]),
            "recall": float(recall_arr[best_idx]),
            "f1": float(f1_scores[best_idx]),
        }
    elif strategy == "recall_90":
        # Find highest threshold where recall >= 0.90
        valid = recall_arr >= 0.90
        if valid.any():
            # Among valid, pick the one with highest precision (= highest threshold)
            valid_indices = np.where(valid)[0]
            best_idx = valid_indices[np.argmax(precision_arr[valid_indices])]
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            best_threshold = 0.3  # fallback: lower threshold to boost recall
            best_idx = np.argmin(np.abs(thresholds - best_threshold)) if len(thresholds) > 0 else 0
        return best_threshold, {
            "threshold": float(best_threshold),
            "precision": float(precision_arr[best_idx]),
            "recall": float(recall_arr[best_idx]),
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_head(encoder: tf.keras.Model, head: tf.keras.Model,
                  features: np.ndarray, labels: np.ndarray, task_name: str,
                  eval_dir: str = None):
    """Evaluate a single classifier head and print metrics."""
    eval_dir = eval_dir or "./logs/eval"
    embeddings = encoder(tf.constant(features, dtype=tf.float32), training=False).numpy()
    logits = head(tf.constant(embeddings, dtype=tf.float32), training=False).numpy()
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    preds = np.argmax(probs, axis=-1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # Per-class metrics
    rec_per_class = recall_score(labels, preds, average=None, zero_division=0)
    prec_per_class = precision_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Evaluation — {task_name} (default threshold=0.5)")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")

    metrics_dict = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "recall_per_class": rec_per_class.tolist(),
        "precision_per_class": prec_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }

    # ROC-AUC (binary tasks) + optimal threshold
    if probs.shape[-1] == 2:
        try:
            roc = roc_auc_score(labels, probs[:, 1])
            pr_auc = average_precision_score(labels, probs[:, 1])
            print(f"  ROC-AUC  : {roc:.4f}")
            print(f"  PR-AUC   : {pr_auc:.4f}")
            metrics_dict["roc_auc"] = float(roc)
            metrics_dict["pr_auc"] = float(pr_auc)

            # --- Optimal threshold search ---
            opt_threshold, opt_info = find_optimal_threshold(labels, probs[:, 1], strategy="f1")
            preds_opt = (probs[:, 1] >= opt_threshold).astype(int)
            acc_opt = accuracy_score(labels, preds_opt)
            rec_opt = recall_score(labels, preds_opt, average="weighted", zero_division=0)
            f1_opt = f1_score(labels, preds_opt, average="weighted", zero_division=0)
            rec_opt_pc = recall_score(labels, preds_opt, average=None, zero_division=0)
            prec_opt_pc = precision_score(labels, preds_opt, average=None, zero_division=0)

            print(f"\n  --- Optimal Threshold: {opt_threshold:.4f} ---")
            print(f"  Accuracy : {acc_opt:.4f}")
            print(f"  Recall   : {rec_opt:.4f}  (class-0: {rec_opt_pc[0]:.4f}, class-1: {rec_opt_pc[1]:.4f})")
            print(f"  F1       : {f1_opt:.4f}")

            metrics_dict["optimal_threshold"] = float(opt_threshold)
            metrics_dict["optimal_accuracy"] = float(acc_opt)
            metrics_dict["optimal_recall_weighted"] = float(rec_opt)
            metrics_dict["optimal_f1_weighted"] = float(f1_opt)
            metrics_dict["optimal_recall_per_class"] = rec_opt_pc.tolist()
            metrics_dict["optimal_precision_per_class"] = prec_opt_pc.tolist()

            # Save ROC curve data for visualization
            fpr, tpr, roc_thresholds = roc_curve(labels, probs[:, 1])
            prec_curve, rec_curve, pr_thresholds = precision_recall_curve(labels, probs[:, 1])
            metrics_dict["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            metrics_dict["pr_curve"] = {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()}

            # Threshold sweep for visualization
            sweep_thresholds = np.linspace(0.05, 0.95, 50)
            sweep_f1, sweep_rec, sweep_prec = [], [], []
            for t in sweep_thresholds:
                p = (probs[:, 1] >= t).astype(int)
                sweep_f1.append(float(f1_score(labels, p, average="weighted", zero_division=0)))
                sweep_rec.append(float(recall_score(labels, p, average="weighted", zero_division=0)))
                sweep_prec.append(float(precision_score(labels, p, average="weighted", zero_division=0)))
            metrics_dict["threshold_sweep"] = {
                "thresholds": sweep_thresholds.tolist(),
                "f1": sweep_f1, "recall": sweep_rec, "precision": sweep_prec
            }

        except ValueError:
            print("  ROC/PR-AUC: not computable (single-class in labels)")

    print(f"\n  Classification Report:\n{classification_report(labels, preds, zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:\n{cm}\n")

    metrics_dict["confusion_matrix"] = cm.tolist()

    # Save confusion matrix plot
    os.makedirs(eval_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(cm, cmap="Blues", alpha=0.7)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {task_name}")
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/cm_{task_name}.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {eval_dir}/cm_{task_name}.png")

    # Save metrics as JSON for the visualization script
    metrics_path = f"{eval_dir}/metrics_{task_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_forgetting_matrix(results_per_task: dict) -> np.ndarray:
    """
    Computes a forgetting matrix.
    results_per_task: dict of {task_name: list_of_f1_after_each_task}
    Each list has length equal to total number of tasks trained so far.
    """
    tasks = list(results_per_task.keys())
    n = len(tasks)
    matrix = np.zeros((n, n))
    for i, t in enumerate(tasks):
        scores = results_per_task[t]
        for j in range(len(scores)):
            matrix[i, j] = scores[j]

    print("\nForgetting Matrix (rows=tasks, cols=after training task j):")
    print(f"{'':>15}", end="")
    for t in tasks:
        print(f"{t:>12}", end="")
    print()
    for i, t in enumerate(tasks):
        print(f"{t:>15}", end="")
        for j in range(n):
            print(f"{matrix[i, j]:>12.4f}", end="")
        print()

    return matrix


def main():
    from encoder.flow_encoder import build_flow_encoder

    parser = argparse.ArgumentParser(description="Evaluate trained IDS heads")
    parser.add_argument("--task", type=str, default="all", choices=["all", "intrusion", "dos", "port_scan"],
                        help="Task head to evaluate")
    parser.add_argument("--test_csv", type=str, default=None, help="Testing CSV")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw"],
                        default=None, help="Load a supported raw dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset directory or raw data file")
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the testing CSV")
    parser.add_argument("--preprocessor_path", type=str, default=None,
                        help="Path to the fitted preprocessor")
    parser.add_argument("--dataset_name", type=str, default="default",
                        help="Dataset identifier for scoping output paths")
    args = parser.parse_args()

    # Resolve dataset-scoped base paths
    ds = args.dataset_name
    ckpt_base = f"./checkpoints/{ds}"
    log_base = f"./logs/{ds}"
    eval_dir = f"{log_base}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    if args.preprocessor_path is None:
        if args.task == "intrusion" or args.task == "all":
            args.preprocessor_path = f"{ckpt_base}/preprocessor.pkl"
        else:
            args.preprocessor_path = f"{ckpt_base}/preprocessor_{args.task}.pkl"

    # Load or build encoder
    encoder_path = f"{ckpt_base}/encoder_frozen.keras"
    if os.path.exists(encoder_path):
        encoder = tf.keras.models.load_model(encoder_path)
    else:
        # Fallback to old flat path for backward compatibility
        old_path = "./checkpoints/encoder_frozen.keras"
        if os.path.exists(old_path):
            print(f"[INFO] Falling back to legacy encoder path: {old_path}")
            encoder = tf.keras.models.load_model(old_path)
        else:
            print("[WARN] No frozen encoder found; using fresh encoder for demo.")
            encoder = build_flow_encoder(input_dim=80)
    encoder.trainable = False
    embed_dim = encoder.output_shape[-1]

    loader = FlowDatasetLoader(data_path=args.data_path or ".")
    if args.dataset:
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset is used")
        if not os.path.exists(args.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {args.preprocessor_path}. "
                "Run SSL/task training on the training split first."
            )
        df = loader.load_dataset(
            args.dataset, split="test", label_col=args.label_col
        )
        preprocessor = FlowPreprocessor.load(args.preprocessor_path)
        features, labels_raw = preprocessor.transform(
            df, label_col=args.label_col
        )
        if features.shape[1] != encoder.input_shape[-1]:
            raise ValueError(
                f"Testing features have {features.shape[1]} columns but the "
                f"frozen encoder expects {encoder.input_shape[-1]}. Use the "
                "preprocessor and encoder from the same training run."
            )
    elif args.test_csv:
        if not os.path.exists(args.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {args.preprocessor_path}. "
                "Run SSL/task training on the training CSV first."
            )
        df = loader.load_csv(args.test_csv, label_col=args.label_col)
        preprocessor = FlowPreprocessor.load(args.preprocessor_path)
        features, labels_raw = preprocessor.transform(df, label_col=args.label_col)
        if features.shape[1] != encoder.input_shape[-1]:
            raise ValueError(
                f"Testing features have {features.shape[1]} columns but the frozen encoder expects "
                f"{encoder.input_shape[-1]}. Use the preprocessor and encoder from the same training run."
            )
    else:
        df = loader.create_synthetic_data(num_samples=2000, num_features=80)
        preprocessor = FlowPreprocessor()
        features, labels_raw = preprocessor.fit_transform(df, label_col=args.label_col)

    # Evaluate each head
    results = {}
    task_names = ["intrusion", "dos", "port_scan"] if args.task == "all" else [args.task]
    for task_name in task_names:
        head = build_task_head(task_name, embed_dim=embed_dim)
        ckpt = tf.train.latest_checkpoint(f"{ckpt_base}/{task_name}")
        if not ckpt:
            # Fallback to old flat path
            ckpt = tf.train.latest_checkpoint(f"./checkpoints/{task_name}")
        if ckpt:
            tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()
        else:
            print(f"[WARN] No checkpoint for {task_name}; using random weights.")

        binary_labels = make_task_labels(task_name, labels_raw, preprocessor.get_classes())
        metrics = evaluate_head(encoder, head, features, binary_labels, task_name,
                                eval_dir=eval_dir)
        results[task_name] = [metrics["f1"]]

    # Print forgetting summary
    if len(results) > 1:
        compute_forgetting_matrix(results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
