"""
evaluate.py — Compute classification and continual-learning metrics.

Usage:
    python training/evaluate.py
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from training.train_task import build_task_head, make_task_labels


def evaluate_head(encoder: tf.keras.Model, head: tf.keras.Model,
                  features: np.ndarray, labels: np.ndarray, task_name: str):
    """Evaluate a single classifier head and print metrics."""
    embeddings = encoder(tf.constant(features, dtype=tf.float32), training=False).numpy()
    logits = head(tf.constant(embeddings, dtype=tf.float32), training=False).numpy()
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    preds = np.argmax(probs, axis=-1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Evaluation — {task_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")

    # ROC-AUC (binary tasks)
    if probs.shape[-1] == 2:
        try:
            roc = roc_auc_score(labels, probs[:, 1])
            pr_auc = average_precision_score(labels, probs[:, 1])
            print(f"  ROC-AUC  : {roc:.4f}")
            print(f"  PR-AUC   : {pr_auc:.4f}")
        except ValueError:
            print("  ROC/PR-AUC: not computable (single-class in labels)")

    print(f"\n  Classification Report:\n{classification_report(labels, preds, zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:\n{cm}\n")

    # Save confusion matrix plot
    os.makedirs("./logs/eval", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(cm, cmap="Blues", alpha=0.7)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {task_name}")
    plt.tight_layout()
    plt.savefig(f"./logs/eval/cm_{task_name}.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to ./logs/eval/cm_{task_name}.png")

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
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the testing CSV")
    parser.add_argument("--preprocessor_path", type=str, default="./checkpoints/preprocessor.pkl",
                        help="Path to the fitted preprocessor")
    args = parser.parse_args()

    # Load or build encoder
    encoder_path = "./checkpoints/encoder_frozen.keras"
    if os.path.exists(encoder_path):
        encoder = tf.keras.models.load_model(encoder_path)
    else:
        print("[WARN] No frozen encoder found; using fresh encoder for demo.")
        encoder = build_flow_encoder(input_dim=80)
    encoder.trainable = False
    embed_dim = encoder.output_shape[-1]

    loader = FlowDatasetLoader(data_path=".")
    if args.test_csv:
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
        ckpt = tf.train.latest_checkpoint(f"./checkpoints/{task_name}")
        if ckpt:
            tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()
        else:
            print(f"[WARN] No checkpoint for {task_name}; using random weights.")

        binary_labels = make_task_labels(task_name, labels_raw, preprocessor.get_classes())
        metrics = evaluate_head(encoder, head, features, binary_labels, task_name)
        results[task_name] = [metrics["f1"]]

    # Print forgetting summary
    if len(results) > 1:
        compute_forgetting_matrix(results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
