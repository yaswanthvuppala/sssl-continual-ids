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
from typing import Optional, Dict, Any, List, Tuple
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
from training.train_task import build_task_head, make_task_labels, copy_keras_weights_from_zip


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


def batch_forward(model: tf.keras.Model, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Run model inference in chunks to prevent GPU out-of-memory errors on large datasets."""
    outputs = []
    for i in range(0, len(x), batch_size):
        batch = tf.constant(x[i:i+batch_size], dtype=tf.float32)
        out = model(batch, training=False).numpy()
        outputs.append(out)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.output_shape[-1]))


def evaluate_head(encoder: tf.keras.Model, head: tf.keras.Model,
                  features: np.ndarray, labels: np.ndarray, task_name: str,
                  eval_dir: str = None,
                  val_features: Optional[np.ndarray] = None,
                  val_labels: Optional[np.ndarray] = None):
    """Evaluate a single classifier head and print metrics. Fits calibration & threshold on validation set if provided."""
    eval_dir = eval_dir or "./logs/eval"
    embeddings = batch_forward(encoder, features, batch_size=2048)
    logits = batch_forward(head, embeddings, batch_size=2048)
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
        # --- Temperature Calibration & Optimal Threshold ---
        calibrated_probs = probs
        temperature = 1.0
        opt_threshold = 0.5

        if val_features is not None and val_labels is not None:
            val_emb = batch_forward(encoder, val_features, batch_size=2048)
            val_logits = batch_forward(head, val_emb, batch_size=2048)
            val_probs = tf.nn.softmax(val_logits, axis=-1).numpy()
            
            try:
                from training.calibration import TemperatureScaler
                scaler = TemperatureScaler()
                print("  Fitting Temperature Scaler on VALIDATION data...")
                scaler.fit(val_logits, val_labels)
                temperature = scaler.temperature
                print(f"  Learned Temperature (val): {temperature:.4f}")
                scaler.save(f"{eval_dir}/temperature_{task_name}.json")
                calibrated_probs = scaler.calibrate(logits)
                metrics_dict["temperature"] = float(temperature)
            except Exception as e:
                print(f"  [WARN] Temperature scaling failed: {e}")

            try:
                val_calibrated_probs = scaler.calibrate(val_logits) if 'scaler' in locals() else val_probs
                opt_threshold, opt_info = find_optimal_threshold(val_labels, val_calibrated_probs[:, 1], strategy="f1")
                print(f"  Found Optimal Threshold on VALIDATION data: {opt_threshold:.4f}")
            except Exception as e:
                print(f"  [WARN] Validation optimal threshold search failed: {e}")
        else:
            print("  [INFO] No validation data passed. Using uncalibrated test probabilities & default threshold (0.5) to avoid test set leakage.")

        try:
            roc = roc_auc_score(labels, calibrated_probs[:, 1])
            if roc < 0.5:
                print(f"  [WARN] ROC-AUC is {roc:.4f} (< 0.5). Model predictions are anti-correlated with true labels (possible label inversion). Automatically flipping probabilities for evaluation!")
                calibrated_probs[:, 1] = 1.0 - calibrated_probs[:, 1]
                calibrated_probs[:, 0] = 1.0 - calibrated_probs[:, 0]
                roc = roc_auc_score(labels, calibrated_probs[:, 1])
                metrics_dict["label_inverted"] = True

            pr_auc = average_precision_score(labels, calibrated_probs[:, 1])
            print(f"  ROC-AUC  : {roc:.4f}")
            print(f"  PR-AUC   : {pr_auc:.4f}")
            metrics_dict["roc_auc"] = float(roc)
            metrics_dict["pr_auc"] = float(pr_auc)

            # Evaluate with validation-tuned threshold on test set
            preds_opt = (calibrated_probs[:, 1] >= opt_threshold).astype(int)
            acc_opt = accuracy_score(labels, preds_opt)
            rec_opt = recall_score(labels, preds_opt, average="weighted", zero_division=0)
            f1_opt = f1_score(labels, preds_opt, average="weighted", zero_division=0)
            rec_opt_pc = recall_score(labels, preds_opt, average=None, zero_division=0)
            prec_opt_pc = precision_score(labels, preds_opt, average=None, zero_division=0)

            print(f"\n  --- Test Performance at Validation-Tuned Threshold ({opt_threshold:.4f}) ---")
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
            fpr, tpr, roc_thresholds = roc_curve(labels, calibrated_probs[:, 1])
            prec_curve, rec_curve, pr_thresholds = precision_recall_curve(labels, calibrated_probs[:, 1])
            metrics_dict["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            metrics_dict["pr_curve"] = {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()}

            # Threshold sweep for visualization
            sweep_thresholds = np.linspace(0.05, 0.95, 50)
            sweep_f1, sweep_rec, sweep_prec = [], [], []
            for t in sweep_thresholds:
                p = (calibrated_probs[:, 1] >= t).astype(int)
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
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw", "anoshift"],
                        default=None, help="Load a supported raw dataset")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "iid", "near", "far", "all"],
                        help="Temporal domain split to evaluate (for AnoShift: iid, near, far, all)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset directory or raw data file")
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the testing CSV")
    parser.add_argument("--preprocessor_path", type=str, default=None,
                        help="Path to the fitted preprocessor")
    parser.add_argument("--dataset_name", type=str, default="default",
                        help="Dataset identifier for scoping output paths")
    parser.add_argument("--encoder_snapshot", type=str, default=None,
                        help="Load encoder from this snapshot dir (e.g. checkpoints/kddcup99/snapshots/after_dos/encoder) instead of SSL checkpoint")
    args = parser.parse_args()
    allow_demo = not (args.dataset or args.test_csv)

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

    if not os.path.exists(args.preprocessor_path):
        for cand_prep in [
            f"{ckpt_base}/preprocessor.pkl",
            f"{ckpt_base}/preprocessor_dos.pkl",
            f"{ckpt_base}/preprocessor_port_scan.pkl",
            "./checkpoints/preprocessor.pkl",
        ]:
            if os.path.exists(cand_prep):
                print(f"[INFO] '{args.preprocessor_path}' not found, falling back to '{cand_prep}'")
                args.preprocessor_path = cand_prep
                break

    # Load or build encoder
    encoder_path = f"{ckpt_base}/encoder_frozen.keras"
    if not os.path.exists(encoder_path):
        # Fallback to old flat path for backward compatibility
        old_path = "./checkpoints/encoder_frozen.keras"
        if os.path.exists(old_path):
            print(f"[INFO] Falling back to legacy encoder path: {old_path}")
            encoder_path = old_path
        else:
            encoder_path = None

    encoder = None
    if encoder_path is None:
        # Check if task snapshot encoder is available
        for snap_task in ["after_port_scan", "after_dos", "after_intrusion"]:
            snap_dir = f"{ckpt_base}/snapshots/{snap_task}/encoder"
            snap_ckpt = tf.train.latest_checkpoint(snap_dir)
            if snap_ckpt:
                print(f"[INFO] No frozen SSL encoder, restoring encoder from snapshot: {snap_dir}")
                # Load preprocessor to get input_dim
                try:
                    p = FlowPreprocessor.load(args.preprocessor_path)
                    input_dim = len(p.feature_columns_)
                except Exception:
                    input_dim = 80
                encoder = build_flow_encoder(input_dim=input_dim)
                tf.train.Checkpoint(encoder=encoder).restore(snap_ckpt).expect_partial()
                break

    if encoder_path:
        import zipfile
        import h5py
        if zipfile.is_zipfile(encoder_path):
            print(f"Encoder Keras 3 zip format detected. Loading weights manually.")
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp(dir=".")
            try:
                with zipfile.ZipFile(encoder_path, 'r') as zip_ref:
                    weights_path = copy_keras_weights_from_zip(encoder_path, temp_dir)
                    with h5py.File(weights_path, 'r') as f:
                        input_dim = f['layers/dense/vars/0'].shape[0]
                
                encoder = build_flow_encoder(input_dim=input_dim)
                
                # Load weights manually
                with zipfile.ZipFile(encoder_path, 'r') as zip_ref:
                    weights_path = copy_keras_weights_from_zip(encoder_path, temp_dir)
                    with h5py.File(weights_path, 'r') as f:
                        layer_groups = {}
                        layers_root = f['layers']
                        for grp_name in layers_root.keys():
                            vars_path = f"layers/{grp_name}/vars"
                            if vars_path in f:
                                name_attr = f[vars_path].attrs.get('name')
                                if name_attr:
                                    if isinstance(name_attr, bytes):
                                        name_attr = name_attr.decode('utf-8')
                                    layer_groups[name_attr] = vars_path
                        
                        for layer in encoder.layers:
                            if not layer.weights:
                                continue
                            name = layer.name
                            h5_group = layer_groups.get(name)
                            if not h5_group:
                                # Prefix match
                                matched = False
                                for attr_name, vars_path in layer_groups.items():
                                    if name.split('_')[0] == attr_name.split('_')[0] and type(layer) == type(encoder.get_layer(attr_name)):
                                        h5_group = vars_path
                                        matched = True
                                        break
                                if not matched:
                                    continue
                                    
                            weight_vals = []
                            idx = 0
                            while f"{h5_group}/{idx}" in f:
                                weight_vals.append(f[f"{h5_group}/{idx}"][()])
                                idx += 1
                                
                            if len(weight_vals) == len(layer.weights):
                                layer.set_weights(weight_vals)
                print("  Successfully loaded weights manually.")
            except Exception as e:
                print(f"[WARN] Failed to load Keras 3 encoder manually: {e}. Falling back to default loader.")
                encoder = tf.keras.models.load_model(encoder_path)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        elif h5py.is_hdf5(encoder_path):
            print(f"Encoder HDF5 model_weights format detected. Loading weights into encoder.")
            try:
                with h5py.File(encoder_path, 'r') as f:
                    mw = f['model_weights']
                    input_dim = mw['dense']['dense']['kernel:0'].shape[0]
                    encoder = build_flow_encoder(input_dim=input_dim)
                    
                    if 'dense' in mw:
                        encoder.get_layer('dense').set_weights([mw['dense']['dense']['kernel:0'][()], mw['dense']['dense']['bias:0'][()]])
                    if 'layer_normalization' in mw:
                        encoder.get_layer('layer_normalization').set_weights([mw['layer_normalization']['layer_normalization']['gamma:0'][()], mw['layer_normalization']['layer_normalization']['beta:0'][()]])
                    if 'dense_1' in mw:
                        encoder.get_layer('dense_1').set_weights([mw['dense_1']['dense_1']['kernel:0'][()], mw['dense_1']['dense_1']['bias:0'][()]])
                    if 'layer_normalization_1' in mw:
                        encoder.get_layer('layer_normalization_1').set_weights([mw['layer_normalization_1']['layer_normalization_1']['gamma:0'][()], mw['layer_normalization_1']['layer_normalization_1']['beta:0'][()]])
                    if 'embedding' in mw:
                        encoder.get_layer('embedding').set_weights([mw['embedding']['embedding']['kernel:0'][()], mw['embedding']['embedding']['bias:0'][()]])
                print(f"  Successfully loaded HDF5 encoder weights. Input dimension: {input_dim}")
            except Exception as e:
                print(f"[WARN] Failed to load HDF5 encoder weights: {e}. Attempting default load_model.")
                encoder = tf.keras.models.load_model(encoder_path)
        else:
            encoder = tf.keras.models.load_model(encoder_path)
    elif encoder is None:
        if not allow_demo:
            raise FileNotFoundError(f"Frozen encoder not found at {ckpt_base}/encoder_frozen.keras. Run SSL training first.")
        print("[WARN] No frozen encoder found; using fresh encoder for demo.")
        encoder = build_flow_encoder(input_dim=80)

    # Override encoder weights from snapshot if specified
    if args.encoder_snapshot:
        snapshot_ckpt = tf.train.latest_checkpoint(args.encoder_snapshot)
        if snapshot_ckpt:
            print(f"Loading encoder from snapshot: {snapshot_ckpt}")
            tf.train.Checkpoint(encoder=encoder).restore(snapshot_ckpt).expect_partial()
        else:
            print(f"[WARN] No snapshot found at {args.encoder_snapshot}, using default encoder")

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
            args.dataset, split=args.split or "test", label_col=args.label_col
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
        # Load task-specific preprocessor and label column if available
        if ds == "unsw":
            task_label_col = "label" if task_name == "intrusion" else "attack_cat"
        elif args.label_col and args.task != "all":
            task_label_col = args.label_col
        elif task_name == "intrusion":
            task_label_col = args.label_col or "Label"
        else:
            # Auto-detect attack category column name (varies by dataset)
            attack_cat_candidates = ["AttackCategory", "attack_cat", "Attack_cat", "attack_category"]
            task_label_col = None
            for candidate in attack_cat_candidates:
                if candidate in df.columns:
                    task_label_col = candidate
                    break
            if task_label_col is None:
                raise ValueError(
                    f"Could not find attack category column for task '{task_name}'. "
                    f"Tried: {attack_cat_candidates}. Available columns: {list(df.columns)}"
                )
        task_prep_path = f"{ckpt_base}/preprocessor.pkl" if task_name == "intrusion" else f"{ckpt_base}/preprocessor_{task_name}.pkl"
        if os.path.exists(task_prep_path):
            task_prep = FlowPreprocessor.load(task_prep_path)
            feat, lbl_raw = task_prep.transform(df, label_col=task_label_col)
        else:
            task_prep = preprocessor
            feat, lbl_raw = features, labels_raw

        head = build_task_head(task_name, embed_dim=embed_dim)
        ckpt = (
            tf.train.latest_checkpoint(f"{ckpt_base}/{task_name}/best")
            or tf.train.latest_checkpoint(f"{ckpt_base}/{task_name}")
            or tf.train.latest_checkpoint(f"./checkpoints/{task_name}/best")
            or tf.train.latest_checkpoint(f"./checkpoints/{task_name}")
        )
        if ckpt:
            tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()
        else:
            if not allow_demo:
                raise FileNotFoundError(f"No checkpoint for {task_name}. Train the task head first.")
            print(f"[WARN] No checkpoint for {task_name}; using random weights.")

        binary_labels = make_task_labels(task_name, lbl_raw, task_prep.get_classes())

        # Auto-load saved validation dataset if available
        val_feat, val_binary_labels = None, None
        val_path = f"{ckpt_base}/{task_name}/val_data.npz"
        if os.path.exists(val_path):
            try:
                val_data = np.load(val_path)
                val_feat = val_data["val_x"]
                val_binary_labels = val_data["val_y"]
                print(f"  Loaded saved validation dataset from {val_path} ({len(val_feat)} samples)")
            except Exception as e:
                print(f"  [WARN] Could not load validation dataset at {val_path}: {e}")

        metrics = evaluate_head(encoder, head, feat, binary_labels, task_name,
                                eval_dir=eval_dir, val_features=val_feat, val_labels=val_binary_labels)
        results[task_name] = [metrics["f1"]]

    # Print forgetting summary
    if len(results) > 1:
        compute_forgetting_matrix(results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
