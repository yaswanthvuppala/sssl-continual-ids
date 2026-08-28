"""
evaluate_zeroday.py — Evaluate SSSL-IDS performance under Zero-Day / Open-World Attacks.

Usage:
    # Evaluate a specific zero-day attack on CICIDS2017 (e.g. infiltration)
    python training/evaluate_zeroday.py --dataset cicids2017 --data_path ../CICIDS2017 --zeroday_attack infiltration

    # Evaluate all held-out zero-day attacks on UNSW-NB15
    python training/evaluate_zeroday.py --dataset unsw --test_csv ../IDS-UNSW_NB/UNSW_NB15_testing-set.csv

    # Evaluate on KDD Cup 99
    python training/evaluate_zeroday.py --dataset kddcup99 --data_path ../KDDCUP99 --zeroday_attack r2l
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from anomaly.autoencoder_detector import AutoencoderDetector
from inference.inference_engine import IDSInferenceEngine
from inference.predict import load_encoder, load_heads, load_calibration_and_thresholds


def batch_forward(model: tf.keras.Model, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Run model inference in batches."""
    outputs = []
    for i in range(0, len(x), batch_size):
        batch = tf.constant(x[i:i+batch_size], dtype=tf.float32)
        out = model(batch, training=False).numpy()
        outputs.append(out)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.output_shape[-1]))


def plot_zeroday_analysis(
    benign_errors: np.ndarray,
    known_errors: np.ndarray,
    zeroday_errors: np.ndarray,
    threshold: float,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    prec_curve: np.ndarray,
    rec_curve: np.ndarray,
    pr_auc: float,
    conf_matrix_3way: np.ndarray,
    class_names: List[str],
    zeroday_name: str,
    save_path: str,
):
    """Generates a 4-panel publication-quality Zero-Day evaluation dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Zero-Day Attack Evaluation Analysis: '{zeroday_name.upper()}'", fontsize=16, fontweight="bold")

    # 1. Reconstruction Error Distribution (KDE/Histogram)
    ax1 = axes[0, 0]
    sns.kdeplot(benign_errors, ax=ax1, label="Benign Flows", color="#2ecc71", fill=True, alpha=0.35, linewidth=2)
    if len(known_errors) > 0:
        sns.kdeplot(known_errors, ax=ax1, label="Known Attacks", color="#e67e22", fill=True, alpha=0.25, linewidth=1.5)
    sns.kdeplot(zeroday_errors, ax=ax1, label=f"Zero-Day ({zeroday_name})", color="#e74c3c", fill=True, alpha=0.45, linewidth=2)
    ax1.axvline(threshold, color="black", linestyle="--", linewidth=1.8, label=f"Anomaly Threshold ({threshold:.4f})")
    ax1.set_title("SSL Autoencoder Reconstruction Error Distribution", fontweight="bold")
    ax1.set_xlabel("Reconstruction Error (MSE)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", frameon=True)

    # 2. ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"Anomaly ROC (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_title("Zero-Day vs. Benign ROC Curve", fontweight="bold")
    ax2.set_xlabel("False Positive Rate (FAR)")
    ax2.set_ylabel("True Positive Rate (Detection Rate)")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Precision-Recall Curve
    ax3 = axes[1, 0]
    ax3.plot(rec_curve, prec_curve, color="#9b59b6", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_title("Zero-Day Precision-Recall Curve", fontweight="bold")
    ax3.set_xlabel("Recall (Detection Rate)")
    ax3.set_ylabel("Precision")
    ax3.legend(loc="lower left", frameon=True)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Open-World 3-Way Confusion Matrix
    ax4 = axes[1, 1]
    sns.heatmap(
        conf_matrix_3way, annot=True, fmt="d", cmap="Blues", ax=ax4,
        xticklabels=["Benign", "Known Attack", "Zero-Day / Unknown"],
        yticklabels=class_names, cbar=False, annot_kws={"size": 11, "weight": "bold"}
    )
    ax4.set_title("Open-World Decision Confusion Matrix", fontweight="bold")
    ax4.set_xlabel("Predicted Alert")
    ax4.set_ylabel("Ground Truth")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [SAVED] Zero-Day Analysis Dashboard saved to: {save_path}")


def evaluate_single_zeroday(
    engine: IDSInferenceEngine,
    features: np.ndarray,
    categories: np.ndarray,
    known_attack_names: List[str],
    zeroday_name: str,
    dataset_name: str,
    eval_dir: str,
    plot_dir: str,
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"  EVALUATING ZERO-DAY ATTACK: '{zeroday_name.upper()}' ({dataset_name.upper()})")
    print(f"{'='*70}")

    cat_series = pd.Series(categories).astype(str).str.lower()
    is_benign = cat_series.isin(["normal", "benign"]).to_numpy()
    is_zeroday = (cat_series == zeroday_name.lower()).to_numpy()
    is_known = (~is_benign) & (~is_zeroday)

    num_benign = int(np.sum(is_benign))
    num_zeroday = int(np.sum(is_zeroday))
    num_known = int(np.sum(is_known))

    print(f"  Sample Counts — Benign: {num_benign:,} | Known Attacks: {num_known:,} | Zero-Day '{zeroday_name}': {num_zeroday:,}")
    if num_zeroday == 0:
        print(f"  [WARN] Zero-day attack '{zeroday_name}' has 0 samples in test set! Skipping.")
        return {}

    # Run batched inference through full IDS engine
    print(f"  Running IDS Inference Engine across {len(features):,} flows...")
    alerts = engine.score_batch(features)

    predicted_labels = [a.attack_type for a in alerts]
    anomaly_scores = np.array([a.anomaly_score for a in alerts])

    # Classify predictions into 3 categories:
    # 0 = Benign (None), 1 = Known Attack Head, 2 = "zero-day / unknown"
    pred_cat_3way = np.zeros(len(features), dtype=int)
    for i, p in enumerate(predicted_labels):
        if p is None:
            pred_cat_3way[i] = 0  # Benign
        elif p == "zero-day / unknown":
            pred_cat_3way[i] = 2  # Zero-day anomaly fallback
        else:
            pred_cat_3way[i] = 1  # Triggered known attack head

    # True Category 3-way:
    # 0 = Benign, 1 = Known Attack, 2 = Zero-Day
    true_cat_3way = np.zeros(len(features), dtype=int)
    true_cat_3way[is_known] = 1
    true_cat_3way[is_zeroday] = 2

    # --- Metrics Computation ---
    # 1. Zero-day specific metrics
    zeroday_indices = np.where(is_zeroday)[0]
    benign_indices = np.where(is_benign)[0]
    known_indices = np.where(is_known)[0]

    # Zero-day detection rate (TPR): fraction predicted as "zero-day / unknown" or any attack
    detected_as_zeroday = np.sum(pred_cat_3way[zeroday_indices] == 2)
    detected_as_known = np.sum(pred_cat_3way[zeroday_indices] == 1)
    missed_as_benign = np.sum(pred_cat_3way[zeroday_indices] == 0)

    dr_zeroday_pure = detected_as_zeroday / max(1, num_zeroday)
    dr_zeroday_total = (detected_as_zeroday + detected_as_known) / max(1, num_zeroday)

    # Benign False Alarm Rate (FAR): fraction of benign predicted as zero-day or known
    benign_flagged_zeroday = np.sum(pred_cat_3way[benign_indices] == 2)
    benign_flagged_known = np.sum(pred_cat_3way[benign_indices] == 1)
    far_zeroday = benign_flagged_zeroday / max(1, num_benign)
    far_total = (benign_flagged_zeroday + benign_flagged_known) / max(1, num_benign)

    # Precision & F1 on zero-day fallback alert
    total_zeroday_alerts = np.sum(pred_cat_3way == 2)
    prec_zeroday = detected_as_zeroday / max(1, total_zeroday_alerts)
    f1_zeroday = (2 * prec_zeroday * dr_zeroday_pure) / max(1e-8, (prec_zeroday + dr_zeroday_pure))

    # Binary evaluation for ROC/PR: Benign vs Zero-Day only
    binary_mask = is_benign | is_zeroday
    binary_true = is_zeroday[binary_mask].astype(int)
    binary_scores = anomaly_scores[binary_mask]

    try:
        roc_auc = float(roc_auc_score(binary_true, binary_scores))
        pr_auc = float(average_precision_score(binary_true, binary_scores))
        fpr, tpr, _ = roc_curve(binary_true, binary_scores)
        prec_curve, rec_curve, _ = precision_recall_curve(binary_true, binary_scores)
    except Exception as e:
        print(f"  [WARN] AUROC computation error: {e}")
        roc_auc, pr_auc = 0.5, 0.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        prec_curve, rec_curve = np.array([1, 0]), np.array([0, 1])

    # 3-Way Confusion Matrix
    cm_3way = confusion_matrix(true_cat_3way, pred_cat_3way, labels=[0, 1, 2])

    print(f"\n  --- ZERO-DAY METRICS SUMMARY ---")
    print(f"  Target Zero-Day Attack Family : {zeroday_name.upper()}")
    print(f"  Pure Zero-Day Recall (Anomaly): {dr_zeroday_pure * 100:.2f}% ({detected_as_zeroday:,}/{num_zeroday:,})")
    print(f"  Total Attack Catch Rate       : {dr_zeroday_total * 100:.2f}% ({(detected_as_zeroday + detected_as_known):,}/{num_zeroday:,})")
    print(f"  Leakage to Known Heads        : {(detected_as_known / max(1, num_zeroday)) * 100:.2f}% ({detected_as_known:,}/{num_zeroday:,})")
    print(f"  False Negatives (Missed)      : {(missed_as_benign / max(1, num_zeroday)) * 100:.2f}% ({missed_as_benign:,}/{num_zeroday:,})")
    print(f"  False Alarm Rate (Benign FAR) : {far_zeroday * 100:.2f}% ({benign_flagged_zeroday:,}/{num_benign:,})")
    print(f"  Zero-Day Precision            : {prec_zeroday * 100:.2f}%")
    print(f"  Zero-Day F1-Score             : {f1_zeroday:.4f}")
    print(f"  Anomaly ROC-AUC               : {roc_auc:.4f}")
    print(f"  Anomaly PR-AUC                : {pr_auc:.4f}")

    print(f"\n  Open-World Confusion Matrix:")
    print(f"                     [Pred Benign]  [Pred Known]  [Pred Zero-Day]")
    print(f"  True Benign       : {cm_3way[0, 0]:>12,d}  {cm_3way[0, 1]:>12,d}  {cm_3way[0, 2]:>14,d}")
    print(f"  True Known Attack : {cm_3way[1, 0]:>12,d}  {cm_3way[1, 1]:>12,d}  {cm_3way[1, 2]:>14,d}")
    print(f"  True Zero-Day     : {cm_3way[2, 0]:>12,d}  {cm_3way[2, 1]:>12,d}  {cm_3way[2, 2]:>14,d}")

    # Generate Visualization Plot
    plot_path = f"{plot_dir}/zeroday_{zeroday_name}.png"
    class_names = ["Benign", "Known Attacks", f"Zero-Day ({zeroday_name})"]
    plot_zeroday_analysis(
        benign_errors=anomaly_scores[benign_indices],
        known_errors=anomaly_scores[known_indices] if num_known > 0 else np.array([]),
        zeroday_errors=anomaly_scores[zeroday_indices],
        threshold=engine.anomaly_threshold,
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        prec_curve=prec_curve, rec_curve=rec_curve, pr_auc=pr_auc,
        conf_matrix_3way=cm_3way,
        class_names=class_names,
        zeroday_name=zeroday_name,
        save_path=plot_path,
    )

    # Save JSON Metrics
    metrics_dict = {
        "dataset": dataset_name,
        "zeroday_attack": zeroday_name,
        "samples_zeroday": num_zeroday,
        "samples_benign": num_benign,
        "samples_known": num_known,
        "zeroday_detection_rate_pure": float(dr_zeroday_pure),
        "zeroday_detection_rate_total": float(dr_zeroday_total),
        "leakage_to_known_heads": float(detected_as_known / max(1, num_zeroday)),
        "missed_as_benign": float(missed_as_benign / max(1, num_zeroday)),
        "false_alarm_rate_benign": float(far_zeroday),
        "total_false_alarm_rate": float(far_total),
        "zeroday_precision": float(prec_zeroday),
        "zeroday_f1": float(f1_zeroday),
        "anomaly_roc_auc": float(roc_auc),
        "anomaly_pr_auc": float(pr_auc),
        "confusion_matrix_3way": cm_3way.tolist(),
        "anomaly_threshold": float(engine.anomaly_threshold),
    }

    metrics_path = f"{eval_dir}/metrics_zeroday_{zeroday_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  [SAVED] Metrics saved to: {metrics_path}")

    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSSL-IDS under Zero-Day Attacks")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw"], default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--zeroday_attack", type=str, default=None,
                        help="Specific attack category to treat as zero-day (e.g. infiltration, web_attack, r2l, u2r, fuzzers). If omitted, evaluates all non-trained classes.")
    parser.add_argument("--anomaly_threshold", type=float, default=None,
                        help="Override anomaly autoencoder decision threshold")
    args = parser.parse_args()

    ds_name = args.dataset_name or args.dataset or "default"
    ckpt_base = f"./checkpoints/{ds_name}"
    eval_dir = f"./logs/{ds_name}/eval"
    plot_dir = f"./logs/{ds_name}/plots"
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Load Preprocessor
    prep_path = f"{ckpt_base}/preprocessor.pkl"
    if not os.path.exists(prep_path):
        for fb in [f"{ckpt_base}/preprocessor_dos.pkl", f"{ckpt_base}/preprocessor_port_scan.pkl", "./checkpoints/preprocessor.pkl"]:
            if os.path.exists(fb):
                prep_path = fb
                break
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor not found at {prep_path}.")
    preprocessor = FlowPreprocessor.load(prep_path)

    # 2. Load Frozen Encoder
    encoder = load_encoder(f"{ckpt_base}/encoder_frozen.keras", allow_demo=False)
    embed_dim = encoder.output_shape[-1]

    # 3. Load Trained Classifier Heads & Thresholds
    heads = load_heads(embed_dim, ckpt_base=ckpt_base, allow_demo=True)
    thresholds, temperatures = load_calibration_and_thresholds(eval_dir, heads)

    # 4. Load or Train Anomaly Detector
    anomaly_detector = AutoencoderDetector(embed_dim=embed_dim)
    ae_path = f"{ckpt_base}/anomaly_ae.keras"
    if not os.path.exists(ae_path) and os.path.exists("./checkpoints/anomaly_ae.keras"):
        ae_path = "./checkpoints/anomaly_ae.keras"

    if os.path.exists(ae_path):
        anomaly_detector.load(ae_path)
    else:
        print(f"[INFO] Anomaly Autoencoder not found at {ae_path}. Auto-training anomaly detector...")
        from training.train_anomaly import train_anomaly_detector
        data_path = args.data_path or {
            "cicids2017": "../CICIDS2017",
            "kddcup99": "../KDDCUP99",
            "unsw": "../IDS-UNSW_NB",
        }.get(args.dataset)
        train_csv = args.test_csv if args.dataset == "unsw" else None
        anomaly_detector = train_anomaly_detector(
            dataset_name=ds_name,
            dataset=args.dataset if args.dataset != "unsw" else None,
            data_path=data_path,
            train_csv="../IDS-UNSW_NB/UNSW_NB15_training-set.csv" if args.dataset == "unsw" else None,
            epochs=15,
        )

    anom_threshold = args.anomaly_threshold if args.anomaly_threshold is not None else anomaly_detector.threshold

    # 5. Build Inference Engine
    engine = IDSInferenceEngine(
        encoder=encoder,
        heads=heads,
        anomaly_detector=anomaly_detector,
        attack_thresholds=thresholds,
        anomaly_threshold=anom_threshold,
        temperatures=temperatures,
    )

    # 6. Load Test Dataset
    data_path = args.data_path
    if not data_path and args.dataset:
        data_path = {
            "cicids2017": "../CICIDS2017",
            "kddcup99": "../KDDCUP99",
            "unsw": "../IDS-UNSW_NB",
        }.get(args.dataset, ".")
    else:
        data_path = args.data_path or "."

    loader = FlowDatasetLoader(data_path=data_path)
    if args.test_csv:
        df = loader.load_csv(args.test_csv, label_col=args.label_col)
    elif args.dataset in {"cicids2017", "kddcup99"}:
        df = loader.load_dataset(args.dataset, split="test", label_col=args.label_col or "Label")
    elif args.dataset == "unsw":
        df = loader.load_csv("../IDS-UNSW_NB/UNSW_NB15_testing-set.csv", label_col=args.label_col)
    else:
        raise ValueError("Must specify either --dataset or --test_csv")

    # Extract categories and features
    if "AttackCategory" in df.columns:
        cat_col = "AttackCategory"
    elif "attack_cat" in df.columns:
        cat_col = "attack_cat"
    elif "Attack_cat" in df.columns:
        cat_col = "Attack_cat"
    elif "AttackLabel" in df.columns:
        cat_col = "AttackLabel"
    else:
        cat_col = "Label"

    categories = df[cat_col].astype(str).str.strip().str.lower().to_numpy()
    features, _ = preprocessor.transform(df, label_col=cat_col)

    # Identify candidate zero-day attacks
    known_attack_names = [name.lower() for name in heads.keys()]
    unique_categories = [c for c in np.unique(categories) if c not in ["normal", "benign"]]

    if args.zeroday_attack:
        target_attacks = [args.zeroday_attack.strip().lower()]
    else:
        # Candidate zero-days: categories not trained as specific heads
        trained_cats = {"dos", "probe", "port_scan", "intrusion", "exfiltration"}
        candidate_zerodays = [c for c in unique_categories if c not in trained_cats]
        target_attacks = candidate_zerodays if candidate_zerodays else unique_categories

    print(f"\nEvaluated Heads: {list(heads.keys())}")
    print(f"Target Zero-Day Attack Families to Test: {target_attacks}")

    all_results = {}
    for attack_name in target_attacks:
        res = evaluate_single_zeroday(
            engine=engine,
            features=features,
            categories=categories,
            known_attack_names=known_attack_names,
            zeroday_name=attack_name,
            dataset_name=ds_name,
            eval_dir=eval_dir,
            plot_dir=plot_dir,
        )
        if res:
            all_results[attack_name] = res

    # Summary table across all zero-day attacks
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print(f"  ALL ZERO-DAY ATTACKS BENCHMARK SUMMARY — {ds_name.upper()}")
        print("=" * 80)
        print(f" {'ATTACK FAMILY':<20} {'COUNT':>8} {'PURE RECALL':>14} {'TOTAL CATCH':>14} {'ROC-AUC':>10} {'F1':>8}")
        print("-" * 80)
        for atk, m in all_results.items():
            print(f" {atk.upper():<20} {m['samples_zeroday']:>8,d} {m['zeroday_detection_rate_pure']*100:>13.2f}% {m['zeroday_detection_rate_total']*100:>13.2f}% {m['anomaly_roc_auc']:>10.4f} {m['zeroday_f1']:>8.4f}")
        print("=" * 80)

    print("\nZero-Day Evaluation complete.")


if __name__ == "__main__":
    main()
