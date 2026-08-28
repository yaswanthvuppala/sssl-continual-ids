"""
predict.py - CLI entry point for local IDS inference.

Usage:
    python inference/predict.py
    python inference/predict.py --csv path/to/flows.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly.autoencoder_detector import AutoencoderDetector
from classifiers.base_head import build_classifier_head
from classifiers.dos_head import build_dos_head
from classifiers.exfiltration_head import build_exfiltration_head
from classifiers.scan_head import build_scan_head
from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from encoder.flow_encoder import build_flow_encoder
from inference.inference_engine import IDSInferenceEngine
from training.train_task import load_frozen_encoder


def load_encoder(path: str, allow_demo: bool) -> tf.keras.Model:
    if os.path.exists(path):
        return load_frozen_encoder(path)
    if not allow_demo:
        raise FileNotFoundError(f"Frozen encoder not found at {path}. Run SSL training first.")
    print(f"[WARN] Frozen encoder not found at {path}. Building a fresh one for demo.")
    enc = build_flow_encoder(input_dim=80)
    enc.trainable = False
    return enc


def load_heads(encoder_out_dim: int, ckpt_base: str, allow_demo: bool) -> dict:
    tasks = [
        ("dos_ddos", build_dos_head, "dos"),
        ("port_scan", build_scan_head, "port_scan"),
    ]
    if os.path.exists(os.path.join(ckpt_base, "intrusion")) or not allow_demo:
        tasks.append(("intrusion", lambda embed_dim: build_classifier_head(embed_dim, 2, "intrusion_head"), "intrusion"))
    else:
        tasks.append(("exfiltration", build_exfiltration_head, "exfiltration"))

    heads = {}
    for name, builder, task_dir in tasks:
        head = builder(embed_dim=encoder_out_dim)
        ckpt = (
            tf.train.latest_checkpoint(f"{ckpt_base}/{task_dir}/best")
            or tf.train.latest_checkpoint(f"{ckpt_base}/{task_dir}")
        )
        if ckpt:
            try:
                tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()
                print(f"  Loaded checkpoint for {name} from {ckpt}")
            except Exception as e:
                print(f"  [WARN] Incompatible checkpoint at {ckpt} ({e}); using random weights.")
        else:
            if not allow_demo:
                raise FileNotFoundError(f"No checkpoint for {name} found in '{ckpt_base}'. Train the {task_dir} head first.")
            print(f"  [INFO] No checkpoint for {name} in '{ckpt_base}'; using uninitialized weights (0% zero-shot / demo mode).")
        heads[name] = head
    return heads


def load_calibration_and_thresholds(eval_dir: str, heads: dict) -> tuple[dict, dict]:
    thresholds = {}
    temperatures = {}
    name_mapping = {
        "dos_ddos": "dos",
        "port_scan": "port_scan",
        "intrusion": "intrusion",
        "exfiltration": "exfiltration",
    }
    if not os.path.exists(eval_dir):
        print(f"[WARN] Evaluation logs directory {eval_dir} not found. Using defaults.")
        return thresholds, temperatures

    for head_name in heads:
        task_name = name_mapping.get(head_name, head_name)
        metrics_path = os.path.join(eval_dir, f"metrics_{task_name}.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    opt_threshold = json.load(f).get("optimal_threshold")
                if opt_threshold is not None:
                    thresholds[head_name] = float(opt_threshold)
                    print(f"  Loaded optimal threshold for {head_name}: {opt_threshold:.6f}")
            except Exception as e:
                print(f"[WARN] Failed to load optimal threshold for {head_name} from {metrics_path}: {e}")

        temp_path = os.path.join(eval_dir, f"temperature_{task_name}.json")
        if os.path.exists(temp_path):
            try:
                with open(temp_path, "r") as f:
                    temp = json.load(f).get("temperature")
                if temp is not None:
                    temperatures[head_name] = float(temp)
                    print(f"  Loaded temperature scaling for {head_name}: {temp:.6f}")
            except Exception as e:
                print(f"[WARN] Failed to load temperature scaling for {head_name} from {temp_path}: {e}")
    return thresholds, temperatures


def load_prediction_features(csv_path: str, ckpt_base: str, encoder: tf.keras.Model) -> np.ndarray:
    loader = FlowDatasetLoader(data_path=".")
    df = loader.load_csv(csv_path)
    preprocessor_path = f"{ckpt_base}/preprocessor.pkl"
    if not os.path.exists(preprocessor_path) and os.path.exists("./checkpoints/preprocessor.pkl"):
        preprocessor_path = "./checkpoints/preprocessor.pkl"
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run training first.")
    features, _ = FlowPreprocessor.load(preprocessor_path).transform(df)
    if features.shape[1] != encoder.input_shape[-1]:
        raise ValueError(
            f"CSV features have {features.shape[1]} columns but encoder expects {encoder.input_shape[-1]}. "
            "Use the preprocessor and encoder from the same training run."
        )
    return features


def main():
    parser = argparse.ArgumentParser(description="IDS Inference - Local Prediction")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with flow features")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of synthetic samples if no CSV")
    parser.add_argument("--dataset_name", type=str, default="default", help="Dataset identifier for scoped checkpoints")
    parser.add_argument("--demo", action="store_true", help="Allow synthetic/random fallback components for smoke tests")
    args = parser.parse_args()

    ckpt_base = f"./checkpoints/{args.dataset_name}"
    allow_demo = args.demo or args.csv is None

    encoder = load_encoder(f"{ckpt_base}/encoder_frozen.keras", allow_demo=allow_demo)
    embed_dim = encoder.output_shape[-1]
    heads = load_heads(embed_dim, ckpt_base=ckpt_base, allow_demo=allow_demo)

    anomaly_det = AutoencoderDetector(embed_dim=embed_dim)
    ae_path = f"{ckpt_base}/anomaly_ae.keras"
    if os.path.exists(ae_path):
        anomaly_det.load(ae_path)
    elif os.path.exists("./checkpoints/anomaly_ae.keras"):
        print("[INFO] Falling back to legacy anomaly AE path")
        anomaly_det.load("./checkpoints/anomaly_ae.keras")
    elif not allow_demo:
        raise FileNotFoundError(f"Anomaly autoencoder not found at {ae_path}. Train or provide the detector first.")
    else:
        print("[WARN] Anomaly autoencoder not trained. Using untrained detector (demo mode).")

    thresholds, temperatures = load_calibration_and_thresholds(f"./logs/{args.dataset_name}/eval", heads)
    engine = IDSInferenceEngine(
        encoder=encoder,
        heads=heads,
        anomaly_detector=anomaly_det,
        attack_thresholds=thresholds,
        anomaly_threshold=0.65,
        temperatures=temperatures,
    )

    if args.csv:
        features = load_prediction_features(args.csv, ckpt_base, encoder)
    else:
        np.random.seed(99)
        feat_dim = encoder.input_shape[1] if encoder.input_shape is not None else 80
        features = np.random.randn(args.num_samples, feat_dim).astype(np.float32)
        print(f"Using {args.num_samples} synthetic flow samples of dimension {feat_dim} for demo.\n")

    print("=" * 90)
    print(f" {'TIMESTAMP':<26} {'SEV':<10} {'FLOW':<12} {'TYPE':<20} {'CONF':>6} {'ANOM':>6}")
    print("=" * 90)
    for alert in engine.score_batch(features):
        label = alert.attack_type if alert.attack_type else "BENIGN"
        print(f" {alert.timestamp:<26} {alert.severity:<10} {alert.flow_id:<12} {label:<20} {alert.confidence:>6.3f} {alert.anomaly_score:>6.3f}")
    print("=" * 90)
    print(f"Total flows scored: {len(features)}")


if __name__ == "__main__":
    main()