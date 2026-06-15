"""
predict.py — CLI entry point for local IDS inference.

Usage:
    python inference/predict.py                       # score synthetic samples
    python inference/predict.py --csv path/to/flows.csv
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from anomaly.autoencoder_detector import AutoencoderDetector
from inference.inference_engine import IDSInferenceEngine


def load_encoder(path: str = "./checkpoints/encoder_frozen.keras") -> tf.keras.Model:
    if not os.path.exists(path):
        print(f"[WARN] Frozen encoder not found at {path}. Building a fresh one for demo.")
        from encoder.flow_encoder import build_flow_encoder
        enc = build_flow_encoder(input_dim=80)
        enc.trainable = False
        return enc
    model = tf.keras.models.load_model(path)
    model.trainable = False
    return model


def load_heads(encoder_out_dim: int, ckpt_base: str = "./checkpoints") -> dict:
    """Load saved heads or build fresh ones for demo."""
    from classifiers.dos_head import build_dos_head
    from classifiers.scan_head import build_scan_head
    from classifiers.exfiltration_head import build_exfiltration_head

    heads = {}
    for name, builder, task_dir in [
        ("dos_ddos", build_dos_head, "dos"),
        ("port_scan", build_scan_head, "port_scan"),
        ("exfiltration", build_exfiltration_head, "exfiltration"),
    ]:
        head = builder(embed_dim=encoder_out_dim)
        ckpt_dir = f"{ckpt_base}/{task_dir}"
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if not ckpt:
            # Fallback to legacy flat path
            ckpt = tf.train.latest_checkpoint(f"./checkpoints/{task_dir}")
        if ckpt:
            tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()
            print(f"  Loaded checkpoint for {name}")
        else:
            print(f"  [WARN] No checkpoint for {name}; using random weights (demo mode)")
        heads[name] = head
    return heads


def main():
    parser = argparse.ArgumentParser(description="IDS Inference — Local Prediction")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with flow features")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of synthetic samples if no CSV")
    parser.add_argument("--dataset_name", type=str, default="default",
                        help="Dataset identifier for scoping checkpoint paths")
    args = parser.parse_args()

    # Resolve dataset-scoped checkpoint base
    ckpt_base = f"./checkpoints/{args.dataset_name}"

    # --- Load / build components ---
    encoder = load_encoder(f"{ckpt_base}/encoder_frozen.keras")
    embed_dim = encoder.output_shape[-1]
    heads = load_heads(embed_dim, ckpt_base=ckpt_base)

    anomaly_det = AutoencoderDetector(embed_dim=embed_dim)
    ae_path = f"{ckpt_base}/anomaly_ae.keras"
    if os.path.exists(ae_path):
        anomaly_det.load(ae_path)
    elif os.path.exists("./checkpoints/anomaly_ae.keras"):
        print("[INFO] Falling back to legacy anomaly AE path")
        anomaly_det.load("./checkpoints/anomaly_ae.keras")
    else:
        print("[WARN] Anomaly autoencoder not trained. Using untrained detector (demo mode).")

    engine = IDSInferenceEngine(
        encoder=encoder,
        heads=heads,
        anomaly_detector=anomaly_det,
        attack_threshold=0.80,
        anomaly_threshold=0.65,
    )

    # --- Prepare input ---
    if args.csv:
        loader = FlowDatasetLoader(data_path=".")
        df = loader.load_csv(args.csv)
        preprocessor = FlowPreprocessor()
        features, _ = preprocessor.fit_transform(df)
    else:
        np.random.seed(99)
        features = np.random.randn(args.num_samples, 80).astype(np.float32)
        print(f"Using {args.num_samples} synthetic flow samples for demo.\n")

    # --- Run inference ---
    print("=" * 90)
    print(f" {'TIMESTAMP':<26} {'SEV':<10} {'FLOW':<12} {'TYPE':<20} {'CONF':>6} {'ANOM':>6}")
    print("=" * 90)
    alerts = engine.score_batch(features)
    for a in alerts:
        label = a.attack_type if a.attack_type else "BENIGN"
        print(f" {a.timestamp:<26} {a.severity:<10} {a.flow_id:<12} {label:<20} {a.confidence:>6.3f} {a.anomaly_score:>6.3f}")
    print("=" * 90)
    print(f"Total flows scored: {len(alerts)}")


if __name__ == "__main__":
    main()
