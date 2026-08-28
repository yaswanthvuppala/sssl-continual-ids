"""
train_anomaly.py — Train and calibrate Autoencoder on SSL embeddings for Zero-Day Anomaly Detection.

Usage:
    python training/train_anomaly.py --dataset cicids2017 --data_path ../CICIDS2017
    python training/train_anomaly.py --dataset unsw --train_csv ../IDS-UNSW_NB/UNSW_NB15_training-set.csv
    python training/train_anomaly.py --dataset kddcup99 --data_path ../KDDCUP99
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from encoder.flow_encoder import build_flow_encoder
from anomaly.autoencoder_detector import AutoencoderDetector
from training.train_task import load_frozen_encoder, copy_keras_weights_from_zip


def batch_encode(encoder: tf.keras.Model, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Encode feature matrix through frozen encoder in batches."""
    outputs = []
    for i in range(0, len(features), batch_size):
        batch = tf.constant(features[i:i+batch_size], dtype=tf.float32)
        emb = encoder(batch, training=False).numpy()
        outputs.append(emb)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, encoder.output_shape[-1]))


def train_anomaly_detector(
    dataset_name: str,
    dataset: str = None,
    data_path: str = None,
    train_csv: str = None,
    label_col: str = None,
    preprocessor_path: str = None,
    epochs: int = 20,
    batch_size: int = 256,
    latent_dim: int = 16,
    val_split: float = 0.2,
    percentile: float = 95.0,
):
    ckpt_base = f"./checkpoints/{dataset_name}"
    log_base = f"./logs/{dataset_name}"
    os.makedirs(ckpt_base, exist_ok=True)
    os.makedirs(log_base, exist_ok=True)

    print("=" * 70)
    print(f"  TRAINING ANOMALY AUTOENCODER (ZERO-DAY DETECTOR) — {dataset_name.upper()}")
    print("=" * 70)

    # 1. Resolve Preprocessor
    prep_path = preprocessor_path or f"{ckpt_base}/preprocessor.pkl"
    if not os.path.exists(prep_path):
        for fallback_prep in [
            f"{ckpt_base}/preprocessor_dos.pkl",
            f"{ckpt_base}/preprocessor_port_scan.pkl",
            "./checkpoints/preprocessor.pkl",
        ]:
            if os.path.exists(fallback_prep):
                prep_path = fallback_prep
                break

    if not os.path.exists(prep_path):
        raise FileNotFoundError(
            f"Preprocessor not found at {prep_path}. Run SSL pretraining first."
        )

    preprocessor = FlowPreprocessor.load(prep_path)
    print(f"Loaded preprocessor from {prep_path}")

    # 2. Resolve Frozen Encoder
    encoder_path = f"{ckpt_base}/encoder_frozen.keras"
    if not os.path.exists(encoder_path) and os.path.exists("./checkpoints/encoder_frozen.keras"):
        encoder_path = "./checkpoints/encoder_frozen.keras"

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Frozen encoder not found at {encoder_path}. Run SSL pretraining first."
        )

    encoder = load_frozen_encoder(encoder_path)
    encoder.trainable = False
    embed_dim = encoder.output_shape[-1]
    print(f"Loaded frozen encoder (Embedding dim: {embed_dim})")

    # 3. Load Dataset and Extract Benign Flows
    loader = FlowDatasetLoader(data_path=data_path or ".")
    if dataset:
        df = loader.load_dataset(dataset, split="train", label_col=label_col or "Label")
    elif train_csv:
        df = loader.load_csv(train_csv, label_col=label_col)
    else:
        raise ValueError("Must provide either --dataset or --train_csv")

    # Identify benign/normal rows
    if "AttackCategory" in df.columns:
        benign_mask = df["AttackCategory"].astype(str).str.lower().isin(["normal", "benign"])
    elif "Label" in df.columns:
        benign_mask = df["Label"].astype(str).str.lower().isin(["normal", "benign", "0", "false"])
    elif "label" in df.columns:
        benign_mask = df["label"].astype(str).str.lower().isin(["normal", "benign", "0", "false"])
    elif label_col and label_col in df.columns:
        benign_mask = df[label_col].astype(str).str.lower().isin(["normal", "benign", "0", "false"])
    else:
        raise ValueError("Could not locate a valid benign label column in dataset.")

    df_benign = df[benign_mask]
    if len(df_benign) == 0:
        raise ValueError("No benign/normal traffic found to train the anomaly autoencoder!")

    print(f"Extracted {len(df_benign):,} benign flow samples for autoencoder training.")

    # 4. Preprocess and Encode Benign Traffic
    norm_label_col = "AttackCategory" if "AttackCategory" in df.columns else (label_col or "Label" if "Label" in df.columns else "label")
    features_benign, _ = preprocessor.transform(df_benign, label_col=norm_label_col)
    
    print("Encoding normal features into SSL latent embeddings...")
    normal_embeddings = batch_encode(encoder, features_benign, batch_size=2048)
    print(f"Normal embeddings matrix shape: {normal_embeddings.shape}")

    # 5. Train Autoencoder & Calibrate Threshold
    detector = AutoencoderDetector(embed_dim=embed_dim, latent_dim=latent_dim)
    detector.train(normal_embeddings, epochs=epochs, batch_size=batch_size, val_split=val_split)
    
    # Save checkpoint
    save_path = f"{ckpt_base}/anomaly_ae.keras"
    detector.save(save_path)
    print(f"\n[SUCCESS] Anomaly Autoencoder trained and saved to {save_path}")
    print(f"Calibrated Zero-Day Decision Threshold: {detector.threshold:.6f}")
    return detector


def main():
    parser = argparse.ArgumentParser(description="Train Anomaly Autoencoder for Zero-Day Detection")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw", "anoshift"], default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    ds_name = args.dataset_name or args.dataset or "default"
    data_path = args.data_path
    if not data_path and args.dataset:
        data_path = {
            "cicids2017": "../CICIDS2017",
            "kddcup99": "../KDDCUP99",
            "unsw": "../IDS-UNSW_NB",
            "anoshift": "./data/anoshift",
        }.get(args.dataset)

    train_csv = args.train_csv
    if not train_csv and args.dataset == "unsw":
        train_csv = "../IDS-UNSW_NB/UNSW_NB15_training-set.csv"

    train_anomaly_detector(
        dataset_name=ds_name,
        dataset=args.dataset if args.dataset != "unsw" else None,
        data_path=data_path,
        train_csv=train_csv,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        val_split=args.val_split,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
