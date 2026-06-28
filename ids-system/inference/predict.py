"""
predict.py — CLI entry point for local IDS inference.

Usage:
    python inference/predict.py                       # score synthetic samples
    python inference/predict.py --csv path/to/flows.csv
"""
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from anomaly.autoencoder_detector import AutoencoderDetector
from inference.inference_engine import IDSInferenceEngine


def load_keras3_weights_manually(model, zip_path: str):
    """Loads Keras 3 weights manually and robustly using type-and-order matching."""
    import zipfile
    import tempfile
    import shutil
    import h5py
    import os
    temp_dir = tempfile.mkdtemp(dir=".")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            weights_path = zip_ref.extract('model.weights.h5', path=temp_dir)
            with h5py.File(weights_path, 'r') as f:
                # 1. Parse all H5 layers and categorize them
                h5_layers = []
                layers_root = f['layers']
                for grp_name in layers_root.keys():
                    vars_path = f"layers/{grp_name}/vars"
                    if vars_path in f:
                        name_attr = f[vars_path].attrs.get('name')
                        if name_attr:
                            if isinstance(name_attr, bytes):
                                name_attr = name_attr.decode('utf-8')
                            
                            weight_vals = []
                            idx = 0
                            while f"{vars_path}/{idx}" in f:
                                weight_vals.append(f[f"{vars_path}/{idx}"][()])
                                idx += 1
                            
                            h5_layers.append({
                                'grp_name': grp_name,
                                'vars_path': vars_path,
                                'saved_name': name_attr,
                                'weights': weight_vals,
                                'count': len(weight_vals)
                            })
                
                # 2. Build mapping using exact names and types/orders
                custom_names_in_h5 = {}
                unnamed_dense_in_h5 = []
                unnamed_bn_in_h5 = []
                
                for h5_layer in h5_layers:
                    saved_name = h5_layer['saved_name']
                    is_default = (
                        saved_name.startswith("dense") or 
                        saved_name.startswith("batch_normalization") or
                        saved_name.startswith("dropout") or
                        saved_name.startswith("input")
                    )
                    if not is_default:
                        custom_names_in_h5[saved_name] = h5_layer
                    else:
                        if h5_layer['count'] == 2:  # Dense
                            unnamed_dense_in_h5.append(h5_layer)
                        elif h5_layer['count'] == 4:  # BatchNormalization
                            unnamed_bn_in_h5.append(h5_layer)
                
                keras2_dense_unnamed = []
                keras2_bn_unnamed = []
                
                for layer in model.layers:
                    if not layer.weights:
                        continue
                    name = layer.name
                    is_custom = name in ['ae_latent', 'ae_reconstruction', 'embedding']
                    if is_custom:
                        h5_layer = custom_names_in_h5.get(name)
                        if h5_layer:
                            layer.set_weights(h5_layer['weights'])
                            print(f"  Matched custom layer '{name}' -> H5 group '{h5_layer['grp_name']}'")
                        else:
                            print(f"  [ERROR] Custom layer '{name}' not found in weights file.")
                    else:
                        if len(layer.weights) == 2:
                            keras2_dense_unnamed.append(layer)
                        elif len(layer.weights) == 4:
                            keras2_bn_unnamed.append(layer)
                
                # Match unnamed Dense layers by order
                for i, layer in enumerate(keras2_dense_unnamed):
                    if i < len(unnamed_dense_in_h5):
                        h5_layer = unnamed_dense_in_h5[i]
                        layer.set_weights(h5_layer['weights'])
                        print(f"  Matched unnamed Dense layer #{i} '{layer.name}' -> H5 group '{h5_layer['grp_name']}'")
                        
                # Match unnamed BN layers by order
                for i, layer in enumerate(keras2_bn_unnamed):
                    if i < len(unnamed_bn_in_h5):
                        h5_layer = unnamed_bn_in_h5[i]
                        layer.set_weights(h5_layer['weights'])
                        print(f"  Matched unnamed BN layer #{i} '{layer.name}' -> H5 group '{h5_layer['grp_name']}'")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def load_encoder(path: str = "./checkpoints/encoder_frozen.keras") -> tf.keras.Model:
    from encoder.flow_encoder import build_flow_encoder
    if not os.path.exists(path):
        print(f"[WARN] Frozen encoder not found at {path}. Building a fresh one for demo.")
        enc = build_flow_encoder(input_dim=80)
        enc.trainable = False
        return enc
        
    import zipfile
    if zipfile.is_zipfile(path):
        print(f"Encoder Keras 3 zip format detected. Loading weights manually.")
        import tempfile
        import shutil
        import h5py
        temp_dir = tempfile.mkdtemp(dir=".")
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                weights_path = zip_ref.extract('model.weights.h5', path=temp_dir)
                with h5py.File(weights_path, 'r') as f:
                    input_dim = f['layers/dense/vars/0'].shape[0]
            
            model = build_flow_encoder(input_dim=input_dim)
            load_keras3_weights_manually(model, path)
            model.trainable = False
            return model
        except Exception as e:
            print(f"[WARN] Failed to load Keras 3 encoder manually: {e}. Falling back to default loader.")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    model = tf.keras.models.load_model(path)
    model.trainable = False
    return model


def load_heads(encoder_out_dim: int, ckpt_base: str = "./checkpoints") -> dict:
    """Load saved heads or build fresh ones for demo."""
    from classifiers.dos_head import build_dos_head
    from classifiers.scan_head import build_scan_head
    from classifiers.base_head import build_classifier_head
    from classifiers.exfiltration_head import build_exfiltration_head

    heads = {}
    tasks = [
        ("dos_ddos", build_dos_head, "dos"),
        ("port_scan", build_scan_head, "port_scan"),
    ]
    
    # Check if intrusion checkpoint exists, else fall back to exfiltration
    if os.path.exists(os.path.join(ckpt_base, "intrusion")) or os.path.exists("./checkpoints/intrusion"):
        tasks.append(("intrusion", lambda embed_dim: build_classifier_head(embed_dim, num_classes=2, name="intrusion_head"), "intrusion"))
    else:
        tasks.append(("exfiltration", build_exfiltration_head, "exfiltration"))

    for name, builder, task_dir in tasks:
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


def load_calibration_and_thresholds(eval_dir: str, heads: dict) -> tuple:
    """
    Loads optimal thresholds and temperature calibration parameters from JSON files in eval_dir.
    """
    thresholds = {}
    temperatures = {}
    
    # Map head keys to task file names
    name_mapping = {
        "dos_ddos": "dos",
        "port_scan": "port_scan",
        "intrusion": "intrusion",
        "exfiltration": "exfiltration"
    }

    if not os.path.exists(eval_dir):
        print(f"[WARN] Evaluation logs directory {eval_dir} not found. Using defaults.")
        return thresholds, temperatures

    for head_name in heads.keys():
        task_name = name_mapping.get(head_name, head_name)
        
        # Load optimal thresholds from metrics_{task_name}.json
        metrics_path = os.path.join(eval_dir, f"metrics_{task_name}.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                opt_threshold = data.get("optimal_threshold")
                if opt_threshold is not None:
                    thresholds[head_name] = float(opt_threshold)
                    print(f"  Loaded optimal threshold for {head_name}: {opt_threshold:.6f}")
            except Exception as e:
                print(f"[WARN] Failed to load optimal threshold for {head_name} from {metrics_path}: {e}")
        
        # Load temperature from temperature_{task_name}.json
        temp_path = os.path.join(eval_dir, f"temperature_{task_name}.json")
        if os.path.exists(temp_path):
            try:
                with open(temp_path, "r") as f:
                    data = json.load(f)
                temp = data.get("temperature")
                if temp is not None:
                    temperatures[head_name] = float(temp)
                    print(f"  Loaded temperature scaling for {head_name}: {temp:.6f}")
            except Exception as e:
                print(f"[WARN] Failed to load temperature scaling for {head_name} from {temp_path}: {e}")

    return thresholds, temperatures


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

    # Load optimal thresholds and temperature calibration
    eval_dir = f"./logs/{args.dataset_name}/eval"
    thresholds, temperatures = load_calibration_and_thresholds(eval_dir, heads)

    engine = IDSInferenceEngine(
        encoder=encoder,
        heads=heads,
        anomaly_detector=anomaly_det,
        attack_thresholds=thresholds,
        anomaly_threshold=0.65,
        temperatures=temperatures,
    )

    # --- Prepare input ---
    if args.csv:
        loader = FlowDatasetLoader(data_path=".")
        df = loader.load_csv(args.csv)
        preprocessor = FlowPreprocessor()
        features, _ = preprocessor.fit_transform(df)
    else:
        np.random.seed(99)
        feat_dim = encoder.input_shape[1] if hasattr(encoder, 'input_shape') and encoder.input_shape is not None else 80
        features = np.random.randn(args.num_samples, feat_dim).astype(np.float32)
        print(f"Using {args.num_samples} synthetic flow samples of dimension {feat_dim} for demo.\n")

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
