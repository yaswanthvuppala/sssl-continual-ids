import os
import sys
import argparse
import shutil
import tempfile
import zipfile
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from data.tf_dataset import make_labeled_dataset, make_unlabeled_dataset, make_balanced_dataset
from sklearn.model_selection import train_test_split
from classifiers.base_head import build_classifier_head
from classifiers.dos_head import build_dos_head
from classifiers.scan_head import build_scan_head
from classifiers.fixmatch_trainer import FixMatchTrainer
from gpm.gpm import GradientProjectionMemory
from gpm.memory_bank import MemoryBank


def copy_keras_weights_from_zip(zip_path: str, temp_dir: str) -> str:
    weights_path = os.path.join(temp_dir, "model.weights.h5")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        with zip_ref.open("model.weights.h5") as src, open(weights_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return weights_path


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
            weights_path = copy_keras_weights_from_zip(zip_path, temp_dir)
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
                unnamed_ln_in_h5 = []
                
                for h5_layer in h5_layers:
                    saved_name = h5_layer['saved_name']
                    is_default = (
                        saved_name.startswith("dense") or 
                        saved_name.startswith("batch_normalization") or
                        saved_name.startswith("layer_normalization") or
                        saved_name.startswith("dropout") or
                        saved_name.startswith("input")
                    )
                    if not is_default:
                        custom_names_in_h5[saved_name] = h5_layer
                    else:
                        if h5_layer['count'] == 2 and h5_layer['weights'][0].ndim == 2:
                            unnamed_dense_in_h5.append(h5_layer)
                        elif h5_layer['count'] == 2 and h5_layer['weights'][0].ndim == 1:
                            unnamed_ln_in_h5.append(h5_layer)
                        elif h5_layer['count'] == 4:
                            unnamed_bn_in_h5.append(h5_layer)
                
                keras2_dense_unnamed = []
                keras2_bn_unnamed = []
                keras2_ln_unnamed = []
                
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
                        if len(layer.weights) == 2 and len(layer.weights[0].shape) == 2:
                            keras2_dense_unnamed.append(layer)
                        elif len(layer.weights) == 2 and len(layer.weights[0].shape) == 1:
                            keras2_ln_unnamed.append(layer)
                        elif len(layer.weights) == 4:
                            keras2_bn_unnamed.append(layer)
                
                # Match unnamed Dense layers by order
                for i, layer in enumerate(keras2_dense_unnamed):
                    if i < len(unnamed_dense_in_h5):
                        h5_layer = unnamed_dense_in_h5[i]
                        layer.set_weights(h5_layer['weights'])
                        print(f"  Matched unnamed Dense layer #{i} '{layer.name}' -> H5 group '{h5_layer['grp_name']}'")
                        
                # Match unnamed LN layers by order
                for i, layer in enumerate(keras2_ln_unnamed):
                    if i < len(unnamed_ln_in_h5):
                        h5_layer = unnamed_ln_in_h5[i]
                        layer.set_weights(h5_layer['weights'])
                        print(f"  Matched unnamed LayerNorm layer #{i} '{layer.name}' -> H5 group '{h5_layer['grp_name']}'")

                # Match unnamed BN layers by order
                for i, layer in enumerate(keras2_bn_unnamed):
                    if i < len(unnamed_bn_in_h5):
                        h5_layer = unnamed_bn_in_h5[i]
                        layer.set_weights(h5_layer['weights'])
                        print(f"  Matched unnamed BN layer #{i} '{layer.name}' -> H5 group '{h5_layer['grp_name']}'")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def load_frozen_encoder(path: str = "./checkpoints/encoder_frozen.keras") -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frozen encoder not found at {path}. Run train_ssl.py first.")
        
    from encoder.flow_encoder import build_flow_encoder
    import zipfile
    if zipfile.is_zipfile(path):
        print(f"Encoder Keras 3 zip format detected. Loading weights manually.")
        import tempfile
        import shutil
        import h5py
        temp_dir = tempfile.mkdtemp(dir=".")
        try:
            weights_path = copy_keras_weights_from_zip(path, temp_dir)
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

    import h5py
    if h5py.is_hdf5(path):
        print(f"Encoder HDF5 model_weights format detected. Loading weights into encoder.")
        try:
            with h5py.File(path, 'r') as f:
                mw = f['model_weights']
                input_dim = mw['dense']['dense']['kernel:0'].shape[0]
                model = build_flow_encoder(input_dim=input_dim)
                
                if 'dense' in mw:
                    model.get_layer('dense').set_weights([mw['dense']['dense']['kernel:0'][()], mw['dense']['dense']['bias:0'][()]])
                if 'layer_normalization' in mw:
                    model.get_layer('layer_normalization').set_weights([mw['layer_normalization']['layer_normalization']['gamma:0'][()], mw['layer_normalization']['layer_normalization']['beta:0'][()]])
                if 'dense_1' in mw:
                    model.get_layer('dense_1').set_weights([mw['dense_1']['dense_1']['kernel:0'][()], mw['dense_1']['dense_1']['bias:0'][()]])
                if 'layer_normalization_1' in mw:
                    model.get_layer('layer_normalization_1').set_weights([mw['layer_normalization_1']['layer_normalization_1']['gamma:0'][()], mw['layer_normalization_1']['layer_normalization_1']['beta:0'][()]])
                if 'embedding_dense' in mw:
                    model.get_layer('embedding_dense').set_weights([mw['embedding_dense']['embedding_dense']['kernel:0'][()], mw['embedding_dense']['embedding_dense']['bias:0'][()]])
                model.trainable = False
                return model
        except Exception as e:
            print(f"[WARN] Manual HDF5 weight loading failed: {e}. Trying standard load_model...")

    try:
        model = tf.keras.models.load_model(path)
        model.trainable = False
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load encoder from {path}: {e}")


def build_task_head(task: str, embed_dim: int) -> tf.keras.Model:
    if task == "intrusion":
        return build_classifier_head(embed_dim=embed_dim, num_classes=2, name="intrusion_head")
    if task == "dos":
        return build_dos_head(embed_dim=embed_dim)
    if task == "port_scan":
        return build_scan_head(embed_dim=embed_dim)
    raise ValueError(f"Unsupported task: {task}")


def make_task_labels(task: str, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    normalized_classes = [str(c).strip().lower() for c in classes]
    if task == "intrusion":
        if len(classes) == 1:
            return np.zeros_like(labels, dtype=np.int32)
        if len(classes) == 2 and set(normalized_classes).issubset({"normal", "attack", "0", "1"}):
            target_indices = [i for i, c in enumerate(normalized_classes) if c in {"attack", "1"}]
            return np.isin(labels, target_indices).astype(np.int32)
        # Multi-class: anything not normal is attack
        target_indices = [i for i, c in enumerate(normalized_classes) if c not in {"normal", "benign", "0", "0.0"}]
        return np.isin(labels, target_indices).astype(np.int32)

    target_names = {
        "dos": ["dos", "ddos", "known_attack"],
        "port_scan": ["portscan", "port scan", "port_scan", "probe", "reconnaissance"],
    }[task]
    target_indices = [i for i, c in enumerate(normalized_classes) if c in target_names]
    if not target_indices:
        raise ValueError(
            f"Could not find task label for '{task}' in classes: {classes.tolist()}. "
            "Use --task intrusion with dataset's 'Label' column."
        )
    return np.isin(labels, target_indices).astype(np.int32)

def main():
    parser = argparse.ArgumentParser(description="Train Task-Specific Classifier")
    parser.add_argument("--task", type=str, required=True, choices=["intrusion", "dos", "port_scan"], help="Task to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Labeled batch size")
    parser.add_argument("--unlabeled_batch_size", type=int, default=128, help="Unlabeled batch size")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw", "anoshift"],
                        default=None, help="Load a supported raw dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset directory or raw data file")
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the training CSV")
    parser.add_argument("--preprocessor_path", type=str, default=None,
                        help="Path to the fitted preprocessor (auto-selected per task if not set)")
    parser.add_argument("--max_labeled", type=int, default=None,
                        help="Optional cap on labeled training samples")
    parser.add_argument("--max_gpm_batches", type=int, default=512,
                        help="Maximum valid labeled batches to use for GPM SVD capture; use 0 to scan all batches")
    parser.add_argument("--dataset_name", type=str, default="default",
                        help="Dataset identifier for scoping output paths")
    parser.add_argument("--balanced", action="store_true",
                        help="Use class-balanced batching (50/50 per batch)")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Number of warmup epochs with no pseudo-label loss")
    parser.add_argument("--label_ratio", type=float, default=0.20,
                        help="Fraction of training data to use as labeled (0.05=5%%, 0.10=10%%, 0.20=20%%, 0.50=50%%, 1.0=100%%)")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Path to pre-trained frozen encoder .keras file")
    parser.add_argument("--unfreeze_encoder", action="store_true",
                        help="Unfreeze encoder for end-to-end fine-tuning with GPM protection")
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="Learning rate for encoder when unfrozen (default: head_lr / 10)")
    args = parser.parse_args()

    print(f"Initializing Continual Learning for Task: {args.task}")

    if not (args.dataset or args.train_csv):
        args.label_col = "Label"
    elif args.label_col is None or (args.label_col == "Label" and args.task != "intrusion"):
        args.label_col = "Label" if args.task == "intrusion" else "AttackCategory"

    # Resolve dataset-scoped base paths
    ds = args.dataset_name
    ckpt_base = f"./checkpoints/{ds}"
    log_base = f"./logs/{ds}"
    os.makedirs(ckpt_base, exist_ok=True)
    os.makedirs(log_base, exist_ok=True)

    # Resolve preprocessor path per task so label encoders never conflict.
    # intrusion uses 'label' (binary 0/1); dos/port_scan use 'attack_cat' (strings).
    if args.preprocessor_path is None:
        if args.task == "intrusion":
            args.preprocessor_path = f"{ckpt_base}/preprocessor.pkl"
        else:
            args.preprocessor_path = f"{ckpt_base}/preprocessor_{args.task}.pkl"
    print(f"Using preprocessor: {args.preprocessor_path}")

    # Resolve encoder path with smart fallbacks
    enc_path = args.encoder_path
    if enc_path is None or not os.path.exists(enc_path):
        base_ds = args.dataset if args.dataset else args.dataset_name.split("_")[0]
        candidates = [
            f"{ckpt_base}/encoder_frozen.keras",
            f"./checkpoints/{base_ds}/encoder_frozen.keras",
            f"../logs_Experimental/{base_ds}/encoder_frozen.keras",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                enc_path = cand
                break

    if enc_path is None or not os.path.exists(enc_path):
        raise FileNotFoundError(
            f"Frozen encoder not found at {ckpt_base}/encoder_frozen.keras. "
            f"Please run train_ssl.py first or specify --encoder_path."
        )

    print(f"Loading encoder from: {enc_path}")
    encoder = load_frozen_encoder(enc_path)

    # Save a copy in ckpt_base if it's not already there
    target_enc_copy = f"{ckpt_base}/encoder_frozen.keras"
    if not os.path.exists(target_enc_copy) and enc_path != target_enc_copy:
        import shutil
        shutil.copyfile(enc_path, target_enc_copy)
        print(f"Saved encoder copy to {target_enc_copy}")
    
    # When unfreezing, load encoder weights from previous task's snapshot if available
    if args.unfreeze_encoder:
        task_order = ["intrusion", "dos", "port_scan"]
        task_idx = task_order.index(args.task)
        if task_idx > 0:
            prev_task = task_order[task_idx - 1]
            prev_snapshot = f"{ckpt_base}/snapshots/after_{prev_task}/encoder"
            prev_ckpt = tf.train.latest_checkpoint(prev_snapshot)
            if prev_ckpt:
                print(f"Loading encoder from previous snapshot: {prev_ckpt}")
                tf.train.Checkpoint(encoder=encoder).restore(prev_ckpt).expect_partial()
            else:
                print(f"[WARN] No previous snapshot found at {prev_snapshot}, using SSL weights")
        encoder.trainable = True
        print(f"Encoder UNFROZEN for end-to-end fine-tuning (encoder_lr={args.encoder_lr or 0.003})")
    
    # Initialize GPM for ALL tasks (intrusion included for gradient basis capture).
    # For intrusion (first task), projection is a no-op with an empty bank,
    # but capture is needed to protect it from subsequent tasks.
    memory_bank = MemoryBank(save_dir=f"{ckpt_base}/gpm")
    memory_bank.load()
    gpm = GradientProjectionMemory(threshold=0.97, memory_bank=memory_bank)
    
    # Initialize Head
    head = build_task_head(args.task, embed_dim=encoder.output_shape[-1])
        
    loader = FlowDatasetLoader(data_path=args.data_path or ".")
    if args.dataset:
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset is used")
        max_samples = args.max_labeled or (50000 if args.dataset == "anoshift" else None)
        df_labeled = loader.load_dataset(
            args.dataset, split="train", label_col=args.label_col, max_samples=max_samples
        )
        expected_input_dim = encoder.input_shape[-1]
        if os.path.exists(args.preprocessor_path):
            preprocessor = FlowPreprocessor.load(args.preprocessor_path)
            refit_needed = (args.task != "intrusion" and set(preprocessor.get_classes()).issubset({"attack", "normal"}))
            try:
                if refit_needed:
                    print(f"[WARN] Preprocessor at {args.preprocessor_path} has binary classes {preprocessor.get_classes()}, refitting for task '{args.task}' on '{args.label_col}'...")
                    raise ValueError("Refitting preprocessor for task-specific attack categories.")
                X_l, y_l = preprocessor.transform(
                    df_labeled, label_col=args.label_col
                )
            except (ValueError, KeyError) as e:
                print(
                    f"[WARN] Saved preprocessor incompatible with label column "
                    f"'{args.label_col}': {e}"
                )
                print("[WARN] Refitting preprocessor on current dataset...")
                preprocessor = FlowPreprocessor()
                X_l, y_l = preprocessor.fit_transform(
                    df_labeled, label_col=args.label_col
                )
                preprocessor.save(args.preprocessor_path)
        else:
            preprocessor = FlowPreprocessor()
            X_l, y_l = preprocessor.fit_transform(
                df_labeled, label_col=args.label_col
            )
            preprocessor.save(args.preprocessor_path)
            print(f"Fitted preprocessor saved to {args.preprocessor_path}")
        if X_l.shape[1] != expected_input_dim:
            raise ValueError(
                f"Training features have {X_l.shape[1]} columns but the frozen "
                f"encoder expects {expected_input_dim}. Re-run SSL pretraining "
                "for this dataset first."
            )
    elif args.train_csv:
        df_labeled = loader.load_csv(args.train_csv, label_col=args.label_col)
        expected_input_dim = encoder.input_shape[-1]
        if os.path.exists(args.preprocessor_path):
            preprocessor = FlowPreprocessor.load(args.preprocessor_path)
            refit_needed = (args.task != "intrusion" and set(preprocessor.get_classes()).issubset({"attack", "normal"}))
            try:
                if refit_needed:
                    print(f"[WARN] Preprocessor at {args.preprocessor_path} has binary classes {preprocessor.get_classes()}, refitting for task '{args.task}' on '{args.label_col}'...")
                    raise ValueError("Refitting preprocessor for task-specific attack categories.")
                X_l, y_l = preprocessor.transform(df_labeled, label_col=args.label_col)
            except (ValueError, KeyError) as e:
                print(f"[WARN] Saved preprocessor incompatible with label column '{args.label_col}': {e}")
                print("[WARN] Refitting preprocessor on current CSV...")
                preprocessor = FlowPreprocessor()
                X_l, y_l = preprocessor.fit_transform(df_labeled, label_col=args.label_col)
                preprocessor.save(args.preprocessor_path)
            if X_l.shape[1] != expected_input_dim:
                print(
                    f"[WARN] Saved preprocessor produces {X_l.shape[1]} features, "
                    f"but encoder expects {expected_input_dim}. Refitting preprocessor on training CSV."
                )
                preprocessor = FlowPreprocessor()
                X_l, y_l = preprocessor.fit_transform(df_labeled, label_col=args.label_col)
                preprocessor.save(args.preprocessor_path)
        else:
            preprocessor = FlowPreprocessor()
            X_l, y_l = preprocessor.fit_transform(df_labeled, label_col=args.label_col)
            preprocessor.save(args.preprocessor_path)
            print(f"Fitted preprocessor saved to {args.preprocessor_path}")
        if X_l.shape[1] != expected_input_dim:
            raise ValueError(
                f"Training features have {X_l.shape[1]} columns but the frozen encoder expects "
                f"{expected_input_dim}. Re-run SSL pretraining for this dataset first."
            )
    else:
        df_labeled = loader.create_synthetic_data(num_samples=2000, num_features=80)
        preprocessor = FlowPreprocessor()
        X_l, y_l = preprocessor.fit_transform(df_labeled, label_col=args.label_col)
    
    y_l_binary = make_task_labels(args.task, y_l, preprocessor.get_classes())
    
    # 1. Carve out a 15% validation split for early stopping and threshold tuning
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_l, y_l_binary, test_size=0.15, random_state=42, stratify=stratify_labels
    )
    
    # Save validation data for evaluator
    val_save_path = f"{ckpt_base}/{args.task}/val_data.npz"
    os.makedirs(os.path.dirname(val_save_path), exist_ok=True)
    np.savez(val_save_path, val_x=X_val, val_y=y_val)
    print(f"Validation data saved to {val_save_path} ({len(X_val)} samples)")
    
    # 2. Split train set into labeled and unlabeled subsets
    label_ratio = args.label_ratio
    if label_ratio >= 1.0:
        # 100% labeled: use all training data as labeled, duplicate as unlabeled for FixMatch
        X_l = X_train_full
        y_l_binary = y_train_full
        X_u = X_train_full  # unlabeled path still needs data for FixMatch pipeline
        print(f"[INFO] Using 100% labeled data ({len(X_l)} samples). Unlabeled set mirrors labeled for FixMatch compatibility.")
    else:
        unlabeled_fraction = 1.0 - label_ratio
        stratify_train = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_l_sub, X_u_sub, y_l_binary, _ = train_test_split(
            X_train_full, y_train_full, test_size=unlabeled_fraction,
            random_state=42, stratify=stratify_train
        )
        X_l = X_l_sub
        X_u = X_u_sub
    
    if args.max_labeled is not None:
        X_l = X_l[:args.max_labeled]
        y_l_binary = y_l_binary[:args.max_labeled]
    
    print(f"Dataset split summary -> Labeled: {len(X_l)} samples | Unlabeled: {len(X_u)} samples | Validation: {len(X_val)} samples")
    
    # Compute class weights (inverse frequency) to address class imbalance
    unique_classes, class_counts = np.unique(y_l_binary, return_counts=True)
    n_samples = len(y_l_binary)
    n_classes = len(unique_classes)
    class_weights = {}
    for cls_id, count in zip(unique_classes, class_counts):
        class_weights[int(cls_id)] = n_samples / (n_classes * count)
    print(f"Class distribution (labeled): {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
    print(f"Class weights: {class_weights}")
    
    # Auto-enable class balancing for imbalanced datasets
    use_balanced = args.balanced
    if len(class_counts) > 1:
        min_c, max_c = min(class_counts), max(class_counts)
        ratio = max_c / max(1, min_c)
        if ratio > 5.0:
            if not use_balanced:
                print(f"[INFO] Class imbalance ratio is {ratio:.1f}:1 (>5.0). Auto-enabling class-balanced batching for '{args.task}'.")
            use_balanced = True
    
    # Create datasets
    if use_balanced:
        if len(class_counts) < 2:
            raise ValueError("Balanced batching requires both classes to be present.")
        print(f"[INFO] Using class-balanced batching (50/50 per batch)")
        labeled_ds = make_balanced_dataset(X_l, y_l_binary, batch_size=args.batch_size)
        minority_count = int(min(class_counts))
        steps_per_epoch = max(100, min(2 * minority_count // args.batch_size, 2000))
        labeled_ds = labeled_ds.take(steps_per_epoch)
        print(f"[INFO] Steps per epoch (balanced): {steps_per_epoch}")
    else:
        labeled_ds = make_labeled_dataset(X_l, y_l_binary, batch_size=args.batch_size)
    
    max_steps_per_epoch = 2000
    if len(X_l) / args.batch_size > max_steps_per_epoch:
        print(f"[INFO] Labeled dataset is large ({len(X_l)} samples). Capping steps per epoch to {max_steps_per_epoch} for speed.")
        labeled_ds = labeled_ds.take(max_steps_per_epoch)
        
    unlabeled_ds = make_unlabeled_dataset(X_u, batch_size=args.unlabeled_batch_size, for_ssl=False)
    
    # Train via FixMatch with focal loss, class weighting, validation & early stopping
    trainer = FixMatchTrainer(
        encoder=encoder, head=head, gpm=gpm, lr=0.03,
        class_weights=class_weights, focal_gamma=2.0, confidence_threshold=0.90,
        log_dir=f"{log_base}/task_{args.task}",
        ckpt_dir=f"{ckpt_base}/{args.task}",
        clip_norm=1.0,
        max_class_weight=10.0,
        weight_decay=1e-4,
        min_mask_rate_threshold=0.50,
        unfreeze_encoder=args.unfreeze_encoder,
        encoder_lr=args.encoder_lr,
    )
    trainer.train(labeled_ds, unlabeled_ds, task_name=args.task, epochs=args.epochs,
                  warmup_epochs=args.warmup_epochs, val_data=(X_val, y_val), patience=5)
    
    # After training, capture the gradient basis for this task to protect it from future tasks
    if gpm is not None and memory_bank is not None:
        print(f"Capturing GPM basis for {args.task}...")
        try:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # Combined model wrapper for GPM; exposes encoder variables when unfrozen.
            class TaskModel(tf.keras.Model):
                def __init__(self, enc, hd, include_encoder=False):
                    super().__init__()
                    self.enc = enc
                    self.hd = hd
                    self.include_encoder = include_encoder
                @property
                def trainable_variables(self):
                    if self.include_encoder and self.enc.trainable:
                        return list(self.enc.trainable_variables) + list(self.hd.trainable_variables)
                    return list(self.hd.trainable_variables)
                def call(self, x, training=False):
                    enc_training = training and self.include_encoder
                    return self.hd(self.enc(x, training=enc_training), training=training)

            combined_model = TaskModel(encoder, head, include_encoder=args.unfreeze_encoder)
            gpm.capture_gradient_basis(
                combined_model,
                labeled_ds,
                loss_fn,
                max_batches=args.max_gpm_batches,
            )
            memory_bank.save()
        except Exception as e:
            print(f"[ERROR] Failed to capture GPM basis for task {args.task}: {e}")

    # Save encoder snapshot for transfer matrix computation
    if args.unfreeze_encoder:
        snapshot_dir = f"{ckpt_base}/snapshots/after_{args.task}/encoder"
        os.makedirs(snapshot_dir, exist_ok=True)
        encoder_ckpt = tf.train.Checkpoint(encoder=encoder)
        encoder_mgr = tf.train.CheckpointManager(encoder_ckpt, snapshot_dir, max_to_keep=1)
        encoder_mgr.save()
        print(f"Encoder snapshot saved to {snapshot_dir}/")
    
    print(f"Pipeline for {args.task} completed successfully.")

if __name__ == "__main__":
    main()
