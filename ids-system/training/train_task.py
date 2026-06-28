import os
import sys
import argparse
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from data.tf_dataset import make_labeled_dataset, make_unlabeled_dataset, make_balanced_dataset
from classifiers.base_head import build_classifier_head
from classifiers.dos_head import build_dos_head
from classifiers.scan_head import build_scan_head
from classifiers.fixmatch_trainer import FixMatchTrainer
from gpm.gpm import GradientProjectionMemory
from gpm.memory_bank import MemoryBank

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


def load_frozen_encoder(path: str = "./checkpoints/encoder_frozen.keras") -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frozen encoder not found at {path}. Run train_ssl.py first.")
        
    import zipfile
    if zipfile.is_zipfile(path):
        print(f"Encoder Keras 3 zip format detected. Loading weights manually.")
        from encoder.flow_encoder import build_flow_encoder
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


def build_task_head(task: str, embed_dim: int) -> tf.keras.Model:
    if task == "intrusion":
        return build_classifier_head(embed_dim=embed_dim, num_classes=2, name="intrusion_head")
    if task == "dos":
        return build_dos_head(embed_dim=embed_dim)
    if task == "port_scan":
        return build_scan_head(embed_dim=embed_dim)
    raise ValueError(f"Unsupported task: {task}")


def make_task_labels(task: str, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if task == "intrusion":
        if len(classes) != 2:
            raise ValueError(f"Intrusion task expects a binary label column, got classes: {classes.tolist()}")
        return labels.astype(np.int32)

    target_names = {
        "dos": ["dos"],
        "port_scan": ["portscan", "port scan", "port_scan", "probe", "reconnaissance"],
    }[task]
    normalized_classes = [str(c).strip().lower() for c in classes]
    target_indices = [i for i, c in enumerate(normalized_classes) if c in target_names]
    if not target_indices:
        raise ValueError(
            f"Could not find task label for '{task}' in classes: {classes.tolist()}. "
            "Use --task intrusion with UNSW's binary 'label' column."
        )
    return np.isin(labels, target_indices).astype(np.int32)

def main():
    parser = argparse.ArgumentParser(description="Train Task-Specific Classifier")
    parser.add_argument("--task", type=str, required=True, choices=["intrusion", "dos", "port_scan"], help="Task to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Labeled batch size")
    parser.add_argument("--unlabeled_batch_size", type=int, default=128, help="Unlabeled batch size")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw"],
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
    args = parser.parse_args()

    print(f"Initializing Continual Learning for Task: {args.task}")

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

    # Load frozen encoder
    encoder = load_frozen_encoder(f"{ckpt_base}/encoder_frozen.keras")
    
    # Initialize GPM only for continual task heads. The generic intrusion task is a single binary head.
    memory_bank = None
    gpm = None
    if args.task != "intrusion":
        memory_bank = MemoryBank(save_dir=f"{ckpt_base}/gpm")
        memory_bank.load()
        gpm = GradientProjectionMemory(threshold=0.97, memory_bank=memory_bank)
    
    # Initialize Head
    head = build_task_head(args.task, embed_dim=encoder.output_shape[-1])
        
    loader = FlowDatasetLoader(data_path=args.data_path or ".")
    if args.dataset:
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset is used")
        df_labeled = loader.load_dataset(
            args.dataset, split="train", label_col=args.label_col
        )
        expected_input_dim = encoder.input_shape[-1]
        if os.path.exists(args.preprocessor_path):
            preprocessor = FlowPreprocessor.load(args.preprocessor_path)
            try:
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
        X_u = X_l
    elif args.train_csv:
        df_labeled = loader.load_csv(args.train_csv, label_col=args.label_col)
        expected_input_dim = encoder.input_shape[-1]
        if os.path.exists(args.preprocessor_path):
            preprocessor = FlowPreprocessor.load(args.preprocessor_path)
            try:
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
        X_u = X_l
    else:
        df_labeled = loader.create_synthetic_data(num_samples=500, num_features=80)
        df_unlabeled = loader.create_synthetic_data(num_samples=20000, num_features=80)
        preprocessor = FlowPreprocessor()
        X_l, y_l = preprocessor.fit_transform(df_labeled, label_col=args.label_col)
        X_u, _ = preprocessor.transform(df_unlabeled, label_col=args.label_col)
    
    y_l_binary = make_task_labels(args.task, y_l, preprocessor.get_classes())
    if args.max_labeled is not None:
        X_l = X_l[:args.max_labeled]
        y_l_binary = y_l_binary[:args.max_labeled]
    
    # Compute class weights (inverse frequency) to address class imbalance
    unique_classes, class_counts = np.unique(y_l_binary, return_counts=True)
    n_samples = len(y_l_binary)
    n_classes = len(unique_classes)
    class_weights = {}
    for cls_id, count in zip(unique_classes, class_counts):
        class_weights[int(cls_id)] = n_samples / (n_classes * count)
    print(f"Class distribution: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
    print(f"Class weights: {class_weights}")
    
    # Log if class imbalance is severe (ratio > 50:1)
    if len(class_counts) > 1:
        min_c, max_c = min(class_counts), max(class_counts)
        ratio = max_c / max(1, min_c)
        if ratio > 50.0:
            print(f"[WARN] Severe class imbalance detected (ratio {ratio:.1f}:1). FixMatchTrainer will cap class weights and clip gradients.")
    
    # Create datasets
    if args.balanced:
        print(f"[INFO] Using class-balanced batching (50/50 per batch)")
        labeled_ds = make_balanced_dataset(X_l, y_l_binary, batch_size=args.batch_size)
        # For balanced datasets, set a fixed number of steps per epoch
        # since the dataset is infinite (uses .repeat() internally)
        minority_count = int(min(class_counts))
        steps_per_epoch = min(2 * minority_count // args.batch_size, 2000)
        labeled_ds = labeled_ds.take(steps_per_epoch)
        print(f"[INFO] Steps per epoch (balanced): {steps_per_epoch}")
    else:
        labeled_ds = make_labeled_dataset(X_l, y_l_binary, batch_size=args.batch_size)
    
    # Cap the steps per epoch if the dataset is large, to keep training times reasonable.
    max_steps_per_epoch = 2000
    if len(X_l) / args.batch_size > max_steps_per_epoch:
        print(f"[INFO] Labeled dataset is large ({len(X_l)} samples). Capping steps per epoch to {max_steps_per_epoch} for speed.")
        labeled_ds = labeled_ds.take(max_steps_per_epoch)
        
    unlabeled_ds = make_unlabeled_dataset(X_u, batch_size=args.unlabeled_batch_size, for_ssl=False)
    
    # Train via FixMatch with focal loss and class weighting
    trainer = FixMatchTrainer(
        encoder=encoder, head=head, gpm=gpm, lr=0.03,
        class_weights=class_weights, focal_gamma=2.0, confidence_threshold=0.90,
        log_dir=f"{log_base}/task_{args.task}",
        ckpt_dir=f"{ckpt_base}/{args.task}",
        clip_norm=1.0,
        max_class_weight=10.0
    )
    trainer.train(labeled_ds, unlabeled_ds, task_name=args.task, epochs=args.epochs,
                  warmup_epochs=args.warmup_epochs)
    
    # After training, capture the gradients for this task to protect it in the future
    if gpm is not None and memory_bank is not None:
        print(f"Capturing GPM basis for {args.task}...")
        try:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
            # We use a combined model just to pass to GPM which expects a single callable.
            class TaskModel(tf.keras.Model):
                def __init__(self, enc, hd):
                    super().__init__()
                    self.enc = enc
                    self.hd = hd
                @property
                def trainable_variables(self):
                    return self.hd.trainable_variables
                def call(self, x, training=False):
                    return self.hd(self.enc(x, training=False), training=training)
                
            combined_model = TaskModel(encoder, head)
            gpm.capture_gradient_basis(
                combined_model,
                labeled_ds,
                loss_fn,
                max_batches=args.max_gpm_batches,
            )
            memory_bank.save()
        except Exception as e:
            print(f"[ERROR] Failed to capture GPM basis for task {args.task}: {e}")
    
    print(f"Pipeline for {args.task} completed successfully.")

if __name__ == "__main__":
    main()
