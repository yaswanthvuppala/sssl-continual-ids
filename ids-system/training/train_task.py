import os
import sys
import argparse
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from data.tf_dataset import make_labeled_dataset, make_unlabeled_dataset
from classifiers.base_head import build_classifier_head
from classifiers.dos_head import build_dos_head
from classifiers.scan_head import build_scan_head
from classifiers.fixmatch_trainer import FixMatchTrainer
from gpm.gpm import GradientProjectionMemory
from gpm.memory_bank import MemoryBank

def load_frozen_encoder(path: str = "./checkpoints/encoder_frozen.keras") -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frozen encoder not found at {path}. Run train_ssl.py first.")
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
        "port_scan": ["portscan", "port scan", "reconnaissance"],
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
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the training CSV")
    parser.add_argument("--preprocessor_path", type=str, default="./checkpoints/preprocessor.pkl",
                        help="Path to the fitted preprocessor")
    parser.add_argument("--max_labeled", type=int, default=None,
                        help="Optional cap on labeled training samples")
    args = parser.parse_args()

    print(f"Initializing Continual Learning for Task: {args.task}")
    
    # Load frozen encoder
    encoder = load_frozen_encoder()
    
    # Initialize GPM only for continual task heads. The generic intrusion task is a single binary head.
    memory_bank = None
    gpm = None
    if args.task != "intrusion":
        memory_bank = MemoryBank(save_dir="./checkpoints/gpm")
        memory_bank.load()
        gpm = GradientProjectionMemory(threshold=0.97, memory_bank=memory_bank)
    
    # Initialize Head
    head = build_task_head(args.task, embed_dim=encoder.output_shape[-1])
        
    loader = FlowDatasetLoader(data_path=".")
    if args.train_csv:
        df_labeled = loader.load_csv(args.train_csv, label_col=args.label_col)
        expected_input_dim = encoder.input_shape[-1]
        if os.path.exists(args.preprocessor_path):
            preprocessor = FlowPreprocessor.load(args.preprocessor_path)
            X_l, y_l = preprocessor.transform(df_labeled, label_col=args.label_col)
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
    
    # Create datasets
    labeled_ds = make_labeled_dataset(X_l, y_l_binary, batch_size=args.batch_size)
    unlabeled_ds = make_unlabeled_dataset(X_u, batch_size=args.unlabeled_batch_size, for_ssl=False)
    
    # Train via FixMatch
    trainer = FixMatchTrainer(encoder=encoder, head=head, gpm=gpm, lr=0.03)
    trainer.train(labeled_ds, unlabeled_ds, task_name=args.task, epochs=args.epochs)
    
    # After training, capture the gradients for this task to protect it in the future
    if gpm is not None and memory_bank is not None:
        print(f"Capturing GPM basis for {args.task}...")
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
        gpm.capture_gradient_basis(combined_model, labeled_ds, loss_fn)
        memory_bank.save()
    
    print(f"Pipeline for {args.task} completed successfully.")

if __name__ == "__main__":
    main()
