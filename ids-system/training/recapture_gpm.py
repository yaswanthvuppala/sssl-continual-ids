"""
recapture_gpm.py - Recompute a GPM basis from an existing trained task checkpoint.

Use this when task training succeeded but GPM capture failed or was skipped. It
does not retrain the head; it reloads the saved checkpoint, captures gradients
from the training split, updates the memory bank, and exits.
"""
import argparse
import os
import sys

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from data.tf_dataset import make_labeled_dataset
from gpm.gpm import GradientProjectionMemory
from gpm.memory_bank import MemoryBank
from training.train_task import build_task_head, load_frozen_encoder, make_task_labels


def load_training_features(args, ckpt_base):
    if args.preprocessor_path is None:
        args.preprocessor_path = f"{ckpt_base}/preprocessor_{args.task}.pkl"

    if not os.path.exists(args.preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found at {args.preprocessor_path}. "
            "Run task training first."
        )

    loader = FlowDatasetLoader(data_path=args.data_path or ".")
    preprocessor = FlowPreprocessor.load(args.preprocessor_path)

    if args.dataset:
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset is used")
        df_labeled = loader.load_dataset(
            args.dataset, split="train", label_col=args.label_col
        )
    elif args.train_csv:
        df_labeled = loader.load_csv(args.train_csv, label_col=args.label_col)
    else:
        raise ValueError("Provide either --dataset/--data_path or --train_csv")

    X_l, y_l = preprocessor.transform(df_labeled, label_col=args.label_col)
    y_l_binary = make_task_labels(args.task, y_l, preprocessor.get_classes())

    if args.max_labeled is not None:
        X_l = X_l[:args.max_labeled]
        y_l_binary = y_l_binary[:args.max_labeled]

    return X_l, y_l_binary


def main():
    parser = argparse.ArgumentParser(description="Recapture GPM basis from a saved task checkpoint")
    parser.add_argument("--task", type=str, required=True, choices=["dos", "port_scan"],
                        help="Continual task to recapture")
    parser.add_argument("--batch_size", type=int, default=32, help="Labeled batch size")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw"],
                        default=None, help="Load a supported raw dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset directory or raw data file")
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the training CSV")
    parser.add_argument("--preprocessor_path", type=str, default=None,
                        help="Path to the fitted preprocessor")
    parser.add_argument("--max_labeled", type=int, default=None,
                        help="Optional cap on labeled training samples")
    parser.add_argument("--max_gpm_batches", type=int, default=512,
                        help="Maximum valid labeled batches to use for GPM SVD capture; use 0 to scan all batches")
    parser.add_argument("--replace_index", type=int, default=None,
                        help="Optional zero-based memory-bank index to replace instead of appending")
    parser.add_argument("--dataset_name", type=str, default="default",
                        help="Dataset identifier for scoping checkpoint paths")
    args = parser.parse_args()

    ckpt_base = f"./checkpoints/{args.dataset_name}"
    encoder = load_frozen_encoder(f"{ckpt_base}/encoder_frozen.keras")
    head = build_task_head(args.task, embed_dim=encoder.output_shape[-1])

    ckpt = tf.train.latest_checkpoint(f"{ckpt_base}/{args.task}")
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint found for {args.task} under {ckpt_base}/{args.task}")
    tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()
    print(f"Loaded task checkpoint: {ckpt}")

    X_l, y_l_binary = load_training_features(args, ckpt_base)
    if X_l.shape[1] != encoder.input_shape[-1]:
        raise ValueError(
            f"Training features have {X_l.shape[1]} columns but encoder expects "
            f"{encoder.input_shape[-1]}."
        )

    labeled_ds = make_labeled_dataset(X_l, y_l_binary, batch_size=args.batch_size)
    memory_bank = MemoryBank(save_dir=f"{ckpt_base}/gpm")
    memory_bank.load()
    before_count = len(memory_bank.bases)

    gpm = GradientProjectionMemory(threshold=0.97, memory_bank=memory_bank)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

    gpm.capture_gradient_basis(
        TaskModel(encoder, head),
        labeled_ds,
        loss_fn,
        max_batches=args.max_gpm_batches,
    )

    if args.replace_index is not None and len(memory_bank.bases) > before_count:
        if args.replace_index < 0 or args.replace_index >= before_count:
            raise IndexError(
                f"--replace_index must be between 0 and {before_count - 1}; "
                f"got {args.replace_index}"
            )
        new_basis = memory_bank.bases.pop()
        memory_bank.bases[args.replace_index] = new_basis
        print(f"Replaced memory-bank basis at index {args.replace_index}.")

    memory_bank.save()
    print(f"GPM recapture complete: {before_count} -> {len(memory_bank.bases)} task bases.")


if __name__ == "__main__":
    main()
