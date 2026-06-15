"""
benchmark.py — End-to-end benchmark: SSL → Task1 → Task2 → Evaluate → Report.

Usage:
    python training/benchmark.py
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from data.tf_dataset import make_labeled_dataset, make_unlabeled_dataset
from encoder.flow_encoder import build_flow_encoder
from encoder.projection_head import build_projection_head
from encoder.losses import nt_xent_loss
from classifiers.dos_head import build_dos_head
from classifiers.scan_head import build_scan_head
from classifiers.fixmatch_trainer import FixMatchTrainer
from gpm.gpm import GradientProjectionMemory
from gpm.memory_bank import MemoryBank
from anomaly.autoencoder_detector import AutoencoderDetector
from training.evaluate import evaluate_head


def main():
    print("=" * 70)
    print("  SSSL-IDS END-TO-END BENCHMARK")
    print("=" * 70)
    t0 = time.time()

    NUM_FEATURES = 80
    EMBED_DIM = 256
    SSL_EPOCHS = 3
    TASK_EPOCHS = 3

    # ── Data Generation ──
    print("\n[1/6] Generating synthetic data...")
    loader = FlowDatasetLoader(data_path=".")
    df_pretrain = loader.create_synthetic_data(num_samples=20000, num_features=NUM_FEATURES)
    df_labeled = loader.create_synthetic_data(num_samples=500, num_features=NUM_FEATURES)
    df_unlabeled = loader.create_synthetic_data(num_samples=10000, num_features=NUM_FEATURES)
    df_eval = loader.create_synthetic_data(num_samples=2000, num_features=NUM_FEATURES)

    preprocessor = FlowPreprocessor()
    X_pretrain, _ = preprocessor.fit_transform(df_pretrain)
    X_labeled, y_labeled = preprocessor.transform(df_labeled)
    X_unlabeled, _ = preprocessor.transform(df_unlabeled)
    X_eval, y_eval = preprocessor.transform(df_eval)

    # ── Stage 1: SSL Pretraining ──
    print(f"\n[2/6] SSL Pretraining ({SSL_EPOCHS} epochs)...")
    encoder = build_flow_encoder(input_dim=NUM_FEATURES, embed_dim=EMBED_DIM)
    projector = build_projection_head(in_dim=EMBED_DIM, out_dim=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    trainable_vars = encoder.trainable_variables + projector.trainable_variables

    ssl_ds = make_unlabeled_dataset(X_pretrain, batch_size=256, for_ssl=True)
    for epoch in range(SSL_EPOCHS):
        losses = []
        for x1, x2 in ssl_ds:
            with tf.GradientTape() as tape:
                z1 = projector(encoder(x1, training=True), training=True)
                z2 = projector(encoder(x2, training=True), training=True)
                loss = nt_xent_loss(z1, z2, temperature=0.1)
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            losses.append(float(loss))
        print(f"  SSL Epoch {epoch+1}/{SSL_EPOCHS} — loss: {np.mean(losses):.4f}")

    encoder.trainable = False
    ckpt_base = "./checkpoints/benchmark"
    os.makedirs(ckpt_base, exist_ok=True)
    encoder.save(f"{ckpt_base}/encoder_frozen.keras")
    print("  Encoder frozen and saved.")

    # ── Anomaly Detector Training ──
    print("\n[3/6] Training anomaly autoencoder...")
    # Use encoder embeddings of "benign" class as normal
    benign_mask = (y_eval == 0)
    if benign_mask.sum() > 0:
        benign_embeddings = encoder(tf.constant(X_eval[benign_mask], dtype=tf.float32), training=False).numpy()
    else:
        benign_embeddings = encoder(tf.constant(X_eval[:500], dtype=tf.float32), training=False).numpy()

    ae_det = AutoencoderDetector(embed_dim=EMBED_DIM)
    ae_det.train(benign_embeddings, epochs=10, batch_size=128)
    ae_det.save()

    # ── Stage 2: Task Learning with GPM ──
    memory_bank = MemoryBank(save_dir=f"{ckpt_base}/gpm")
    gpm = GradientProjectionMemory(threshold=0.97, memory_bank=memory_bank)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    results = {}

    tasks = [
        ("dos", build_dos_head, 1),
        ("port_scan", build_scan_head, 2),
    ]

    for task_name, build_fn, target_class in tasks:
        print(f"\n[4/6] Training task: {task_name} ({TASK_EPOCHS} epochs)...")
        head = build_fn(embed_dim=EMBED_DIM)
        y_binary = (y_labeled == target_class).astype(np.int32)

        labeled_ds = make_labeled_dataset(X_labeled, y_binary, batch_size=32)
        unlabeled_ds = make_unlabeled_dataset(X_unlabeled, batch_size=128, for_ssl=False)

        trainer = FixMatchTrainer(encoder=encoder, head=head, gpm=gpm, lr=0.03)
        trainer.train(labeled_ds, unlabeled_ds, task_name=task_name, epochs=TASK_EPOCHS)

        # Capture GPM basis
        class _CombinedModel(tf.keras.Model):
            def __init__(self, enc, hd):
                super().__init__()
                self.enc = enc
                self.hd = hd
            def call(self, x, training=False):
                return self.hd(self.enc(x, training=False), training=training)

        combined = _CombinedModel(encoder, head)
        combined._trainable_variables = head.trainable_variables  # only head vars
        gpm.capture_gradient_basis(combined, labeled_ds, loss_fn)

        # Evaluate on all previous tasks
        y_eval_bin = (y_eval == target_class).astype(np.int32)
        metrics = evaluate_head(encoder, head, X_eval, y_eval_bin, task_name)
        results[task_name] = [metrics["f1"]]

    memory_bank.save()

    # ── Summary ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    for t, m in results.items():
        print(f"  {t:>15}: F1 = {m[0]:.4f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
