import tensorflow as tf
import numpy as np
from typing import Optional

class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector operating on SSL embedding space.
    Trained on normal traffic embeddings; high reconstruction error indicates anomaly.
    """
    def __init__(self, embed_dim: int = 256, latent_dim: int = 64):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.model = self._build_autoencoder()
        self.threshold = 0.5  # Default, should be calibrated
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    def _build_autoencoder(self) -> tf.keras.Model:
        """Builds a symmetric autoencoder for reconstruction-based anomaly detection."""
        inputs = tf.keras.Input(shape=(self.embed_dim,), name="ae_input")
        
        # Encoder
        x = tf.keras.layers.Dense(128, activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.latent_dim, activation="relu", name="ae_latent")(x)
        
        # Decoder
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(self.embed_dim, activation=None, name="ae_reconstruction")(x)
        
        model = tf.keras.Model(inputs, outputs, name="anomaly_autoencoder")
        return model
    
    @tf.function
    def train_step(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Single training step: minimize reconstruction error on normal embeddings."""
        with tf.GradientTape() as tape:
            reconstructed = self.model(embeddings, training=True)
            loss = tf.reduce_mean(tf.square(embeddings - reconstructed))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def train(self, normal_embeddings: np.ndarray, epochs: int = 20, batch_size: int = 256):
        """Train the autoencoder on embeddings from normal (benign) traffic."""
        print(f"Training anomaly autoencoder on {len(normal_embeddings)} normal embeddings...")
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(normal_embeddings, dtype=tf.float32)
        ).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += float(loss)
                steps += 1
            avg_loss = total_loss / max(1, steps)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  AE Epoch {epoch+1}/{epochs} — Reconstruction Loss: {avg_loss:.6f}")
        
        # Auto-calibrate threshold from training data
        self._calibrate_threshold(normal_embeddings)
        print(f"Anomaly autoencoder training complete. Threshold: {self.threshold:.6f}")
        
    def _calibrate_threshold(self, normal_embeddings: np.ndarray, percentile: float = 95.0):
        """Set threshold as the Nth percentile of reconstruction errors on normal data."""
        errors = self.compute_reconstruction_errors(normal_embeddings)
        self.threshold = float(np.percentile(errors, percentile))
        
    def compute_reconstruction_errors(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE)."""
        emb_tf = tf.constant(embeddings, dtype=tf.float32)
        reconstructed = self.model(emb_tf, training=False)
        errors = tf.reduce_mean(tf.square(emb_tf - reconstructed), axis=1)
        return errors.numpy()
    
    def score(self, embedding: np.ndarray) -> float:
        """Returns a normalized anomaly score for a single embedding (or batch)."""
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        errors = self.compute_reconstruction_errors(embedding)
        # Normalize: score = error / threshold, capped at 1.0
        scores = np.clip(errors / max(self.threshold, 1e-8), 0.0, 1.0)
        return float(scores[0]) if len(scores) == 1 else scores
    
    def is_anomaly(self, embedding: np.ndarray) -> bool:
        """Returns True if the embedding is anomalous."""
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        errors = self.compute_reconstruction_errors(embedding)
        return bool(errors[0] > self.threshold)
    
    def save(self, path: str = "./checkpoints/anomaly_ae.keras"):
        """Save the trained autoencoder."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        # Save threshold alongside
        np.save(path.replace(".keras", "_threshold.npy"), np.array([self.threshold]))
        print(f"Anomaly autoencoder saved to {path}")
        
    def load(self, path: str = "./checkpoints/anomaly_ae.keras"):
        """Load a previously trained autoencoder."""
        self.model = tf.keras.models.load_model(path)
        threshold_path = path.replace(".keras", "_threshold.npy")
        import os
        if os.path.exists(threshold_path):
            self.threshold = float(np.load(threshold_path)[0])
        print(f"Anomaly autoencoder loaded from {path} (threshold={self.threshold:.6f})")
