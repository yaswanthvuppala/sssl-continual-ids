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
    
    def save(self, path: str = None):
        """Save the trained autoencoder."""
        path = path or "./checkpoints/anomaly_ae.keras"
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        # Save threshold alongside
        np.save(path.replace(".keras", "_threshold.npy"), np.array([self.threshold]))
        print(f"Anomaly autoencoder saved to {path}")
        
    def load(self, path: str = None):
        """Load a previously trained autoencoder."""
        path = path or "./checkpoints/anomaly_ae.keras"
        import os
        import zipfile
        import tempfile
        import shutil
        import h5py
        
        if os.path.exists(path) and zipfile.is_zipfile(path):
            print(f"Anomaly autoencoder Keras 3 zip format detected. Loading weights manually.")
            try:
                temp_dir = tempfile.mkdtemp(dir=".")
                try:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        weights_path = zip_ref.extract('model.weights.h5', path=temp_dir)
                        with h5py.File(weights_path, 'r') as f:
                            # /layers/dense/vars/0 has shape (embed_dim, 128)
                            embed_dim = f['layers/dense/vars/0'].shape[0]
                            
                            # Find ae_latent layer to get latent_dim
                            latent_dim = self.latent_dim
                            layers_root = f['layers']
                            for grp_name in layers_root.keys():
                                vars_path = f"layers/{grp_name}/vars"
                                if vars_path in f:
                                    name_attr = f[vars_path].attrs.get('name')
                                    if name_attr:
                                        if isinstance(name_attr, bytes):
                                            name_attr = name_attr.decode('utf-8')
                                        if name_attr == 'ae_latent':
                                            latent_dim = f[f"{vars_path}/0"].shape[1]
                                            break
                    
                    print(f"Detected autoencoder dimensions: embed_dim={embed_dim}, latent_dim={latent_dim}")
                    self.embed_dim = embed_dim
                    self.latent_dim = latent_dim
                    self.model = self._build_autoencoder()
                    
                    # Manual weight loader
                    with zipfile.ZipFile(path, 'r') as zip_ref:
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
                            
                            for layer in self.model.layers:
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
                    print("  Successfully loaded weights manually.")
                finally:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"  [ERROR] Keras 3 manual load failed: {e}. Falling back to standard load.")
                self.model = tf.keras.models.load_model(path)
        else:
            self.model = tf.keras.models.load_model(path)
            
        threshold_path = path.replace(".keras", "_threshold.npy")
        if os.path.exists(threshold_path):
            self.threshold = float(np.load(threshold_path)[0])
        print(f"Anomaly autoencoder loaded from {path} (threshold={self.threshold:.6f})")
