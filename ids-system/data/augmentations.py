import tensorflow as tf
from typing import Optional

def apply_noise(x: tf.Tensor, stddev: float = 0.05, continuous_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Adds Gaussian noise to flow features (optionally restricted to continuous features)."""
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=tf.float32)
    if continuous_mask is not None:
        noise = noise * tf.cast(continuous_mask, tf.float32)
    return x + noise

def apply_feature_swap(x: tf.Tensor, swap_prob: float = 0.15) -> tf.Tensor:
    """
    Randomly swaps feature values across rows in a batch (marginal distribution sampling).
    If a single sample 1D tensor is passed, falls back to small Gaussian perturbations.
    """
    shape = tf.shape(x)
    if tf.rank(x) > 1:
        batch_size = shape[0]
        # Random row indices for feature values
        random_indices = tf.random.uniform(shape, minval=0, maxval=batch_size, dtype=tf.int32)
        # Gather swapped feature values along axis 0
        swapped = tf.gather(x, random_indices, batch_dims=0)
        swap_mask = tf.cast(tf.random.uniform(shape) < swap_prob, dtype=tf.float32)
        return x * (1.0 - swap_mask) + swapped * swap_mask
    else:
        mask = tf.cast(tf.random.uniform(shape) > swap_prob, dtype=tf.float32)
        noise = tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=tf.float32)
        return x * mask + noise * (1.0 - mask)

def augment_weak(x: tf.Tensor, continuous_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Weak augmentation for tabular flow data.
    Used for generating pseudo-labels in FixMatch.
    """
    return apply_noise(x, stddev=0.02, continuous_mask=continuous_mask)

def augment_strong(x: tf.Tensor, continuous_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Strong augmentation for tabular flow data.
    Combines Gaussian noise with feature swap sampling.
    """
    x = apply_noise(x, stddev=0.08, continuous_mask=continuous_mask)
    x = apply_feature_swap(x, swap_prob=0.15)
    return x
