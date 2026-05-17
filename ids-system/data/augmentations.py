import tensorflow as tf

def apply_noise(x: tf.Tensor, stddev: float = 0.05) -> tf.Tensor:
    """Adds Gaussian noise to flow features."""
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=tf.float32)
    return x + noise

def apply_feature_masking(x: tf.Tensor, mask_prob: float = 0.1) -> tf.Tensor:
    """Randomly masks out features (sets to 0) with a given probability."""
    mask = tf.cast(tf.random.uniform(shape=tf.shape(x)) > mask_prob, dtype=tf.float32)
    return x * mask

def augment_weak(x: tf.Tensor) -> tf.Tensor:
    """
    Weak augmentation for tabular data.
    Used for generating pseudo-labels in FixMatch.
    """
    return apply_noise(x, stddev=0.02)

def augment_strong(x: tf.Tensor) -> tf.Tensor:
    """
    Strong augmentation for tabular data.
    Used for consistency regularization in FixMatch and views in SimCLR.
    """
    x = apply_noise(x, stddev=0.1)
    x = apply_feature_masking(x, mask_prob=0.2)
    return x
