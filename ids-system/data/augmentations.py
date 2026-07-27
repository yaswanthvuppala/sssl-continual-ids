import tensorflow as tf

def apply_noise(x: tf.Tensor, stddev: float = 0.05, continuous_mask: tf.Tensor = None) -> tf.Tensor:
    """
    Adds Gaussian noise to tabular flow features.
    If continuous_mask is provided or detected, noise is only added to continuous columns
    to preserve binary/one-hot categorical structures.
    """
    x_float = tf.cast(x, dtype=tf.float32)
    noise = tf.random.normal(shape=tf.shape(x_float), mean=0.0, stddev=stddev, dtype=tf.float32)

    if continuous_mask is not None:
        mask = tf.cast(continuous_mask, dtype=tf.float32)
        noise = noise * mask
    else:
        # Detect binary/one-hot columns (features where all values in batch are strictly 0.0 or 1.0)
        is_binary = tf.reduce_all(
            tf.math.logical_or(tf.equal(x_float, 0.0), tf.equal(x_float, 1.0)),
            axis=0, keepdims=True
        )
        is_continuous = tf.cast(tf.math.logical_not(is_binary), dtype=tf.float32)
        noise = noise * is_continuous

    return x_float + noise

def apply_feature_masking(x: tf.Tensor, mask_prob: float = 0.1) -> tf.Tensor:
    """Randomly masks out features (sets to 0) with a given probability."""
    x_float = tf.cast(x, dtype=tf.float32)
    mask = tf.cast(tf.random.uniform(shape=tf.shape(x_float)) > mask_prob, dtype=tf.float32)
    return x_float * mask

def augment_weak(x: tf.Tensor, stddev: float = 0.02) -> tf.Tensor:
    """
    Weak augmentation for tabular data.
    Used for generating pseudo-labels in FixMatch.
    """
    return apply_noise(x, stddev=stddev)

def augment_strong(x: tf.Tensor, stddev: float = 0.1, mask_prob: float = 0.2) -> tf.Tensor:
    """
    Strong augmentation for tabular data.
    Used for consistency regularization in FixMatch and views in SimCLR.
    """
    x_noisy = apply_noise(x, stddev=stddev)
    x_masked = apply_feature_masking(x_noisy, mask_prob=mask_prob)
    return x_masked

