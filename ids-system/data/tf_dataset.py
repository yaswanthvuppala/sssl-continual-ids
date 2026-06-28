import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
from data.augmentations import augment_weak, augment_strong

def make_unlabeled_dataset(
    features: np.ndarray,
    batch_size: int = 512,
    shuffle_buffer: int = 50000,
    for_ssl: bool = True
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset for unlabeled data.
    If for_ssl=True, yields two strongly augmented views of the same sample (SimCLR).
    If for_ssl=False, yields a weakly and strongly augmented view (FixMatch unlabeled stream).
    """
    ds = tf.data.Dataset.from_tensor_slices(features)
    
    def simclr_views(x):
        return augment_strong(x), augment_strong(x)
        
    def fixmatch_views(x):
        return augment_weak(x), augment_strong(x)
        
    map_func = simclr_views if for_ssl else fixmatch_views
    
    ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(features)), reshuffle_each_iteration=True)
    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def make_labeled_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle_buffer: int = 5000
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset for labeled data.
    Applies weak augmentation.
    """
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    
    def process(x, y):
        return augment_weak(x), y
        
    ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(features)), reshuffle_each_iteration=True)
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def make_balanced_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """
    Creates a class-balanced tf.data.Dataset for labeled data.
    Each batch contains ~50% class-0 and ~50% class-1 samples.
    The minority class is repeated (oversampled) to match the majority class.
    """
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]

    ds_0 = tf.data.Dataset.from_tensor_slices(
        (features[idx_0], labels[idx_0])
    ).shuffle(len(idx_0), reshuffle_each_iteration=True).repeat()

    ds_1 = tf.data.Dataset.from_tensor_slices(
        (features[idx_1], labels[idx_1])
    ).shuffle(len(idx_1), reshuffle_each_iteration=True).repeat()

    half = batch_size // 2

    def merge_fn(batch_0, batch_1):
        x = tf.concat([batch_0[0], batch_1[0]], axis=0)
        y = tf.concat([batch_0[1], batch_1[1]], axis=0)
        # Shuffle within batch to avoid class ordering bias
        indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        return tf.gather(x, indices), tf.gather(y, indices)

    ds_0 = ds_0.map(lambda x, y: (augment_weak(x), y),
                     num_parallel_calls=tf.data.AUTOTUNE).batch(half)
    ds_1 = ds_1.map(lambda x, y: (augment_weak(x), y),
                     num_parallel_calls=tf.data.AUTOTUNE).batch(half)

    ds = tf.data.Dataset.zip((ds_0, ds_1)).map(
        merge_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
