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
