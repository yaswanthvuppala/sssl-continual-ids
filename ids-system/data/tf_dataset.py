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


def make_eval_dataset(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    batch_size: int = 256
) -> tf.data.Dataset:
    """
    Creates an evaluation tf.data.Dataset.
    Applies zero augmentation, no shuffling, and drop_remainder=False.
    """
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(features)
        
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_balanced_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """
    Creates a class-balanced tf.data.Dataset for arbitrary multi-class labels.
    Subsamples/oversamples each class to have equal representation per batch.
    """
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    per_class_batch = max(1, batch_size // num_classes)
    
    class_datasets = []
    for c in unique_classes:
        idx = np.where(labels == c)[0]
        c_ds = (
            tf.data.Dataset.from_tensor_slices((features[idx], labels[idx]))
            .shuffle(buffer_size=max(1, len(idx)), reshuffle_each_iteration=True)
            .repeat()
            .map(lambda x, y: (augment_weak(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(per_class_batch)
        )
        class_datasets.append(c_ds)
        
    def merge_fn(*batches):
        x = tf.concat([b[0] for b in batches], axis=0)
        y = tf.concat([b[1] for b in batches], axis=0)
        indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        return tf.gather(x, indices), tf.gather(y, indices)

    ds = tf.data.Dataset.zip(tuple(class_datasets)).map(
        merge_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.prefetch(tf.data.AUTOTUNE)
