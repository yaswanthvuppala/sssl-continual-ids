import tensorflow as tf

def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR.
    z1: Projections of view 1 (Batch_Size, Proj_Dim)
    z2: Projections of view 2 (Batch_Size, Proj_Dim)
    """
    # Normalize projections
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    
    batch_size = tf.shape(z1)[0]
    
    # Combine views: [z1, z2]
    # Shape: (2*Batch_Size, Proj_Dim)
    z = tf.concat([z1, z2], axis=0)
    
    # Compute cosine similarity matrix
    # Shape: (2*Batch_Size, 2*Batch_Size)
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature
    
    # Mask out self-similarity (diagonal)
    mask = tf.eye(2 * batch_size, dtype=tf.bool)
    sim_matrix = tf.where(mask, tf.constant(-1e9, dtype=sim_matrix.dtype), sim_matrix)
    
    # Labels: The positive pairs are shifted by batch_size
    labels = tf.concat([
        tf.range(batch_size, 2 * batch_size),
        tf.range(0, batch_size)
    ], axis=0)
    
    # Cross entropy loss
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, sim_matrix, from_logits=True
        )
    )
    
    return loss
