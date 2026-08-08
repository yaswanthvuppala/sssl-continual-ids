import tensorflow as tf

def build_projection_head(in_dim: int = 256, out_dim: int = 128, l2_reg: float = 1e-4) -> tf.keras.Model:
    """
    Builds the non-linear projection head used during SimCLR pretraining.
    Uses LayerNormalization and L2 regularization. Discarded after pretraining.
    """
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(in_dim,), name="embedding")
    
    x = tf.keras.layers.Dense(in_dim, activation="relu", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Final projection layer to contrastive space
    z = tf.keras.layers.Dense(out_dim, activation=None, kernel_regularizer=regularizer, name="projection")(x)
    
    model = tf.keras.Model(inputs, z, name="projection_head")
    return model
