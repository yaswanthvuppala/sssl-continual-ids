import tensorflow as tf

def build_projection_head(in_dim: int = 256, out_dim: int = 128) -> tf.keras.Model:
    """
    Builds the non-linear projection head used during SimCLR pretraining.
    This head is discarded after pretraining.
    """
    inputs = tf.keras.Input(shape=(in_dim,), name="embedding")
    
    x = tf.keras.layers.Dense(in_dim, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Final projection layer to contrastive space
    z = tf.keras.layers.Dense(out_dim, activation=None, name="projection")(x)
    
    model = tf.keras.Model(inputs, z, name="projection_head")
    return model
