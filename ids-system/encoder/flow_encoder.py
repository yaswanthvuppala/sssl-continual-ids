import tensorflow as tf

def build_flow_encoder(input_dim: int, hidden_dim: int = 512, embed_dim: int = 256, l2_reg: float = 1e-4) -> tf.keras.Model:
    """
    Builds the main frozen encoder network.
    Uses LayerNormalization and L2 regularization for contrastive SSL stability.
    """
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,), name="flow_features")
    
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Final embedding layer (linear activation before projection head)
    embeddings = tf.keras.layers.Dense(embed_dim, activation=None, kernel_regularizer=regularizer, name="embedding")(x)
    
    model = tf.keras.Model(inputs, embeddings, name="flow_encoder")
    return model
