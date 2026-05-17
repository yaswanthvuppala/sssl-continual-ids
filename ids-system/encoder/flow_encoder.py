import tensorflow as tf

def build_flow_encoder(input_dim: int, hidden_dim: int = 512, embed_dim: int = 256) -> tf.keras.Model:
    """
    Builds the main frozen encoder network.
    It takes tabular flow features and maps them to a dense embedding space.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="flow_features")
    
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Final embedding layer (linear activation typical before projection head)
    embeddings = tf.keras.layers.Dense(embed_dim, activation=None, name="embedding")(x)
    
    model = tf.keras.Model(inputs, embeddings, name="flow_encoder")
    return model
