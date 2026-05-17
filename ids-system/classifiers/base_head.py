import tensorflow as tf

def build_classifier_head(embed_dim: int = 256, num_classes: int = 2, name: str = "classifier_head") -> tf.keras.Model:
    """
    Builds a generic classification head that sits on top of the frozen SSL encoder.
    """
    inputs = tf.keras.Input(shape=(embed_dim,), name="embedding_input")
    
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name=f"logits_{name}")(x)
    
    model = tf.keras.Model(inputs, outputs, name=name)
    return model
