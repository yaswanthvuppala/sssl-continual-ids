import tensorflow as tf

def build_classifier_head(embed_dim: int = 256, num_classes: int = 2, name: str = "classifier_head", l2_reg: float = 1e-4) -> tf.keras.Model:
    """
    Builds a lightweight, regularized classification head for frozen SSL representations.
    A 2-layer MLP prevents memorization on downstream labeled data.
    """
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(embed_dim,), name="embedding_input")
    
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=regularizer, name=f"logits_{name}")(x)
    
    model = tf.keras.Model(inputs, outputs, name=name)
    return model
