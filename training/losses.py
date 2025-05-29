import tensorflow as tf

def policy_crossentropy(y_true, y_pred):
    """Policy loss function with numerical stability improvements"""
    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Add epsilon to avoid numerical instability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=False)
