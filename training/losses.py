import tensorflow as tf

def policy_crossentropy(y_true, y_pred):
    """Policy loss function with improved numerical stability"""
    # Normalize predictions to sum to 1
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Clip predictions to avoid log(0) with larger epsilon
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)
    
    # Add epsilon to avoid numerical instability
    epsilon = 1e-5
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy with label smoothing
    smoothing = 0.1
    y_true = y_true * (1 - smoothing) + smoothing / tf.cast(tf.shape(y_true)[-1], tf.float32)
    
    # Calculate cross entropy
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=False)
