import tensorflow as tf

def policy_crossentropy(y_true, y_pred):
    """Policy loss function"""
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=False)
