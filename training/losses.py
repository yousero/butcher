import tensorflow as tf

def policy_crossentropy(y_true, y_pred):
    """Функция потерь для политики"""
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=False)
