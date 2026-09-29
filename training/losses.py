import tensorflow as tf

NEG_INF = -1e9


def masked_policy_loss(y_true, logits, legal_mask):
    logits = tf.where(
        legal_mask,
        logits,
        tf.fill(tf.shape(logits), tf.constant(NEG_INF, dtype=logits.dtype)),
    )

    log_probs = tf.nn.log_softmax(logits, axis=-1)

    smoothing = 0.05
    n_legal = tf.reduce_sum(
        tf.cast(legal_mask, logits.dtype), axis=-1, keepdims=True
    )  # (B, 1)

    y_smooth = y_true * (1.0 - smoothing)
    y_smooth = y_smooth + smoothing / n_legal * tf.cast(legal_mask, logits.dtype)

    # Нелегальные: y_smooth = 0 → вклада в loss нет
    loss = -tf.reduce_sum(y_smooth * log_probs, axis=-1)
    return loss
