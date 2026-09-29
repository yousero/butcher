import tensorflow as tf

NEG_INF = -1e9


def masked_policy_loss(y_true, logits, legal_mask):
    """Cross-entropy по логитам с маской легальных ходов.

    y_true:     (B, POLICY_SIZE) one-hot
    logits:     (B, POLICY_SIZE)
    legal_mask: (B, POLICY_SIZE) bool
    """
    logits = tf.where(legal_mask, logits, tf.fill(tf.shape(logits), NEG_INF))

    # label smoothing
    smoothing = 0.05
    n = tf.cast(tf.shape(y_true)[-1], tf.float32)
    y_true = y_true * (1.0 - smoothing) + smoothing / n

    # log_softmax вручную, чтобы не ловить NaN
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_sum(y_true * log_probs, axis=-1)
    return loss
