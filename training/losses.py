import tensorflow as tf

NEG_INF = -1e9


def masked_policy_loss(y_true, logits, legal_mask):
    """Cross-entropy по логитам с маской легальных ходов.

    y_true:     (B, POLICY_SIZE) one-hot по легальному ходу
    logits:     (B, POLICY_SIZE)
    legal_mask: (B, POLICY_SIZE) bool
    """
    # Маскируем нелегальные логиты
    logits = tf.where(
        legal_mask,
        logits,
        tf.fill(tf.shape(logits), tf.constant(NEG_INF, dtype=logits.dtype)),
    )

    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Label smoothing ТОЛЬКО среди легальных ходов:
    #   target = (1 - eps) * one_hot + eps / n_legal * legal_mask
    smoothing = 0.05
    n_legal = tf.reduce_sum(
        tf.cast(legal_mask, logits.dtype), axis=-1, keepdims=True
    )  # (B, 1)

    y_smooth = y_true * (1.0 - smoothing)
    y_smooth = y_smooth + smoothing / n_legal * tf.cast(legal_mask, logits.dtype)

    # Нелегальные: y_smooth = 0 → вклада в loss нет
    loss = -tf.reduce_sum(y_smooth * log_probs, axis=-1)
    return loss
