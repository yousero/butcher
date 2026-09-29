import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    Flatten, Dense, Dropout,
)
from tensorflow.keras.regularizers import l2

from engine.board import ButcherBoard
from engine.move_encoding import (
    POLICY_SIZE,
    move_to_index as encode_move,
    legal_move_mask,
)


NEG_INF = -1e9


class PolicyModel:
    def __init__(self, input_shape=(8, 8, 18), policy_shape=POLICY_SIZE):
        self.input_shape = input_shape
        self.policy_shape = policy_shape
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)

        x = Conv2D(128, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.1)(x)

        for _ in range(3):
            residual = x
            x = Conv2D(128, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(0.1)(x)
            x = Conv2D(128, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.add([x, residual])
            x = ReLU()(x)

        # Policy head БЕЗ softmax — логиты
        policy = Conv2D(64, 1, activation="relu",
                        kernel_regularizer=l2(1e-4))(x)
        policy = BatchNormalization()(policy)
        policy = Flatten()(policy)
        policy = Dropout(0.2)(policy)
        policy = Dense(self.policy_shape, name="policy_logits",
                       kernel_regularizer=l2(1e-4))(policy)

        return tf.keras.Model(inputs, policy)

    # -------- inference --------

    def predict_logits(self, board: ButcherBoard) -> np.ndarray:
        if not isinstance(board, ButcherBoard):
            raise ValueError("Board must be an instance of ButcherBoard")
        tensor = board.to_tensor()
        return self.model.predict(tensor[np.newaxis, ...], verbose=0)[0]

    def predict_probs(self, board: ButcherBoard) -> np.ndarray:
        """Вероятности с маской нелегальных ходов."""
        logits = self.predict_logits(board)
        mask = legal_move_mask(board)
        logits = np.where(mask, logits, NEG_INF)
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def predict(self, board: ButcherBoard):
        probs = self.predict_probs(board)
        idx = int(np.argmax(probs))
        from engine.move_encoding import index_to_move
        return index_to_move(idx, board)

    @staticmethod
    def move_to_index(move, board=None) -> int:
        return encode_move(move)

    # -------- persistence --------

    def save_model(self, path):
        if not path:
            raise ValueError("Model path cannot be empty")
        self.model.save(path)

    @staticmethod
    def load_model(path):
        if not path:
            raise ValueError("Model path cannot be empty")
        model = PolicyModel()
        model.model = tf.keras.models.load_model(path, compile=False)
        out_shape = model.model.output_shape
        if out_shape[-1] != POLICY_SIZE:
            raise ValueError(
                f"Model output size {out_shape[-1]} != POLICY_SIZE {POLICY_SIZE}"
            )
        return model
