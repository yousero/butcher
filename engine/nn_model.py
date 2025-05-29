import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense
from engine.board import ButcherBoard

class PolicyModel:
    def __init__(self, input_shape=(8, 8, 18), policy_shape=4672):
        self.input_shape = input_shape
        self.policy_shape = policy_shape
        self.model = self.build_model()
    
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # Блок 1
        x = Conv2D(256, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Residual блоки
        for _ in range(5):
            residual = x
            x = Conv2D(256, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(256, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.add([x, residual])
            x = ReLU()(x)
        
        # Policy head
        policy = Conv2D(128, 1, activation='relu')(x)
        policy = Flatten()(policy)
        policy = Dense(self.policy_shape, activation='softmax', name='policy')(policy)
        
        return tf.keras.Model(inputs, policy)
    
    def predict(self, board):
        tensor = board.to_tensor()
        move_probs = self.model.predict(tensor[np.newaxis, ...], verbose=0)
        return self.decode_move(move_probs[0], board)
    
    def decode_move(self, probs, board):
        legal_moves = list(board.generate_legal_moves())
        move_indices = [self.move_to_index(m, board) for m in legal_moves]
        
        # Фильтрация только легальных ходов
        legal_probs = np.zeros(self.policy_shape)
        for idx in move_indices:
            if idx < len(legal_probs):
                legal_probs[idx] = probs[idx]
        
        # Выбор хода с максимальной вероятностью
        move_idx = np.argmax(legal_probs)
        return next(m for m in legal_moves if self.move_to_index(m, board) == move_idx)
    
    def move_to_index(self, move, board):
        """Конвертация хода в индекс (упрощенная реализация)"""
        from_sq = move.from_square
        to_sq = move.to_square
        return from_sq * 64 + to_sq

    def save_model(self, path):
        self.model.save(path)
    
    @staticmethod
    def load_model(path):
        model = PolicyModel()
        model.model = tf.keras.models.load_model(path)
        return model

    @staticmethod
    def load_model(path):
        try:
            import tensorflow as tf
            model = PolicyModel()
            model.model = tf.keras.models.load_model(path, compile=False)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to new model
            return PolicyModel()