import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from engine.board import ButcherBoard

class PolicyModel:
    def __init__(self, input_shape=(8, 8, 18), policy_shape=4672):
        self.input_shape = input_shape
        self.policy_shape = policy_shape
        self.model = self.build_model()
    
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # Initial normalization
        x = BatchNormalization()(inputs)
        
        # Block 1 - Initial convolution
        x = Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.1)(x)
        
        # Residual blocks with improved stability
        for _ in range(3):  # Reduced number of blocks
            residual = x
            x = Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(0.1)(x)
            x = Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.add([x, residual])
            x = ReLU()(x)
        
        # Policy head with improved stability
        policy = Conv2D(64, 1, activation='relu', kernel_regularizer=l2(1e-4))(x)
        policy = BatchNormalization()(policy)
        policy = Flatten()(policy)
        policy = Dropout(0.2)(policy)
        policy = Dense(self.policy_shape, activation='softmax', name='policy',
                      kernel_regularizer=l2(1e-4))(policy)
        
        return tf.keras.Model(inputs, policy)
    
    def predict(self, board):
        if not isinstance(board, ButcherBoard):
            raise ValueError("Board must be an instance of ButcherBoard")
            
        tensor = board.to_tensor()
        move_probs = self.model.predict(tensor[np.newaxis, ...], verbose=0)
        return self.decode_move(move_probs[0], board)
    
    def decode_move(self, probs, board):
        if not isinstance(board, ButcherBoard):
            raise ValueError("Board must be an instance of ButcherBoard")
            
        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        move_indices = [self.move_to_index(m, board) for m in legal_moves]
        
        # Filter only legal moves
        legal_probs = np.zeros(self.policy_shape)
        for idx in move_indices:
            if idx < len(legal_probs):
                legal_probs[idx] = probs[idx]
        
        # Choose move with highest probability
        move_idx = np.argmax(legal_probs)
        return next(m for m in legal_moves if self.move_to_index(m, board) == move_idx)
    
    def move_to_index(self, move, board):
        """Convert move to index"""
        if not isinstance(board, ButcherBoard):
            raise ValueError("Board must be an instance of ButcherBoard")
            
        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        try:
            return legal_moves.index(move)
        except ValueError:
            raise ValueError(f"Illegal move: {move}")

    def save_model(self, path):
        if not path:
            raise ValueError("Model path cannot be empty")
        self.model.save(path)
    
    @staticmethod
    def load_model(path):
        if not path:
            raise ValueError("Model path cannot be empty")
            
        try:
            model = PolicyModel()
            model.model = tf.keras.models.load_model(path, compile=False)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to new model
            return PolicyModel()