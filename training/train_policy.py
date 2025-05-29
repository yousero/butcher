import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .puzzle_loader import load_puzzles
from engine.board import ButcherBoard
from engine.nn_model import PolicyModel
from .losses import policy_crossentropy

class PuzzleTrainer:
    def __init__(self, model_path=None):
        self.model = self.load_or_create_model(model_path)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.puzzles = []
        self.batch_size = 128
    
    def load_or_create_model(self, path):
        if path:
            try:
                return PolicyModel.load_model(path)
            except:
                print(f"Model not found at {path}, creating new model")
        return PolicyModel()
    
    def load_puzzles(self, pgn_path, max_puzzles=10000):
        self.puzzles = load_puzzles(pgn_path, max_puzzles)
    
    def prepare_batch(self, batch_size):
        """Подготовка батча данных"""
        indices = np.random.choice(len(self.puzzles), batch_size)
        X = []
        y = []
        
        for idx in indices:
            fen, solution = self.puzzles[idx]
            board = ButcherBoard(fen)
            
            # Входные данные
            X.append(board.to_tensor())
            
            # Целевой вектор
            target = np.zeros(self.model.policy_shape, dtype=np.float32)
            move_idx = self.model.move_to_index(solution, board)
            if move_idx < self.model.policy_shape:
                target[move_idx] = 1
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_epoch(self):
        """Обучение на одной эпохе"""
        total_loss = 0
        steps = len(self.puzzles) // self.batch_size
        
        for _ in tqdm(range(steps), desc="Training"):
            X_batch, y_batch = self.prepare_batch(self.batch_size)
            
            with tf.GradientTape() as tape:
                predictions = self.model.model(X_batch, training=True)
                loss = tf.reduce_mean(policy_crossentropy(y_batch, predictions))
            
            grads = tape.gradient(loss, self.model.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.model.trainable_variables))
            
            total_loss += loss.numpy()
        
        return total_loss / steps
    
    def save_model(self, path):
        self.model.save_model(path)
