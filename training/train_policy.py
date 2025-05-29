import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from .puzzle_loader import load_puzzles
from engine.board import ButcherBoard
from engine.nn_model import PolicyModel
from training.losses import policy_crossentropy

class PuzzleTrainer:
    def __init__(self, model_path=None):
        self.model = self.load_or_create_model(model_path)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.puzzles = []
        self.batch_size = 128
        self.input_shape = (8, 8, 18)  # Важно: должно соответствовать board.py!
    
    def load_or_create_model(self, path):
        if path and os.path.exists(path):
            try:
                return PolicyModel.load_model(path)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        print("Creating new model")
        return PolicyModel(input_shape=self.input_shape)
    
    def load_puzzles(self, pgn_path, max_puzzles=10000):
        self.puzzles = load_puzzles(pgn_path, max_puzzles)
    
    def prepare_batch(self, batch_size):
        """Подготовка батча данных"""
        indices = np.random.choice(len(self.puzzles), batch_size)
        X = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        y = np.zeros((batch_size, self.model.policy_shape), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            fen, solution = self.puzzles[idx]
            try:
                # Создаем доску с валидацией FEN
                board = ButcherBoard(fen)
                
                # Проверяем, что ход легальный
                if solution not in board.legal_moves:
                    # Если ход нелегальный, пропускаем пример
                    continue
                    
                # Входные данные
                X[i] = board.to_tensor()
                
                # Целевой вектор
                move_idx = self.model.move_to_index(solution, board)
                if move_idx < self.model.policy_shape:
                    y[i, move_idx] = 1
                    
            except Exception as e:
                print(f"Skipping invalid puzzle: {fen} | {solution} - {str(e)}")
                continue
        
        return X, y
    
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
