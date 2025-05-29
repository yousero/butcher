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
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00001,
            clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        self.puzzles = []
        self.batch_size = 32
        self.input_shape = (8, 8, 18)  # Important: must match board.py!
    
    def load_or_create_model(self, path):
        if path and os.path.exists(path):
            try:
                return PolicyModel.load_model(path)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        print("Creating new model")
        return PolicyModel(input_shape=self.input_shape)
    
    def load_puzzles(self, pgn_path, max_puzzles=10000):
        if not os.path.exists(pgn_path):
            raise FileNotFoundError(f"Puzzle file not found: {pgn_path}")
            
        if max_puzzles < 1:
            raise ValueError("max_puzzles must be positive")
            
        self.puzzles = load_puzzles(pgn_path, max_puzzles)
        if not self.puzzles:
            raise ValueError("No valid puzzles loaded")
    
    def prepare_batch(self, batch_size):
        """Prepare data batch with additional validation"""
        if not self.puzzles:
            raise ValueError("No puzzles loaded. Call load_puzzles first.")
            
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
            
        indices = np.random.choice(len(self.puzzles), batch_size)
        X = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        y = np.zeros((batch_size, self.model.policy_shape), dtype=np.float32)
        
        valid_samples = 0
        for i, idx in enumerate(indices):
            fen, solution = self.puzzles[idx]
            try:
                # Create board with FEN validation
                board = ButcherBoard(fen)
                
                # Check if move is legal
                if solution not in board.legal_moves:
                    continue
                    
                # Input data with validation
                board_tensor = board.to_tensor()
                if np.any(np.isnan(board_tensor)):
                    print(f"Warning: NaN detected in board tensor for FEN: {fen}")
                    continue
                    
                X[valid_samples] = board_tensor
                
                # Target vector with validation
                move_idx = self.model.move_to_index(solution, board)
                if move_idx < self.model.policy_shape:
                    y[valid_samples, move_idx] = 1
                    valid_samples += 1
                    
            except Exception as e:
                print(f"Skipping invalid puzzle: {fen} | {solution} - {str(e)}")
                continue
        
        if valid_samples == 0:
            raise ValueError("No valid samples in batch")
            
        return X[:valid_samples], y[:valid_samples]
    
    def train_epoch(self):
        """Train for one epoch with improved stability"""
        if not self.puzzles:
            raise ValueError("No puzzles loaded. Call load_puzzles first.")
            
        total_loss = 0
        steps = len(self.puzzles) // self.batch_size
        valid_steps = 0
        
        for _ in tqdm(range(steps), desc="Training"):
            try:
                X_batch, y_batch = self.prepare_batch(self.batch_size)
                
                # Validate input data
                if np.any(np.isnan(X_batch)) or np.any(np.isnan(y_batch)):
                    print("Warning: NaN detected in batch data, skipping")
                    continue
                
                with tf.GradientTape() as tape:
                    predictions = self.model.model(X_batch, training=True)
                    
                    # Validate predictions
                    if tf.reduce_any(tf.math.is_nan(predictions)):
                        print("Warning: NaN detected in model predictions, skipping batch")
                        continue
                        
                    loss = tf.reduce_mean(policy_crossentropy(y_batch, predictions))
                
                if tf.math.is_nan(loss):
                    print("Warning: NaN loss detected, skipping batch")
                    continue
                
                grads = tape.gradient(loss, self.model.model.trainable_variables)
                
                # More thorough gradient validation
                if any(grad is None for grad in grads):
                    print("Warning: None gradients detected, skipping batch")
                    continue
                    
                if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in grads):
                    print("Warning: NaN gradients detected, skipping batch")
                    continue
                
                # Apply gradients with validation
                try:
                    self.optimizer.apply_gradients(
                        zip(grads, self.model.model.trainable_variables))
                except Exception as e:
                    print(f"Warning: Error applying gradients: {e}")
                    continue
                
                total_loss += loss.numpy()
                valid_steps += 1
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        if valid_steps == 0:
            raise ValueError("No valid training steps completed")
            
        return total_loss / valid_steps
    
    def save_model(self, path):
        if not path:
            raise ValueError("Model path cannot be empty")
        self.model.save_model(path)
