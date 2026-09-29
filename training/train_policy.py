import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from training.puzzle_loader import load_puzzles
from training.losses import masked_policy_loss
from engine.board import ButcherBoard
from engine.nn_model import PolicyModel
from engine.move_encoding import POLICY_SIZE, move_to_index, legal_move_mask


class PuzzleTrainer:
    def __init__(self, model_path=None):
        self.input_shape = (8, 8, 18)
        self.batch_size = 32
        self.puzzles = []

        self.model = self.load_or_create_model(model_path)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

    def load_or_create_model(self, path):
        if path and os.path.exists(path):
            try:
                return PolicyModel.load_model(path)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead.")
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
        if not self.puzzles:
            raise ValueError("No puzzles loaded. Call load_puzzles first.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        indices = np.random.choice(len(self.puzzles), batch_size)
        X = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        y = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        M = np.zeros((batch_size, POLICY_SIZE), dtype=bool)

        valid = 0
        for idx in indices:
            fen, solution = self.puzzles[idx]
            try:
                board = ButcherBoard(fen)
                if solution not in board.legal_moves:
                    continue
                t = board.to_tensor()
                if np.any(np.isnan(t)):
                    continue
                mi = move_to_index(solution)
                X[valid] = t
                y[valid, mi] = 1.0
                M[valid] = legal_move_mask(board)
                # гарантируем, что маска содержит правильный ход
                M[valid, mi] = True
                valid += 1
            except Exception as e:
                print(f"Skipping invalid puzzle: {fen} | {solution} - {e}")
                continue

        if valid == 0:
            raise ValueError("No valid samples in batch")
        return X[:valid], y[:valid], M[:valid]

    def train_epoch(self):
        if not self.puzzles:
            raise ValueError("No puzzles loaded. Call load_puzzles first.")

        total_loss = 0.0
        steps = len(self.puzzles) // self.batch_size
        valid_steps = 0

        for _ in tqdm(range(steps), desc="Training"):
            try:
                X_batch, y_batch, m_batch = self.prepare_batch(self.batch_size)
                if np.any(np.isnan(X_batch)) or np.any(np.isnan(y_batch)):
                    continue

                with tf.GradientTape() as tape:
                    logits = self.model.model(X_batch, training=True)
                    loss_vec = masked_policy_loss(y_batch, logits, m_batch)
                    loss = tf.reduce_mean(loss_vec)

                if tf.math.is_nan(loss):
                    continue

                grads = tape.gradient(loss, self.model.model.trainable_variables)
                if any(g is None for g in grads):
                    continue
                if any(tf.reduce_any(tf.math.is_nan(g)) for g in grads):
                    continue

                self.optimizer.apply_gradients(
                    zip(grads, self.model.model.trainable_variables)
                )
                total_loss += float(loss.numpy())
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
