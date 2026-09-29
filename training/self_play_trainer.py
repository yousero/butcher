import numpy as np
import tensorflow as tf
from tqdm import tqdm

from engine.nn_model import PolicyModel
from engine.board import ButcherBoard
from engine.move_encoding import POLICY_SIZE, move_to_index


class SelfPlayTrainer:
    def __init__(self, model_path):
        self.model = PolicyModel.load_model(model_path)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.batch_size = 128

    def train(self, training_data, epochs=5):
        X, y = self.prepare_data(training_data)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(self.batch_size)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for Xb, yb in tqdm(dataset, desc=f"Epoch {epoch}/{epochs}"):
                loss = self.train_step(Xb, yb)
                epoch_loss += float(loss)
                n_batches += 1
            print(f"Epoch {epoch} - Avg Loss: {epoch_loss / max(n_batches,1):.4f}")

    def train_step(self, X_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = self.model.model(X_batch, training=True)
            loss = self.loss_fn(y_batch, predictions)
        grads = tape.gradient(loss, self.model.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.model.trainable_variables)
        )
        return loss.numpy()

    def prepare_data(self, training_data):
        X = []
        y = []
        for dp in training_data:
            board_tensor = dp["board_tensor"]
            move = dp["best_move"]
            target = np.zeros(POLICY_SIZE, dtype=np.float32)
            target[move_to_index(move)] = 1.0
            X.append(board_tensor)
            y.append(target)
        return np.array(X), np.array(y)

    def save_model(self, path):
        self.model.save_model(path)
