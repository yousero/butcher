import numpy as np
import tensorflow as tf
from tqdm import tqdm
from engine.nn_model import PolicyModel

class SelfPlayTrainer:
    def __init__(self, model_path):
        self.model = PolicyModel.load_model(model_path)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.batch_size = 128
    
    def train(self, training_data, epochs=5):
        """Training on self-play data"""
        # Convert data to tensors
        X, y = self.prepare_data(training_data)
        
        # Create TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(self.batch_size)
        
        # Training loop over epochs
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in tqdm(dataset, desc=f"Epoch {epoch}/{epochs}"):
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")
    
    def train_step(self, X_batch, y_batch):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Model prediction
            predictions = self.model.model(X_batch, training=True)
            
            # Calculate loss
            loss = self.loss_fn(y_batch, predictions)
        
        # Compute gradients and update weights
        grads = tape.gradient(loss, self.model.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.model.trainable_variables))
        
        return loss.numpy()
    
    def prepare_data(self, training_data):
        """Prepare data for training"""
        X = []
        y = []
        
        for data_point in training_data:
            # Extract data
            board_tensor = data_point['board_tensor']
            move = data_point['best_move']
            fen = data_point['fen']
            
            # Create target vector
            target = np.zeros(self.model.policy_shape, dtype=np.float32)
            move_idx = self.model.move_to_index(move, fen)
            if move_idx < self.model.policy_shape:
                target[move_idx] = 1
            
            X.append(board_tensor)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def save_model(self, path):
        """Save the model"""
        self.model.save_model(path)
