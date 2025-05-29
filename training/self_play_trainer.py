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
        """Обучение на данных самоигры"""
        # Преобразование данных в тензоры
        X, y = self.prepare_data(training_data)
        
        # Создание TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(self.batch_size)
        
        # Цикл обучения по эпохам
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
        """Один шаг обучения"""
        with tf.GradientTape() as tape:
            # Предсказание модели
            predictions = self.model.model(X_batch, training=True)
            
            # Расчет потерь
            loss = self.loss_fn(y_batch, predictions)
        
        # Вычисление градиентов и обновление весов
        grads = tape.gradient(loss, self.model.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.model.trainable_variables))
        
        return loss.numpy()
    
    def prepare_data(self, training_data):
        """Подготовка данных для обучения"""
        X = []
        y = []
        
        for data_point in training_data:
            # Извлекаем данные
            board_tensor = data_point['board_tensor']
            move = data_point['best_move']
            fen = data_point['fen']
            
            # Создаем целевой вектор
            target = np.zeros(self.model.policy_shape, dtype=np.float32)
            move_idx = self.model.move_to_index(move, fen)
            if move_idx < self.model.policy_shape:
                target[move_idx] = 1
            
            X.append(board_tensor)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def save_model(self, path):
        """Сохраняем модель"""
        self.model.save_model(path)
