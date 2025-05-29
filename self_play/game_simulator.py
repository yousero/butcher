import chess
import numpy as np
from tqdm import tqdm
from engine.board import ButcherBoard

class SelfPlay:
    def __init__(self, model_path, max_moves=200):
        self.model = PolicyModel.load_model(model_path)
        self.board = ButcherBoard()
        self.max_moves = max_moves
    
    def simulate_game(self):
        """Симуляция одной игры"""
        game_history = []
        self.board.reset()
        move_count = 0
        
        while not self.board.is_game_over() and move_count < self.max_moves:
            # Конвертация доски в тензор
            board_tensor = self.board.to_tensor()
            
            # Предсказание модели
            move = self.model.predict(self.board)
            
            # Сохранение состояния
            game_history.append((self.board.fen(), move))
            
            # Применение хода
            self.board.push(move)
            move_count += 1
        
        return game_history
