import os
import numpy as np
from tqdm import tqdm
from self_play.game_simulator import SelfPlay
from engine.board import ButcherBoard

class DataGenerator:
    def __init__(self, model_path):
        self.simulator = SelfPlay(model_path)
    
    def generate_games(self, num_games, max_moves=200):
        """Генерация игровых данных"""
        training_data = []
        
        for game_idx in tqdm(range(num_games), desc="Generating games"):
            # Симуляция одной игры
            game_history = self.simulator.simulate_game(max_moves)
            
            # Сохраняем все позиции и ходы из игры
            for fen, move in game_history:
                board = ButcherBoard(fen)
                training_data.append({
                    'fen': fen,
                    'board_tensor': board.to_tensor(),
                    'best_move': move
                })
        
        print(f"Generated {len(training_data)} training positions from {num_games} games")
        return training_data
