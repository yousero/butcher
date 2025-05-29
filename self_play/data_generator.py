import chess.pgn
import numpy as np
import os
from tqdm import tqdm
from .game_simulator import SelfPlay
from engine.board import ButcherBoard

class DataGenerator:
    def __init__(self, model_path):
        self.simulator = SelfPlay(model_path)
    
    def generate_games(self, num_games, output_dir="data/self_play"):
        """Генерация игровых данных"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for game_idx in tqdm(range(num_games), desc="Generating games"):
            game_history = self.simulator.simulate_game()
            pgn_path = os.path.join(output_dir, f"game_{game_idx}.pgn")
            
            with open(pgn_path, "w") as pgn_file:
                game = chess.pgn.Game()
                node = game
                
                for fen, move in game_history:
                    node = node.add_variation(move)
                    node.comment = fen
                
                game.headers["Event"] = "Butcher Self-Play"
                game.headers["White"] = "Butcher AI"
                game.headers["Black"] = "Butcher AI"
                game.headers["Result"] = game.headers.get("Result", "*")
                
                print(game, file=pgn_file, end="\n\n")
