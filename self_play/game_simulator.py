import chess
from engine.board import ButcherBoard
from engine.nn_model import PolicyModel

class SelfPlay:
    def __init__(self, model_path, max_moves=200):
        self.model = PolicyModel.load_model(model_path)
        self.max_moves = max_moves
    
    def simulate_game(self, max_moves=None):
        """Simulate one game between two AI copies"""
        if max_moves is None:
            max_moves = self.max_moves
        
        game_history = []
        board = ButcherBoard()
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            # Predict move
            move = self.model.predict(board)
            
            # Save state
            game_history.append((board.fen(), move))
            
            # Apply move
            board.push(move)
            move_count += 1
        
        return game_history
