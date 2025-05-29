import chess
from engine.board import ButcherBoard
from engine.nn_model import PolicyModel

class SelfPlay:
    def __init__(self, model_path, max_moves=200):
        self.model = PolicyModel.load_model(model_path)
        self.max_moves = max_moves
    
    def simulate_game(self, max_moves=None):
        """Симуляция одной игры между двумя копиями ИИ"""
        if max_moves is None:
            max_moves = self.max_moves
        
        game_history = []
        board = ButcherBoard()
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            # Предсказание хода
            move = self.model.predict(board)
            
            # Сохранение состояния
            game_history.append((board.fen(), move))
            
            # Применение хода
            board.push(move)
            move_count += 1
        
        return game_history
