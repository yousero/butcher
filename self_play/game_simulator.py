import numpy as np
import chess

from engine.board import ButcherBoard
from engine.nn_model import PolicyModel
from engine.move_encoding import move_to_index, index_to_move, legal_move_mask


class SelfPlay:
    def __init__(self, model_path, max_moves=200, temperature=1.0):
        self.model = PolicyModel.load_model(model_path)
        self.max_moves = max_moves
        self.temperature = temperature

    def _choose_move(self, board: ButcherBoard) -> chess.Move:
        probs = self.model.predict_probs(board)  # уже с маской
        legal = list(board.legal_moves)
        p = np.array([probs[move_to_index(m)] for m in legal], dtype=np.float64)
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / s

        if self.temperature <= 1e-3:
            return legal[int(np.argmax(p))]

        logits = np.log(p + 1e-12) / self.temperature
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        return legal[int(np.random.choice(len(legal), p=w))]

    def simulate_game(self, max_moves=None):
        if max_moves is None:
            max_moves = self.max_moves
        game_history = []
        board = ButcherBoard()
        move_count = 0
        while not board.is_game_over(claim_draw=True) and move_count < max_moves:
            move = self._choose_move(board)
            game_history.append((board.fen(), move))
            board.push(move)
            move_count += 1
        return game_history
