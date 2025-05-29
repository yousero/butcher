import chess
import numpy as np

class MoveGenerator:
    @staticmethod
    def generate_legal_moves(board):
        """Generate legal moves with basic sorting"""
        legal_moves = list(board.generate_legal_moves())
        
        # Priority: captures > promotions > regular moves
        captures = [m for m in legal_moves if board.is_capture(m)]
        promotions = [m for m in legal_moves if m.promotion]
        others = [m for m in legal_moves if m not in captures and m not in promotions]
        
        return captures + promotions + others

    @staticmethod
    def move_to_tensor(move, board):
        """Convert move to policy tensor index"""
        # Create dictionary of all possible moves
        move_dict = {}
        for idx, m in enumerate(board.generate_legal_moves()):
            move_dict[m.uci()] = idx
        
        return move_dict.get(move.uci(), 0)
