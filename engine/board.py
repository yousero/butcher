import chess
import numpy as np

class ButcherBoard(chess.Board):
    def __init__(self, fen=chess.STARTING_FEN):
        super().__init__(fen)
        self.input_planes = 18
    
    def to_tensor(self):
        """Конвертация доски в 8x8x18 тензор"""
        tensor = np.zeros((8, 8, self.input_planes), dtype=np.float32)
        
        # Плоскости 0-11: Фигуры (6 типов × 2 цвета)
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square, piece in self.piece_map().items():
            row, col = 7 - square // 8, square % 8
            plane_offset = 0 if piece.color == chess.WHITE else 6
            plane = piece_map[piece.piece_type] + plane_offset
            tensor[row, col, plane] = 1
        
        # Плоскость 12: Двойные ходы пешек
        if self.ep_square:
            ep_row, ep_col = 7 - self.ep_square // 8, self.ep_square % 8
            tensor[ep_row, ep_col, 12] = 1
        
        # Плоскость 13: Рокировки (белые короткие)
        tensor[:, :, 13] = self.has_kingside_castling_rights(chess.WHITE)
        
        # Плоскость 14: Рокировки (белые длинные)
        tensor[:, :, 14] = self.has_queenside_castling_rights(chess.WHITE)
        
        # Плоскость 15: Рокировки (черные короткие)
        tensor[:, :, 15] = self.has_kingside_castling_rights(chess.BLACK)
        
        # Плоскость 16: Рокировки (черные длинные)
        tensor[:, :, 16] = self.has_queenside_castling_rights(chess.BLACK)
        
        # Плоскость 17: Цвет хода (0=белые, 1=черные)
        tensor[:, :, 17] = int(self.turn == chess.BLACK)
        
        return tensor

    def reset(self):
        self.set_fen(chess.STARTING_FEN)
    
    def set_fen(self, fen):
        super().__init__(fen)
    
    def __str__(self):
        return super().__str__()
