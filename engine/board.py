import chess
import numpy as np

class ButcherBoard(chess.Board):
    def __init__(self, fen=chess.STARTING_FEN):
        try:
            # Use base constructor
            super().__init__(fen)
        except ValueError as e:
            # In case of error, use starting position
            super().__init__(chess.STARTING_FEN)
            print(f"Invalid FEN: {fen}")
            print(f"Error details: {str(e)}")
            print("Using default starting position instead")
    
    @property
    def input_planes(self):
        return 18

    def to_tensor(self):
        """Convert board to 8x8x18 tensor"""
        tensor = np.zeros((8, 8, self.input_planes), dtype=np.float32)
        
        # Planes 0-11: Pieces (6 types × 2 colors)
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
        
        # Plane 12: En passant square
        if self.ep_square is not None:
            ep_row, ep_col = 7 - self.ep_square // 8, self.ep_square % 8
            tensor[ep_row, ep_col, 12] = 1
        
        # Planes 13-16: Castling rights
        # Plane 13: White kingside
        tensor[:, :, 13] = self.has_kingside_castling_rights(chess.WHITE)
        # Plane 14: White queenside
        tensor[:, :, 14] = self.has_queenside_castling_rights(chess.WHITE)
        # Plane 15: Black kingside
        tensor[:, :, 15] = self.has_kingside_castling_rights(chess.BLACK)
        # Plane 16: Black queenside
        tensor[:, :, 16] = self.has_queenside_castling_rights(chess.BLACK)
        
        # Plane 17: Side to move (0 - White, 1 - Black)
        tensor[:, :, 17] = 1 if self.turn == chess.BLACK else 0
        
        return tensor

    def reset(self):
        """Reset board to starting position"""
        self.set_fen(chess.STARTING_FEN)
