import chess
from engine.move_encoding import move_to_index as encode_move


class MoveGenerator:
    @staticmethod
    def generate_legal_moves(board: chess.Board):
        """Легальные ходы с простой сортировкой: взятия > превращения > остальные."""
        legal_moves = list(board.generate_legal_moves())
        captures = [m for m in legal_moves if board.is_capture(m)]
        promotions = [m for m in legal_moves if m.promotion]
        others = [m for m in legal_moves
                  if m not in captures and m not in promotions]
        return captures + promotions + others

    @staticmethod
    def move_to_tensor(move: chess.Move, board: chess.Board = None) -> int:
        """Индекс хода в фиксированной AlphaZero-кодировке (4672)."""
        return encode_move(move)
