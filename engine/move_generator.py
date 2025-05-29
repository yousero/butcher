import chess
import numpy as np

class MoveGenerator:
    @staticmethod
    def generate_legal_moves(board):
        """Генерация легальных ходов с базовой сортировкой"""
        legal_moves = list(board.generate_legal_moves())
        
        # Приоритет: взятия > промоции > обычные ходы
        captures = [m for m in legal_moves if board.is_capture(m)]
        promotions = [m for m in legal_moves if m.promotion]
        others = [m for m in legal_moves if m not in captures and m not in promotions]
        
        return captures + promotions + others

    @staticmethod
    def move_to_tensor(move, board):
        """Конвертация хода в индекс тензора политики"""
        # Создаем словарь всех возможных ходов
        move_dict = {}
        for idx, m in enumerate(board.generate_legal_moves()):
            move_dict[m.uci()] = idx
        
        return move_dict.get(move.uci(), 0)
