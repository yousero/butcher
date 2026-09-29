"""AlphaZero-подобное кодирование ходов: 8*8*73 = 4672 индексов.

Индекс = from_square * 73 + plane, где plane:
  0..55   — "ферзевые" ходы: 8 направлений × 7 дистанций
  56..63  — коневые ходы (8 направлений)
  64..72  — недо-превращения: 3 фигуры × 3 направления
            (порядок: 0=forward-left, 1=forward, 2=forward-right
             относительно пешки, для белых dr=+1, для чёрных dr=-1)

Нумерация полей как в python-chess: a1=0, ..., h8=63.
"""
from __future__ import annotations

import chess
import numpy as np

POLICY_SIZE = 8 * 8 * 73  # 4672

# Порядок направлений "ферзевых" ходов
QUEEN_DIRS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]

# 8 коневых ходов
KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _underpromo_dir_index(df: int, dr: int):
    """df,dr → 0/1/2 для underpromotion, или None если не подходит.

    d=0: forward-left, d=1: forward, d=2: forward-right
    "forward" = +rank для белых, -rank для чёрных.
    df одинаков для обоих: -1, 0, +1.
    """
    if dr == 1:  # белые
        table = {(-1, 1): 0, (0, 1): 1, (1, 1): 2}
    elif dr == -1:  # чёрные
        table = {(-1, -1): 0, (0, -1): 1, (1, -1): 2}
    else:
        return None
    return table.get((df, dr))


def _underpromo_offsets(d: int, from_sq: int):
    """d=0/1/2 → (df, dr). Цвет определяется по горизонтали пешки."""
    rank = chess.square_rank(from_sq)
    if rank == 6:      # 7-я горизонталь — белая пешка
        dr = 1
    elif rank == 1:    # 2-я горизонталь — чёрная пешка
        dr = -1
    else:
        raise ValueError(
            f"Underpromotion from wrong rank: {chess.square_name(from_sq)}"
        )
    df = (-1, 0, 1)[d]
    return df, dr


def move_to_index(move: chess.Move) -> int:
    """chess.Move → индекс в [0, POLICY_SIZE)."""
    from_sq = move.from_square
    df = chess.square_file(move.to_square) - chess.square_file(from_sq)
    dr = chess.square_rank(move.to_square) - chess.square_rank(from_sq)

    # Underpromotion (конь / слон / ладья), оба цвета
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        d = _underpromo_dir_index(df, dr)
        if d is None:
            raise ValueError(
                f"Bad underpromotion direction df={df} dr={dr} for {move}"
            )
        p = UNDERPROMO_PIECES.index(move.promotion)
        plane = 64 + p * 3 + d
        return from_sq * 73 + plane

    # Коневые ходы
    if (df, dr) in KNIGHT_DIRS:
        return from_sq * 73 + 56 + KNIGHT_DIRS.index((df, dr))

    # Ферзевые ходы (включая превращение в ферзя)
    if df == 0 and dr == 0:
        raise ValueError(f"Zero-length move: {move}")
    if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        raise ValueError(f"Not a queen-like move: {move}")
    dist = max(abs(df), abs(dr))
    if not 1 <= dist <= 7:
        raise ValueError(f"Distance out of range: {move}")
    step = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
    d = QUEEN_DIRS.index(step)
    return from_sq * 73 + d * 7 + (dist - 1)


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Индекс в [0, POLICY_SIZE) → chess.Move (с проверкой легальности)."""
    if not 0 <= idx < POLICY_SIZE:
        raise ValueError(f"Index out of range: {idx}")
    from_sq, plane = divmod(idx, 73)

    if plane >= 64:
        p, d = divmod(plane - 64, 3)
        piece = UNDERPROMO_PIECES[p]
        df, dr = _underpromo_offsets(d, from_sq)
    elif plane >= 56:
        piece = None
        df, dr = KNIGHT_DIRS[plane - 56]
    else:
        d, dist = divmod(plane, 7)
        sf, sr = QUEEN_DIRS[d]
        df, dr = sf * (dist + 1), sr * (dist + 1)
        piece = None

    tf = chess.square_file(from_sq) + df
    tr = chess.square_rank(from_sq) + dr
    if not (0 <= tf <= 7 and 0 <= tr <= 7):
        raise ValueError(f"Decoded move leaves board (idx={idx})")

    to_sq = chess.square(tf, tr)
    move = chess.Move(from_sq, to_sq, promotion=piece)

    # Недо-превращения уже имеют promotion. Для «ферзевых» ходов promotion=None,
    # но если без него ход нелегален — возможно, это превращение в ферзя.
    if piece is None and move not in board.legal_moves:
        promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        if promo in board.legal_moves:
            move = promo

    if move not in board.legal_moves:
        raise ValueError(f"Decoded move {move} is not legal in {board.fen()}")
    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean-маска легальных ходов размера POLICY_SIZE."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for m in board.legal_moves:
        mask[move_to_index(m)] = True
    return mask


def decode_best_legal(probs: np.ndarray, board: chess.Board):
    """Лучший легальный ход по вектору вероятностей."""
    if probs.shape[0] < POLICY_SIZE:
        raise ValueError(f"probs shape {probs.shape} < {POLICY_SIZE}")
    best = None
    best_p = -1.0
    for m in board.legal_moves:
        p = float(probs[move_to_index(m)])
        if p > best_p:
            best, best_p = m, p
    if best is None:
        raise ValueError("No legal moves on the board")
    return best, best_p

def legal_move_indices(board: chess.Board) -> list[tuple[chess.Move, int]]:
    """Список (move, index) для всех легальных ходов в позиции."""
    return [(m, move_to_index(m)) for m in board.legal_moves]
