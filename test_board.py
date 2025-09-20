import os
import sys
import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.board import ButcherBoard

# Test valid FEN strings
test_fens = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "invalid fen string"
]

for fen in test_fens:
    print(f"\nTesting FEN: {fen}")
    try:
        board = ButcherBoard(fen)
        print("Board created successfully")
        print(board)
        print("Tensor shape:", board.to_tensor().shape)
    except Exception as e:
        print(f"Error: {str(e)}")
