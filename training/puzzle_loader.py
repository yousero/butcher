import chess.pgn
import os
import chess

def load_puzzles(pgn_path, max_puzzles=None):
    puzzles = []
    count = 0
    invalid_count = 0
    
    with open(pgn_path) as pgn_file:
        while True:
            if max_puzzles and count >= max_puzzles:
                break
                
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
                
            try:
                board = game.board()
                
                # Проверка валидности FEN
                test_board = chess.Board(board.fen())
                
                # Извлекаем решение
                solution = None
                node = game
                while node:
                    if node.variations:
                        solution = node.variations[0].move
                        break
                    node = node.next()
                
                # Проверка валидности хода
                if solution and test_board.is_legal(solution):
                    puzzles.append((board.fen(), solution))
                    count += 1
                else:
                    invalid_count += 1
                    
            except (ValueError, AssertionError) as e:
                invalid_count += 1
                continue
    
    print(f"Loaded {len(puzzles)} valid puzzles, skipped {invalid_count} invalid from {os.path.basename(pgn_path)}")
    return puzzles
