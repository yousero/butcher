import chess.pgn
import os

def load_puzzles(pgn_path, max_puzzles=None):
    puzzles = []
    count = 0
    
    with open(pgn_path) as pgn_file:
        while True:
            if max_puzzles and count >= max_puzzles:
                break
                
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
                
            # Извлекаем решение из первого комментария
            solution = None
            node = game
            while node:
                if node.variations:
                    solution = node.variations[0].move
                    break
                node = node.next()
            
            if solution and game.board().is_legal(solution):
                puzzles.append((game.board().fen(), solution))
                count += 1
    
    print(f"Loaded {len(puzzles)} puzzles from {os.path.basename(pgn_path)}")
    return puzzles
