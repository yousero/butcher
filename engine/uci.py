import chess
import sys
import argparse
import time
from .board import ButcherBoard
from .nn_model import PolicyModel

class ButcherEngine:
    def __init__(self, model_path="models/policy_net.h5"):
        self.board = ButcherBoard()
        self.model = PolicyModel.load_model(model_path)
        self.thinking_time = 1.0  # Default thinking time in seconds
    
    def set_position(self, fen=chess.STARTING_FEN):
        try:
            self.board.set_fen(fen)
        except ValueError:
            print(f"Invalid FEN: {fen}, using default")
            self.board.set_fen(chess.STARTING_FEN)
    
    def get_best_move(self):
        """Get best move from neural network with timing"""
        start_time = time.time()
        move = self.model.predict(self.board)
        elapsed = time.time() - start_time
        
        # Ensure we use at least minimal thinking time
        if elapsed < self.thinking_time:
            time.sleep(self.thinking_time - elapsed)
        
        return move
    
    def handle_position(self, command):
        """Parse UCI position command"""
        parts = command.split()
        if len(parts) < 2:
            return
        
        # Parse position type
        if parts[1] == 'startpos':
            self.set_position(chess.STARTING_FEN)
            moves_start = 2
        elif parts[1] == 'fen':
            fen_parts = parts[2:8]
            fen = " ".join(fen_parts)
            self.set_position(fen)
            moves_start = 8
        else:
            return
        
        # Apply moves if any
        if len(parts) > moves_start and parts[moves_start] == 'moves':
            for move_str in parts[moves_start+1:]:
                move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)
    
    def handle_go(self, command):
        """Parse UCI go command"""
        parts = command.split()
        self.thinking_time = 1.0  # default
        
        # Parse time controls
        if "movetime" in parts:
            idx = parts.index("movetime")
            self.thinking_time = float(parts[idx+1]) / 1000.0
        elif "wtime" in parts and self.board.turn == chess.WHITE:
            idx = parts.index("wtime")
            time_left = float(parts[idx+1]) / 1000.0
            self.thinking_time = max(0.1, min(time_left / 40.0, 5.0))
        elif "btime" in parts and self.board.turn == chess.BLACK:
            idx = parts.index("btime")
            time_left = float(parts[idx+1]) / 1000.0
            self.thinking_time = max(0.1, min(time_left / 40.0, 5.0))
        
        return self.get_best_move()
    
    def uci_loop(self):
        """Main UCI loop"""
        while True:
            try:
                cmd = input().strip()
            except EOFError:
                break
            
            if cmd == "uci":
                self.handle_uci()
            elif cmd == "isready":
                print("readyok")
            elif cmd == "ucinewgame":
                self.board.reset()
            elif cmd.startswith("position"):
                self.handle_position(cmd)
            elif cmd.startswith("go"):
                best_move = self.handle_go(cmd)
                print(f"bestmove {best_move.uci()}")
            elif cmd == "quit":
                break
            elif cmd == "print":
                print(self.board)
    
    def handle_uci(self):
        """Respond to UCI initialization"""
        print("id name Butcher Chess Engine")
        print("id author YourName")
        print("option name ModelPath type string default models/policy_net.h5")
        print("option name ThinkingTime type spin default 1000 min 10 max 60000")
        print("uciok")

def main():
    parser = argparse.ArgumentParser(description='Butcher Chess Engine (UCI)')
    parser.add_argument('--model', type=str, default="models/policy_net.h5", 
                        help='Path to the model file')
    args = parser.parse_args()
    
    try:
        engine = ButcherEngine(args.model)
        print(f"Butcher Engine started with model: {args.model}")
        engine.uci_loop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
