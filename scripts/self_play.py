from self_play.game_simulator import SelfPlay
import argparse
import chess.pgn

def main():
    parser = argparse.ArgumentParser(description='Self-play simulation')
    parser.add_argument('--model', type=str, default="models/policy_net.h5", help='Path to model')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--output', type=str, default="self_play.pgn", help='Output PGN file')
    args = parser.parse_args()

    simulator = SelfPlay(args.model)
    with open(args.output, "w") as pgn_file:
        for i in range(args.games):
            print(f"Playing game {i+1}/{args.games}")
            game_history = simulator.simulate_game()
            
            # Save game to PGN
            game = chess.pgn.Game()
            node = game
            for _, move in game_history:
                node = node.add_variation(move)
            
            game.headers["Event"] = "Butcher Self-Play"
            game.headers["White"] = "Butcher AI"
            game.headers["Black"] = "Butcher AI"
            game.headers["Result"] = game.headers.get("Result", "*")
            
            print(game, file=pgn_file, end="\n\n")
            print(f"Game {i+1} saved")

if __name__ == "__main__":
    main()
