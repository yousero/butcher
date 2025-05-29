import sys
import os
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import project modules
from training.train_policy import PuzzleTrainer

def main():
    parser = argparse.ArgumentParser(description='Train on chess puzzles')
    parser.add_argument('--puzzles', type=str, required=True, help='Path to puzzles PGN file')
    parser.add_argument('--model', type=str, default="models/policy_net.h5", help='Path to model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--save-interval', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--output', type=str, default="models/trained_policy_net.h5", help='Output model path')
    args = parser.parse_args()

    trainer = PuzzleTrainer(args.model)
    trainer.load_puzzles(args.puzzles)
    
    print(f"Training on {len(trainer.puzzles)} puzzles for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch()
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        
        if epoch % args.save_interval == 0:
            interim_path = f"{args.output}_epoch{epoch}.h5"
            trainer.save_model(interim_path)
            print(f"Model saved to {interim_path}")
    
    trainer.save_model(args.output)
    print(f"Training complete. Model saved to {args.output}")

if __name__ == "__main__":
    main()