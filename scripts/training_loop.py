import argparse
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from self_play.data_generator import DataGenerator
from training.self_play_trainer import SelfPlayTrainer
from engine.nn_model import PolicyModel

def training_loop(initial_model="models/initial.h5", 
                  iterations=10, 
                  games_per_iter=100,
                  epochs_per_iter=5,
                  output_dir="models"):
    
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check initial model
    if not os.path.exists(initial_model):
        print(f"Initial model not found at {initial_model}, creating new model...")
        model = PolicyModel()
        model.save_model(initial_model)
        print(f"New model created at {initial_model}")
    
    current_model = initial_model
    print(f"\nStarting training loop with model: {current_model}")
    
    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration+1}/{iterations} ===")
        
        # 1. Self-play: generate data
        print(f"Generating {games_per_iter} self-play games...")
        generator = DataGenerator(current_model)
        training_data = generator.generate_games(games_per_iter)
        print(f"Generated {len(training_data)} training positions")
        
        # 2. Train on new data
        trainer = SelfPlayTrainer(current_model)
        new_model_path = os.path.join(output_dir, f"iter_{iteration+1}.h5")
        
        print(f"Training on new data for {epochs_per_iter} epochs...")
        trainer.train(training_data, epochs=epochs_per_iter)
        trainer.save_model(new_model_path)
        print(f"Model saved to {new_model_path}")
        
        # 3. Update model for next iteration
        current_model = new_model_path
    
    print("\nTraining completed!")
    return current_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-play training loop')
    parser.add_argument('--initial_model', type=str, default="models/initial.h5", 
                        help='Path to initial model')
    parser.add_argument('--iterations', type=int, default=10, 
                        help='Number of training iterations')
    parser.add_argument('--games_per_iter', type=int, default=100, 
                        help='Number of self-play games per iteration')
    parser.add_argument('--epochs_per_iter', type=int, default=5, 
                        help='Training epochs per iteration')
    parser.add_argument('--output_dir', type=str, default="models", 
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    final_model = training_loop(
        initial_model=args.initial_model,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        epochs_per_iter=args.epochs_per_iter,
        output_dir=args.output_dir
    )
    
    print(f"\nFinal model saved at: {final_model}")
