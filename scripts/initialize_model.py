import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.nn_model import PolicyModel

def main():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model = PolicyModel()
    model_path = os.path.join(model_dir, "policy_net.h5")
    model.model.save(model_path)
    print(f"Initial model created at {model_path}")

if __name__ == "__main__":
    main()
