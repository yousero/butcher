import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.uci import ButcherEngine

if __name__ == "__main__":
    model_path = "models/policy_net.h5"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    engine = ButcherEngine(model_path)
    print(f"Butcher Engine started with model: {model_path}")
    engine.uci_loop()
