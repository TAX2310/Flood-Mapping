import argparse
from pathlib import Path
import sys

# Add the root directory to the Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import src.util.io as io
import src.train.training as training

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=Path)

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    # Load the configuration from the specified path
    cfg_path = args.path  
    cfg = io.load_config_pickle(cfg_path)

    # Train the model using the loaded configuration
    training.train_model(cfg)

    # Remove the temporary configuration file if it exists
    if cfg_path.exists():
        print("Removing temporary config.")
        print("")
        cfg_path.unlink()