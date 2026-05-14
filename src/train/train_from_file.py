import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

import src.util.io as io
import src.train.training as training

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=Path)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.path  
    cfg = io.load_config_pickle(cfg_path)

    training.train_model(cfg)

    if cfg_path.exists():
        print("Training complete. Removing temporary config.")
        cfg_path.unlink()