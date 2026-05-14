import json
import csv
from src.util import metrics
import torch
from pathlib import Path
from datetime import datetime
import rasterio
import pickle
from types import SimpleNamespace

from src.config import CFG

def save_config(cfg, path):
    path = Path(path)

    cfg_dict = {
        k: str(v) for k, v in vars(cfg).items()
        if not k.startswith("__")
    }

    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=4)

def save_model(model, path):
    path = Path(path)
    torch.save(model.state_dict(), path)

def save_history_csv(history, path):
    path = Path(path)

    if len(history) == 0:
        return

    keys = history[0].keys()

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)

def save_summary(summary, path):
    path = Path(path)

    with open(path, "w") as f:
        json.dump(summary, f, indent=4)

def save_checkpoint(path, model, optimizer, scheduler, epoch, history, best_val_loss, best_epoch, best_metrics):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }

    torch.save(checkpoint, path)

def experiment_dir(cfg):

    exp_dir = Path(cfg.EXP_DIR) / Path(cfg.DATASET) / Path(cfg.DATA_TYPE) / f"{cfg.MODEL}__{cfg.SPLIT_METHOD}__{f'rotation_{cfg.USE_ROTATIONS}'}" / f'lr_{cfg.LR}__bs_{cfg.BATCH_SIZE}__e_{cfg.EPOCHS}'

    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir

def save_config_pickle(cfg, path):
    path = Path(path)

    with open(path, "wb") as f:
        pickle.dump(cfg, f)


def load_config_pickle(path):
    path = Path(path)

    with open(path, "rb") as f:
        cfg = pickle.load(f)

    return cfg        

def load_config_json(path):
    """
    Load a JSON config file and return it as an object.

    Example:
        cfg = load_config_json("config.json")
        print(cfg.LR)
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config_dict = json.load(f)

    return SimpleNamespace(**config_dict)

def read_image_tif(path):
    """
    Returns:
        torch.Tensor [C, H, W] float32
    """
    with rasterio.open(path) as src:
        arr = src.read()  # numpy [C,H,W]

    return torch.from_numpy(arr).float()

def read_mask_tif(path):
    """
    Returns:
        torch.Tensor [H, W] int64
    """
    with rasterio.open(path) as src:
        arr = src.read(1)  # numpy [H,W]

    return torch.from_numpy(arr).long()