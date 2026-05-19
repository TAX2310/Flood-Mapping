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

def save_checkpoint(path, model, optimizer, scheduler, epoch, history, best_val_loss, best_epoch, best_metrics, epochs_without_improvement):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epochs_without_improvement": epochs_without_improvement
    }

    torch.save(checkpoint, path)

def experiment_dir(cfg):

    exp_dir = Path(cfg.EXP_DIR) / Path(cfg.DATASET) / Path(cfg.DATA_TYPE) / f'lr_{cfg.LR}' / f'bs_{cfg.BATCH_SIZE}' / f'wd_{cfg.WEIGHT_DECAY}' / f'dr_{cfg.DROPOUT_RATE}'

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

    
    # Remove environment-specific fields
    fields_to_remove = {
        "ROOT",
        "DEVICE",
    }

    for key in fields_to_remove:
        config_dict.pop(key, None)

    if "DATA_TYPE" in config_dict:
        if config_dict["DATA_TYPE"] == "Sentinel1_SAR":
            from src.config import S1_CFG
            return S1_CFG(**config_dict)
        elif config_dict["DATA_TYPE"] == "Sentinel2_Optical":
            from src.config import S2_CFG
            return S2_CFG(**config_dict)
        elif config_dict["DATA_TYPE"] == "Fusion_SAR_Optical":
            from src.config import Fusion_CFG
            return Fusion_CFG(**config_dict)
    elif "DATA_TYPE" not in config_dict:
        raise ValueError("Config file must contain 'DATA_TYPE' key")    
    
def update_summary(path, new_values, section=None):
    """
    Update an existing summary.json with new values.

    If section is provided, values are saved under that section.
    Example:
        section="test_metrics"
    """

    path = Path(path)

    if path.exists():
        with open(path, "r") as f:
            summary = json.load(f)
    else:
        summary = {}

    if section is None:
        summary.update(new_values)
    else:
        summary[section] = new_values

    with open(path, "w") as f:
        json.dump(summary, f, indent=4)

    return summary

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