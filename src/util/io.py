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

def get_leaf_subdirs(root_dir):
    root_dir = Path(root_dir)

    return [
        p for p in root_dir.rglob("*")
        if p.is_dir() and not any(child.is_dir() for child in p.iterdir())
    ]

import ipywidgets as widgets
from IPython.display import display, clear_output

def select_model(cfg, on_select, button_text="Run", description="Model:"):
    root_dir = cfg.EXP_DIR / cfg.DATASET / cfg.DATA_TYPE
    subdirs = get_leaf_subdirs(root_dir)

    if not subdirs:
        raise ValueError(f"No model directories found in {root_dir}")

    dropdown = widgets.Dropdown(
        options=subdirs,
        value=subdirs[0],
        description=description,
        layout=widgets.Layout(width="700px"),
        style={"description_width": "initial"}
    )

    run_button = widgets.Button(description=button_text)
    output = widgets.Output()

    def on_click(button):
        with output:
            clear_output(wait=True)

            model_dir = cfg.EXP_DIR / dropdown.value
            on_select(model_dir)

    run_button.on_click(on_click)

    display(dropdown, run_button, output)

    return dropdown, run_button, output

from pathlib import Path
import numpy as np
import rasterio


def save_prediction_tif(pred_array, reference_tif_path, output_tif_path, dtype="uint8"):
    """
    Save a predicted mask/probability map as a GeoTIFF using metadata
    from a reference image.

    Args:
        pred_array: 2D numpy array or torch tensor, shape [H, W]
        reference_tif_path: path to source GeoTIFF to copy metadata from
        output_tif_path: path to save prediction GeoTIFF
        dtype: "uint8" for binary masks, "float32" for probability maps
    """

    reference_tif_path = Path(reference_tif_path)
    output_tif_path = Path(output_tif_path)
    output_tif_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert torch tensor to numpy if needed
    if hasattr(pred_array, "detach"):
        pred_array = pred_array.detach().cpu().numpy()

    pred_array = np.asarray(pred_array)

    # Remove extra dimensions if shape is [1, H, W] or [1, 1, H, W]
    pred_array = np.squeeze(pred_array)

    if pred_array.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {pred_array.shape}")

    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype=dtype,
        compress="lzw",
        nodata=0 if dtype == "uint8" else None,
    )

    pred_array = pred_array.astype(dtype)

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(pred_array, 1)

def export_prediction_tifs(results):
    output_dir = cfg.EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        sample_id = result["sample_id"]

        prob_path = output_dir / f"{sample_id}_prob.tif"
        pred_path = output_dir / f"{sample_id}_pred.tif"

        save_prediction_tif(
            result["prob"].numpy(),
            result["reference_path"],
            prob_path,
            dtype="float32",
        )

        save_prediction_tif(
            result["pred"].numpy(),
            result["reference_path"],
            pred_path,
            dtype="uint8",
        )