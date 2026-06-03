from email.mime import image
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import rasterio
import torch


import src.util.io as io

def plot_metrics(model_dir, save_path=None):
    """
    Read a metrics CSV and plot train loss vs validation loss, precision vs recall, and IoU vs F1.
    """

    plot_train_val_loss(model_dir, save_path=save_path)
    plot_precision_recall(model_dir, save_path=save_path)
    plot_iou_f1(model_dir, save_path=save_path)

def plot_train_val_loss(model_dir, save_path=None):
    """
    Read a metrics CSV and plot train loss vs validation loss.
    """

    csv_path = Path(model_dir) / "metrics.csv"

    summary_path = Path(model_dir) / "summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        title = f'{summary["title"]} - Train Val Loss'
    else:
        title = f'Train Val Loss - training in progress...'

    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"epoch", "train_loss", "val_loss"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_precision_recall(model_dir, save_path=None):
    """
    Read a metrics CSV and plot precision vs recall over epochs.
    """

    csv_path = Path(model_dir) / "metrics.csv"
    summary_path = Path(model_dir) / "summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        title = f'{summary["title"]} - Precision Recall'
    else:
        title = f'Precision Recall - training in progress...'

    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"epoch", "precision", "recall"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["precision"], marker="o", label="Precision")
    plt.plot(df["epoch"], df["recall"], marker="o", label="Recall")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_iou_f1(model_dir, save_path=None):
    """
    Read a metrics CSV and plot IoU vs F1 over epochs.
    """

    csv_path = Path(model_dir) / "metrics.csv"
    summary_path = Path(model_dir) / "summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        title = f'{summary["title"]} - IoU F1'
    else:
        title = f'IoU F1 - training in progress...'

    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"epoch", "iou", "f1"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["iou"], marker="o", label="IoU")
    plt.plot(df["epoch"], df["f1"], marker="o", label="F1 / Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def view_training_metrics(cfg):
    io.select_model(
        cfg,
        on_select=lambda model_dir: plot_metrics(model_dir),
        button_text="Plot metrics"
    )

def plot_f1_iou_bar(title, models, f1_scores, iou_scores, figsize=(12, 5)):
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=figsize)

    bars_iou = plt.bar(x - width / 2, iou_scores, width, label="IoU", color="blue")
    bars_f1 = plt.bar(x + width / 2, f1_scores, width, label="F1", color="orange")

    max_iou = max(iou_scores)
    max_f1 = max(f1_scores)

    # Add IoU values above bars
    for bar in bars_iou:
        height = bar.get_height()

        text_color = "red" if height == max_iou else "black"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=text_color,
            fontweight="bold" if height == max_iou else "normal"
        )

    # Add F1 values above bars
    for bar in bars_f1:
        height = bar.get_height()

        text_color = "red" if height == max_f1 else "black"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=text_color,
            fontweight="bold" if height == max_f1 else "normal"
        )

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def view_f1_iou_bar(cfg):
    root_dir = cfg.EXP_DIR / cfg.DATASET / cfg.DATA_TYPE
    title = f"{root_dir.name} Training Performance"
    subdirs = io.get_leaf_subdirs(root_dir)

    print(f"Found {len(subdirs)} model subdirectories in {root_dir}")

    models = []
    f1_scores = []
    iou_scores = []
    for subdir in subdirs:
        summary = subdir / "summary.json"
        if summary.exists():
            with open(summary, "r") as f:
                s = json.load(f)
                models.append(s["title"])
                f1_scores.append(s["f1"])
                iou_scores.append(s["iou"])


    plot_f1_iou_bar(title, models, f1_scores, iou_scores)

def plot_mask_tif(tif_path, title="Mask", band=1, figsize=(7, 7)):
    """
    Read a mask GeoTIFF and display it.

    Args:
        tif_path: path to the mask .tif file
        title: plot title
        band: raster band to read
        figsize: matplotlib figure size
    """

    with rasterio.open(tif_path) as src:
        mask = src.read(band)

    print("Shape:", mask.shape)
    print("Dtype:", mask.dtype)
    print("Unique values:", np.unique(mask))

    plt.figure(figsize=figsize)
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

    return mask

def plot_mask_tensor(mask, title="Mask", figsize=(7, 7)):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    plt.figure(figsize=figsize)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.show()

    return mask

def plot_fp_fn_mask_tensor(mask, pred, title="FP/FN Mask", figsize=(7, 7)):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    mask = np.squeeze(mask)
    pred = np.squeeze(pred)

    fp_fn_mask = mask * 2 + pred

    cmap = ListedColormap([
        "black",   # 0
        "blue",   # 1
        "red",  # 2
        "white"      # 3
    ])

    
    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap.N
    )

    plt.figure(figsize=figsize)
    plt.imshow(fp_fn_mask, cmap=cmap, norm=norm)

    legend_items = [
        mpatches.Patch(color="black", label="No flood"),
        mpatches.Patch(color="blue", label="FP"),
        mpatches.Patch(color="red", label="FN"),
        mpatches.Patch(color="white", label="Flood"),
    ]

    plt.legend(handles=legend_items, title="Comparison")

    plt.title(title)
    plt.axis("off")
    plt.show()

    return fp_fn_mask

def plot_prob_tensor(prob, title="Probability", figsize=(7, 7)):
    if isinstance(prob, torch.Tensor):
        prob = prob.detach().cpu().numpy()

    prob = np.squeeze(prob)

    plt.figure(figsize=figsize)
    plt.imshow(prob, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

    return prob

def plot_s1_tif(tif_path, title="S1_sar_image", figsize=(7, 7)):
    """
    Plot S1 GeoTIFF as a single QGIS-like RGB image.

    Fixed rendering:
        Red   = Band 1
        Green = Band 2
        Blue  = empty / zero
        Stretch = min-max per band
    """

    with rasterio.open(tif_path) as src:
        image = src.read()  # [C, H, W]

    if image.ndim != 3:
        raise ValueError(f"Expected shape [C, H, W], got {image.shape}")

    if image.shape[0] < 2:
        raise ValueError("Expected at least 2 bands for S1")

    band1 = image[0].astype(np.float32)
    band2 = image[1].astype(np.float32)

    b1_min, b1_max = np.nanmin(band1), np.nanmax(band1)
    b2_min, b2_max = np.nanmin(band2), np.nanmax(band2)

    red = (band1 - b1_min) / (b1_max - b1_min + 1e-8)
    green = (band2 - b2_min) / (b2_max - b2_min + 1e-8)
    blue = np.zeros_like(red)

    red = np.clip(red, 0, 1)
    green = np.clip(green, 0, 1)

    rgb = np.dstack([red, green, blue])

    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

    return rgb

def plot_s1_tensor(tensor, title="S1 SAR Image", figsize=(7, 7)):
    """
    Plot S1 tensor as a single QGIS-like RGB image.

    Fixed rendering:
        Red   = Band 1
        Green = Band 2
        Blue  = empty / zero
        Stretch = min-max per band

    Accepts:
        [C, H, W]
        [1, C, H, W]
    """

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    tensor = np.squeeze(tensor)

    if tensor.ndim != 3:
        raise ValueError(f"Expected shape [C, H, W], got {tensor.shape}")

    if tensor.shape[0] < 2:
        raise ValueError("Expected at least 2 bands for S1")

    band1 = tensor[0].astype(np.float32)
    band2 = tensor[1].astype(np.float32)

    b1_min, b1_max = np.nanmin(band1), np.nanmax(band1)
    b2_min, b2_max = np.nanmin(band2), np.nanmax(band2)

    red = (band1 - b1_min) / (b1_max - b1_min + 1e-8)
    green = (band2 - b2_min) / (b2_max - b2_min + 1e-8)
    blue = np.zeros_like(red)

    red = np.clip(red, 0, 1)
    green = np.clip(green, 0, 1)

    image = np.dstack([red, green, blue])

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

    return image

def plot_s2_tif(tif_path, title="S2 Optical Image", figsize=(7, 7)):
    """
    Plot Sentinel-2 GeoTIFF as an RGB image.

    Assumes STURM S2 band order:
        Band 1 = B2 Blue
        Band 2 = B3 Green
        Band 3 = B4 Red

    Fixed rendering:
        Red   = B4
        Green = B3
        Blue  = B2
        Stretch = min-max per band
    """

    with rasterio.open(tif_path) as src:
        image = src.read()  # [C, H, W]

    if image.ndim != 3:
        raise ValueError(f"Expected shape [C, H, W], got {image.shape}")

    if image.shape[0] < 3:
        raise ValueError("Expected at least 3 bands for Sentinel-2 RGB")

    blue_band = image[0].astype(np.float32)
    green_band = image[1].astype(np.float32)
    red_band = image[2].astype(np.float32)

    r_min, r_max = np.nanmin(red_band), np.nanmax(red_band)
    g_min, g_max = np.nanmin(green_band), np.nanmax(green_band)
    b_min, b_max = np.nanmin(blue_band), np.nanmax(blue_band)

    red = (red_band - r_min) / (r_max - r_min + 1e-8)
    green = (green_band - g_min) / (g_max - g_min + 1e-8)
    blue = (blue_band - b_min) / (b_max - b_min + 1e-8)

    red = np.clip(red, 0, 1)
    green = np.clip(green, 0, 1)
    blue = np.clip(blue, 0, 1)

    image = np.dstack([red, green, blue])

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

    return image

def plot_s2_tensor(tensor, title="S2 Optical Image", figsize=(7, 7)):
    """
    Plot Sentinel-2 tensor as an RGB image.

    Assumes STURM S2 band order:
        Band 1 = B2 Blue
        Band 2 = B3 Green
        Band 3 = B4 Red

    Fixed rendering:
        Red   = B4
        Green = B3
        Blue  = B2
        Stretch = min-max per band

    Accepts:
        [C, H, W]
        [1, C, H, W]
    """

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    tensor = np.squeeze(tensor)

    if tensor.ndim != 3:
        raise ValueError(f"Expected shape [C, H, W], got {tensor.shape}")

    if tensor.shape[0] < 3:
        raise ValueError("Expected at least 3 bands for Sentinel-2 RGB")

    blue_band = tensor[0].astype(np.float32)
    green_band = tensor[1].astype(np.float32)
    red_band = tensor[2].astype(np.float32)

    r_min, r_max = np.nanmin(red_band), np.nanmax(red_band)
    g_min, g_max = np.nanmin(green_band), np.nanmax(green_band)
    b_min, b_max = np.nanmin(blue_band), np.nanmax(blue_band)

    red = (red_band - r_min) / (r_max - r_min + 1e-8)
    green = (green_band - g_min) / (g_max - g_min + 1e-8)
    blue = (blue_band - b_min) / (b_max - b_min + 1e-8)

    red = np.clip(red, 0, 1)
    green = np.clip(green, 0, 1)
    blue = np.clip(blue, 0, 1)

    image = np.dstack([red, green, blue])

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

    return image

def plot_s1_results(results, figsize=(7,7)):
    for sample in results:
        for key, value in sample.items():
            print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

        image = sample["image"].cpu()
        mask = sample["mask"].cpu()
        pred = sample["pred"].cpu()
        prob = sample["prob"].cpu()

        plot_s1_tensor(image)
        plot_mask_tensor(mask, figsize=figsize)
        plot_mask_tensor(pred, title="Predicted Mask", figsize=figsize)
        plot_fp_fn_mask_tensor(mask, pred, figsize=figsize)
        plot_prob_tensor(prob, figsize=figsize)

def plot_s2_results(results, figsize=(7,7)):
    for sample in results:
        for key, value in sample.items():
            print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

        image = sample["image"].cpu()
        mask = sample["mask"].cpu()
        pred = sample["pred"].cpu()
        prob = sample["prob"].cpu()

        plot_s2_tensor(image)
        plot_mask_tensor(mask, figsize=figsize)
        plot_mask_tensor(pred, title="Predicted Mask", figsize=figsize)
        plot_fp_fn_mask_tensor(mask, pred, figsize=figsize)
        plot_prob_tensor(prob, figsize=figsize)

def plot_fusion_results(results, figsize=(7,7)):
    for sample in results:
        for key, value in sample.items():
            print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

        s1_image = sample["s1_image"].cpu()
        s2_image = sample["s2_image"].cpu()
        mask = sample["mask"].cpu()
        pred = sample["pred"].cpu()
        prob = sample["prob"].cpu()

        plot_s1_tensor(s1_image, title="S1 SAR Image", figsize=figsize)
        plot_s2_tensor(s2_image, title="S2 Optical Image", figsize=figsize)
        plot_mask_tensor(mask, title="Ground Truth Mask", figsize=figsize)
        plot_mask_tensor(pred, title="Predicted Mask", figsize=figsize)
        plot_fp_fn_mask_tensor(mask, pred, title="FP/FN Comparison", figsize=figsize)
        plot_prob_tensor(prob, title="Predicted Probability", figsize=figsize)

def plot_metric_distribution(
    results,
    metric="iou",
    bins=20,
    title=None,
    figsize=(8, 5),
    show_stats=True
):
    """
    Plot the distribution of a metric from a list of result dictionaries.

    Parameters
    ----------
    results : list[dict] or pd.DataFrame
        Inference results containing metrics such as iou, f1, precision, recall.
    metric : str
        Metric to plot. Default is "iou".
    bins : int
        Number of histogram bins.
    title : str or None
        Plot title. If None, a default title is used.
    figsize : tuple
        Figure size.
    show_stats : bool
        Whether to show mean and median lines.
    """

    # Convert list of dictionaries to DataFrame if needed
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")

    values = df[metric].dropna()

    plt.figure(figsize=figsize)

    plt.hist(
        values,
        bins=bins,
        edgecolor="black"
    )

    if show_stats:
        mean_value = values.mean()
        median_value = values.median()

        plt.axvline(mean_value, linestyle="--", linewidth=2, label=f"Mean: {mean_value:.3f}")
        plt.axvline(median_value, linestyle=":", linewidth=2, label=f"Median: {median_value:.3f}")
        plt.legend()

    plt.xlabel(metric.upper())
    plt.ylabel("Number of samples")

    if title is None:
        title = f"Distribution of {metric.upper()} scores"

    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.show()
