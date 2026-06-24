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

def plot_f1_iou_bar(title, models, f1_scores, iou_scores, figsize=(12, 5), save_path=None):
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=figsize)
    bars_iou = plt.bar(x - width / 2, iou_scores, width, label="IoU", color="C0")
    bars_f1 = plt.bar(x + width / 2, f1_scores, width, label="F1", color="C1")
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
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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

def plot_metric_distribution_from_csv(
    results_csv,
    metric="iou",
    bins=20,
    title=None,
    figsize=(8, 5),
    show_stats=True,
    save_path=None,
):
    results_csv = Path(results_csv)
    df = pd.read_csv(results_csv)
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    plt.figure(figsize=figsize)
    plt.hist(
        values,
        bins=bins,
        color="C0",
        edgecolor="black",
        alpha=0.75,
        label="Samples",
    )
    if show_stats:
        mean_value = values.mean()
        median_value = values.median()
        std_value = values.std()
        plt.axvline(
            mean_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_value:.3f}",
        )
        plt.axvline(
            median_value,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Median = {median_value:.3f}",
        )
        # -1 SD
        plt.axvline(
            mean_value - std_value,
            color="purple",
            linestyle="-.",
            linewidth=2,
            label=f"±1 SD = {std_value:.3f}",
        )
        # +1 SD, no extra legend label
        plt.axvline(
            mean_value + std_value,
            color="purple",
            linestyle="-.",
            linewidth=2,
        )
        plt.legend()
    plt.xlabel(metric.upper())
    plt.ylabel("Number of samples")
    if title is None:
        title = f"Distribution of {metric.upper()} scores"
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_iou_vs_flood_scatter(
    csv_paths,
    iou_col="iou",
    flood_pixels_col="flood_pixels",
    total_pixels_col="total_valid_pixels",
    data_type_col="data_type",
    figsize=(8, 6),
    title="Flood coverage vs IoU",
):
    """
    Plot IoU against flood percentage for multiple model result CSVs.
    Uses dedicated labels, markers, and colours based on data_type.
    """

    label_map = {
        "Sentinel1_SAR": "S1",
        "Sentinel2_Optical": "S2",
        "Fusion_SAR_Optical": "Fusion",
    }

    marker_map = {
        "S1": "o",
        "S2": "^",
        "Fusion": "*",
    }

    color_map = {
        "S1": "blue",
        "S2": "orange",
        "Fusion": "green",
    }

    plt.figure(figsize=figsize)

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if data_type_col not in df.columns:
            raise ValueError(f"'{data_type_col}' column not found in {csv_path}")

        data_type = df[data_type_col].iloc[0]
        label = label_map.get(data_type, data_type)

        marker = marker_map.get(label, "o")
        color = color_map.get(label, None)

        df["flood_percentage"] = (
            df[flood_pixels_col] / df[total_pixels_col]
        ) * 100

        plt.scatter(
            df["flood_percentage"],
            df[iou_col],
            label=label,
            marker=marker,
            color=color,
            alpha=0.7,
        )

    plt.xlabel("Flood coverage (%)")
    plt.ylabel("IoU")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(title="Model")
    plt.show()

def plot_average_iou_per_event(
    results_csv,
    event_col="ems_code",
    iou_col="iou",
    figsize=(12, 5),
    title="Average IoU per EMSR event",
    sort=True,
    save_path=None,
):
    """
    Plot average IoU per EMSR event from one test results CSV.
    """
    df = pd.read_csv(Path(results_csv))
    event_iou = (
        df.groupby(event_col)[iou_col]
        .mean()
        .reset_index()
        .rename(columns={event_col: "event_id", iou_col: "mean_iou"})
    )
    if sort:
        event_iou = event_iou.sort_values("mean_iou")
    plt.figure(figsize=figsize)
    bars = plt.bar(
        event_iou["event_id"],
        event_iou["mean_iou"],
        color="C0",
        edgecolor="black",
    )
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xlabel("EMSR event")
    plt.ylabel("Average IoU")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return event_iou

def plot_fusion_improvement_distribution(
    csv_paths,
    sample_id_col="tile_id",
    data_type_col="data_type",
    iou_col="iou",
    figsize=(8, 5),
    bins=30,
    save_path=None,
):
    """
    Plot Fusion IoU improvement over the best single-modality baseline
    using S1, S2, and Fusion result CSV files.
    Args:
        csv_paths: list of CSV paths containing S1, S2, and Fusion results.
        sample_id_col: column used to match samples.
        data_type_col: column identifying data type/model.
        iou_col: IoU column.
        figsize: plot size.
        bins: number of histogram bins.
        save_path: optional path to save the figure.
    Returns:
        pd.DataFrame with joined IoU values and improvement column.
    """
    label_map = {
        "Sentinel1_SAR": "s1",
        "Sentinel2_Optical": "s2",
        "Fusion_SAR_Optical": "fusion",
    }
    dfs = {}
    for csv_path in csv_paths:
        df = pd.read_csv(Path(csv_path))
        if data_type_col not in df.columns:
            raise ValueError(f"'{data_type_col}' column not found in {csv_path}")
        data_type = df[data_type_col].iloc[0]
        model_key = label_map.get(data_type)
        if model_key is None:
            raise ValueError(f"Unknown data_type '{data_type}' in {csv_path}")
        dfs[model_key] = df[[sample_id_col, iou_col]].rename(
            columns={iou_col: f"{model_key}_iou"}
        )
    required = {"s1", "s2", "fusion"}
    missing = required - set(dfs.keys())
    if missing:
        raise ValueError(f"Missing CSVs for: {missing}")
    agreement_df = (
        dfs["fusion"]
        .merge(dfs["s1"], on=sample_id_col, how="inner")
        .merge(dfs["s2"], on=sample_id_col, how="inner")
    )
    agreement_df["best_baseline_iou"] = agreement_df[
        ["s1_iou", "s2_iou"]
    ].max(axis=1)
    agreement_df["fusion_minus_best_baseline"] = (
        agreement_df["fusion_iou"] - agreement_df["best_baseline_iou"]
    )
    plt.figure(figsize=figsize)
    plt.hist(
        agreement_df["fusion_minus_best_baseline"],
        bins=bins,
        color="C0",
        edgecolor="black",
        alpha=0.75,
    )
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="No change")
    plt.xlabel("Fusion IoU - best baseline IoU")
    plt.ylabel("Number of samples")
    plt.title("Fusion improvement over best single-modality baseline")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return agreement_df