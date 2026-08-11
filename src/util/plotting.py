from email.mime import image
import json
from pathlib import Path
from pyexpat import model
from time import sleep
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

def plot_controle(fig, save_path=None, show=True, dpi=None):
    """
    Save the figure if save_path is given, then either display if show=True or return it if show=False.
    """
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix.lower() != ".pdf":
            save_path = save_path.with_suffix(".pdf")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            save_path.unlink()
        if dpi is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig

def plot_train_val_loss(model_dir, save_path=None, show=True):
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

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(
        loc="upper right",
        fontsize=16,
        title_fontsize=17,
        handlelength=2,
        handletextpad=0.5,
        labelspacing=0.6,
        borderpad=0.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plot_controle(fig, save_path, show)


def plot_precision_recall(model_dir, save_path=None, show=True):
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

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["precision"], marker="o", label="Precision")
    plt.plot(df["epoch"], df["recall"], marker="o", label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(
        loc="lower right",
        fontsize=16,
        title_fontsize=17,
        handlelength=2,
        handletextpad=0.5,
        labelspacing=0.6,
        borderpad=0.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plot_controle(fig, save_path, show)


def plot_iou_f1(model_dir, save_path=None, show=True):
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

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["iou"], marker="o", label="IoU")
    plt.plot(df["epoch"], df["f1"], marker="o", label="F1 / Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(
        loc="lower right",
        fontsize=16,
        title_fontsize=17,
        handlelength=2,
        handletextpad=0.5,
        labelspacing=0.6,
        borderpad=0.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_metrics(cfg, model_dir, show=True):
    """
    Read a metrics CSV and plot train loss vs validation loss, precision vs recall, and IoU vs F1.
    """
    if cfg.DATA_TYPE == "Sentinel1_SAR":
        model = "s1"
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        model = "s2"
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        model = "fusion"

    figs = [
        plot_train_val_loss(model_dir, save_path=cfg.FIG_EXPORTS_DIR/(model+"_loss.pdf"), show=show),
        plot_precision_recall(model_dir, save_path=cfg.FIG_EXPORTS_DIR/(model+"_prec_rec.pdf"), show=show),
        plot_iou_f1(model_dir, save_path=cfg.FIG_EXPORTS_DIR/(model+"_iou_f1.pdf"), show=show),
    ]

    if not show:
        return figs

def view_training_metrics(cfg):
    io.select_model(
        cfg,
        on_select=lambda model_dir: plot_metrics(cfg, model_dir),
        button_text="Plot metrics"
    )

def plot_hp_comparison_bar(cfg, figsize=(12, 5), save_path=None, show=True):
    """
    Plot a bar chart comparing F1 and IoU scores across model subdirectories.
    """
    root_dir = cfg.EXP_DIR / cfg.DATASET / cfg.DATA_TYPE
    title = f"{root_dir.name} Training Performance"
    subdirs = io.get_leaf_subdirs(root_dir)
    models, f1_scores, iou_scores = [], [], []

    for subdir in subdirs:
        summary = subdir / "summary.json"
        if summary.exists():
            with open(summary, "r") as f:
                s = json.load(f)
                models.append(s["title"])
                f1_scores.append(s["f1"])
                iou_scores.append(s["iou"])
    x = np.arange(len(models))
    width = 0.35
    fig = plt.figure(figsize=figsize)

    bars_iou = plt.bar(x - width / 2, iou_scores, width, label="IoU", color="C0")
    bars_f1 = plt.bar(x + width / 2, f1_scores, width, label="F1", color="C1")
    max_iou = max(iou_scores)
    max_f1 = max(f1_scores)

    for bars, max_val in [(bars_iou, max_iou), (bars_f1, max_f1)]:
        for bar in bars:
            height = bar.get_height()
            is_max = height == max_val
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="red" if is_max else "black",
                fontweight="bold" if is_max else "normal",
            )

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend(
        loc="lower right",
        fontsize=16,
        title_fontsize=17,
        handlelength=2,
        handletextpad=0.5,
        labelspacing=0.6,
        borderpad=0.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_mask_tif(tif_path, title="Mask", band=1, figsize=(7, 7), save_path=None, show=True):
    """
    Read a mask GeoTIFF and display it.
    """
    with rasterio.open(tif_path) as src:
        mask = src.read(band)

    fig = plt.figure(figsize=figsize)
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_mask_tensor(mask, title="Mask", figsize=(7, 7), save_path=None, show=True):
    """
    Display a mask tensor as a grayscale image.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    fig = plt.figure(figsize=figsize)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_fp_fn_mask_tensor(
    mask,
    pred,
    title="FP/FN Mask",
    figsize=(7, 7),
    show_legend=False,
    save_path=None,
    show=True,
):
    """
    Display a mask tensor and predicted mask tensor as a combined FP/FN mask.
    """
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
    fig = plt.figure(figsize=figsize)
    plt.imshow(fp_fn_mask, cmap=cmap, norm=norm)

    if show_legend:
        legend_items = [
            mpatches.Patch(color="black", label="No flood"),
            mpatches.Patch(color="blue", label="FP"),
            mpatches.Patch(color="red", label="FN"),
            mpatches.Patch(color="white", label="Flood"),
        ]
        plt.legend(
            handles=legend_items,
            title="Comparison",
            loc="lower right",
            fontsize=16,
            title_fontsize=17,
            handlelength=2,
            handletextpad=0.5,
            labelspacing=0.6,
            borderpad=0.8,
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show, dpi=300)

def plot_prob_tensor(prob, title="Probability", figsize=(7, 7), save_path=None, show=True):
    """
    Display a probability tensor as a grayscale image.
    """
    if isinstance(prob, torch.Tensor):
        prob = prob.detach().cpu().numpy()

    prob = np.squeeze(prob)

    fig = plt.figure(figsize=figsize)
    plt.imshow(prob, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show, dpi=300)

def plot_s1_tif(tif_path, title="S1_sar_image", figsize=(7, 7), save_path=None, show=True):
    """
    Plot S1 GeoTIFF as a single normalised RGB image.
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

    fig = plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show, dpi=300)

def plot_s1_tensor(tensor, title="S1 SAR Image", figsize=(7, 7), save_path=None, show=True):
    """
    Plot S1 tensor as a single normalised RGB image.
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

    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show, dpi=300)

def plot_s2_tif(tif_path, title="S2 Optical Image", figsize=(7, 7), save_path=None, show=True):
    """
    Plot Sentinel-2 GeoTIFF as a normalised RGB image.
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

    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_s2_tensor(tensor, title="S2 Optical Image", figsize=(7, 7), save_path=None, show=True):
    """
    Plot Sentinel-2 tensor as a normalised RGB image.
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

    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_sample_results(cfg, results, figsize=(7,7), show=True):
    """
    Plot sample results including image, ground truth mask, predicted mask, FP/FN comparison, and probability map.
    """
    if cfg.DATA_TYPE == "Sentinel1_SAR":
        model = "s1"
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        model = "s2"
    else:
        model = "fusion"

    figs = []

    for sample in results:
        for key, value in sample.items():
            print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

        sample_id = Path(sample["sample_id"]).stem

        if model == "fusion":
            s1_image = sample["s1_image"].cpu()
            s2_image = sample["s2_image"].cpu()
        else:
            image = sample["image"].cpu()
        mask = sample["mask"].cpu()
        pred = sample["pred"].cpu()
        prob = sample["prob"].cpu()

        sleep(1)
        if model == "s1":
            figs.append(plot_s1_tensor(image, figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR/(f"{sample_id}_s1.pdf"), show=show))
        elif model == "s2":
            figs.append(plot_s2_tensor(image, figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR / f"{sample_id}_s2.pdf", show=show))
        else:
            figs.append(plot_s1_tensor(s1_image, title="S1 SAR Image", figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR / f"{sample_id}_s1.pdf", show=show))
            sleep(1)
            figs.append(plot_s2_tensor(s2_image, title="S2 Optical Image", figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR / f"{sample_id}_s2.pdf", show=show))

        sleep(1)
        figs.append(plot_mask_tensor(mask, title="Ground Truth", figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR/(f"{sample_id}_gt.pdf"), show=show))
        sleep(1)
        figs.append(plot_mask_tensor(pred, title="Predicted Mask", figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR/(f"{sample_id}_{model}_pred.pdf"), show=show))
        sleep(1)
        figs.append(plot_fp_fn_mask_tensor(mask, pred, figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR/(f"{sample_id}_{model}_fnfp.pdf"), show=show))
        sleep(1)
        figs.append(plot_prob_tensor(prob, figsize=figsize, save_path=cfg.FIG_EXPORTS_DIR/(f"{sample_id}_{model}_prob.pdf"), show=show))

    if not show:
        return figs

def plot_metric_distribution_from_csv(
    results_csv,
    metric="iou",
    bins=20,
    title=None,
    figsize=(10, 6),
    show_stats=True,
    save_path=None,
    show=True,
):
    """
    Plot the distribution of a specified metric from a CSV file.
    """
    results_csv = Path(results_csv)
    df = pd.read_csv(results_csv)
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    fig = plt.figure(figsize=figsize)
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
        plt.axvline(
            mean_value - std_value,
            color="purple",
            linestyle="-.",
            linewidth=2,
            label=f"±1 SD = {std_value:.3f}",
        )
        plt.axvline(
            mean_value + std_value,
            color="purple",
            linestyle="-.",
            linewidth=2,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(labels),
            frameon=False,
            fontsize=15,
            handlelength=1.2,
            handletextpad=0.2,
            columnspacing=0.5,
        )
    plt.xlabel(metric.upper())
    plt.ylabel("Number of samples")

    if title is None:
        title = f"Distribution of {metric.upper()} scores"

    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    return plot_controle(fig, save_path, show)

def plot_iou_vs_flood_scatter(
    csv_paths,
    iou_col="iou",
    flood_pixels_col="flood_pixels",
    total_pixels_col="total_valid_pixels",
    data_type_col="data_type",
    figsize=(8, 6),
    title="Flood coverage vs IoU",
    show_legend=True,
    save_path=None,
    show=True,
):
    """
    Plot IoU against flood percentage for multiple model result CSVs.
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
    fig = plt.figure(figsize=figsize)
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

    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles,
            labels,
            title="Model",
            loc="lower right",
            fontsize=13,
            title_fontsize=13,
            markerscale=1.5,
            handlelength=1.5,
            handletextpad=0.3,
            labelspacing=0.3,
            borderpad=0.5,
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
        )
        plt.tight_layout()
    else:
        plt.tight_layout()

    return plot_controle(fig, save_path, show)

def plot_iou_vs_flood_median(
    csv_paths,
    iou_col="iou",
    flood_pixels_col="flood_pixels",
    total_pixels_col="total_valid_pixels",
    data_type_col="data_type",
    bin_width=10,          # coverage bin width in percentage points
    min_samples=1,         # suppress a model's bin if it has fewer than this many tiles
    show_scatter=True,     # faint raw points behind the lines
    scatter_alpha=0.15,
    band_alpha=0.20,       # IQR shading opacity
    figsize=(8, 6),
    title="Flood coverage vs IoU",
    show_legend=True,
    save_path=None,
    show=True,
):
    """
    Plot median IoU against flood coverage for multiple model result CSVs,
    with an interquartile-range (Q1-Q3) shaded band per model. Coverage is
    binned in fixed-width intervals; the median and quartiles are computed
    within each bin.
    """
    label_map = {
        "Sentinel1_SAR": "S1",
        "Sentinel2_Optical": "S2",
        "Fusion_SAR_Optical": "Fusion",
    }
    marker_map = {"S1": "o", "S2": "^", "Fusion": "*"}
    color_map = {"S1": "blue", "S2": "orange", "Fusion": "green"}

    # Fixed bin edges across the full 0-100% coverage range so all models
    # share the same x-axis bins and remain directly comparable.
    bin_edges = np.arange(0, 100 + bin_width, bin_width)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = plt.figure(figsize=figsize)

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

        # Optional faint scatter for context (shows the raw spread and the
        # cluster of zeros at low coverage).
        if show_scatter:
            plt.scatter(
                df["flood_percentage"],
                df[iou_col],
                marker=marker,
                color=color,
                alpha=scatter_alpha,
                s=20,
                linewidths=0,
                zorder=1,
            )

        # Assign each tile to a coverage bin, then aggregate IoU per bin.
        df["bin"] = pd.cut(
            df["flood_percentage"],
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )
        grouped = df.groupby("bin")[iou_col]
        stats = grouped.agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            n="count",
        )

        # Drop sparse bins so the line isn't dragged around by 1-2 tiles.
        stats = stats[stats["n"] >= min_samples]
        if stats.empty:
            continue

        x = bin_centres[stats.index.to_numpy()]
        med = stats["median"].to_numpy()
        q1 = stats["q1"].to_numpy()
        q3 = stats["q3"].to_numpy()

        # IQR band, then the median line on top.
        plt.fill_between(x, q1, q3, color=color, alpha=band_alpha, zorder=2)
        plt.plot(
            x, med,
            label=label,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            zorder=3,
        )

    plt.xlabel("Flood coverage (%)")
    plt.ylabel("IoU")
    plt.title(title)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles,
            labels,
            title="Model",
            loc="lower right",
            fontsize=13,
            title_fontsize=13,
            markerscale=1.5,
            handlelength=1.5,
            handletextpad=0.3,
            labelspacing=0.3,
            borderpad=0.5,
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
        )

    plt.tight_layout()
    return plot_controle(fig, save_path, show)

def plot_average_iou_per_event(
    results_csv,
    event_col="ems_code",
    iou_col="iou",
    figsize=(12, 5),
    title="Average IoU per EMSR event",
    sort=False,
    save_path=None,
    show=True,
):
    """
    Plot average IoU per EMSR event from results CSV.
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
    fig = plt.figure(figsize=figsize)
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

    return plot_controle(fig, save_path, show)

def plot_fusion_improvement_distribution(
    csv_paths,
    sample_id_col="tile_id",
    data_type_col="data_type",
    iou_col="iou",
    figsize=(8, 5),
    bins=30,
    save_path=None,
    show=True,
):
    """
    Plot Fusion IoU improvement over the best single-modality baseline using result CSV files.
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

    fig = plt.figure(figsize=figsize)
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

    return plot_controle(fig, save_path, show)
