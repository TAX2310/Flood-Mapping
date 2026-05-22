import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

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