from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
    config_path = Path(model_dir) / "config.json"

    cfg = io.load_config_json(config_path)

    title = f'Train Val Loss - lr_{cfg.LR}__bs_{cfg.BATCH_SIZE}__e_{cfg.EPOCHS}'

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

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_precision_recall(model_dir, save_path=None):
    """
    Read a metrics CSV and plot precision vs recall over epochs.
    """

    csv_path = Path(model_dir) / "metrics.csv"
    config_path = Path(model_dir) / "config.json"

    cfg = io.load_config_json(config_path)

    title = f'Precision Recall - lr_{cfg.LR}__bs_{cfg.BATCH_SIZE}__e_{cfg.EPOCHS}'

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
    config_path = Path(model_dir) / "config.json"

    cfg = io.load_config_json(config_path)

    title = f'IoU F1 - lr_{cfg.LR}__bs_{cfg.BATCH_SIZE}__e_{cfg.EPOCHS}'

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