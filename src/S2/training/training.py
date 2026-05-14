import sys
import subprocess
from pathlib import Path

import torch
import torch.optim as optim
from tqdm.auto import tqdm

import src.data.dataloader as DataLoader
import src.losses.losses as losses
import src.S2.models.models as models

from src.util.metrics import metrics_from_logits
import src.util.io as io

def run_epoch(model, dataloader, loss_fn, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    epoch_loss = 0.0
    all_metrics = []

    progress_bar = tqdm(
        dataloader,
        desc="Train" if is_train else "Val",
        leave=False
    )

    for batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            if is_train:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item() * images.size(0)

        if not is_train:
            batch_metrics = metrics_from_logits(outputs, masks)
            all_metrics.append(batch_metrics)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_loss = epoch_loss / len(dataloader.dataset)

    if is_train:
        return avg_loss
    
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    return avg_loss, avg_metrics

def train_model(cfg):

    exp_dir = io.experiment_dir(cfg)
    print(f"Experiment directory: {exp_dir}")
    io.save_config(cfg, exp_dir / "config.json")

    model = models.get_model(cfg).to(cfg.DEVICE)

    train_loader, val_loader, _ = DataLoader.make_s2_dataloaders(cfg)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    loss_fn = losses.bce_dice

    checkpoint_path = exp_dir / "checkpoint.pth"

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    start_epoch = 0

    # Resume if checkpoint exists
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)

        checkpoint_epoch = checkpoint["epoch"]

        if checkpoint_epoch < cfg.EPOCHS:
            print(f"Resuming from epoch {checkpoint_epoch + 1}")

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            history = checkpoint["history"]
            best_val_loss = checkpoint["best_val_loss"]
            best_epoch = checkpoint["best_epoch"]
            best_metrics = checkpoint["best_metrics"]

            start_epoch = checkpoint_epoch + 1

        else:
            print("Checkpoint already reached requested number of epochs.")
            return

    for epoch in range(start_epoch, cfg.EPOCHS):
        train_loss = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device=cfg.DEVICE
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer=None,
            device=cfg.DEVICE
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val IoU: {val_metrics['iou']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_metrics = val_metrics
            io.save_model(model, exp_dir / "best_model.pth")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics
        })

        # Save after every epoch
        io.save_model(model, exp_dir / "last_model.pth")
        io.save_history_csv(history, exp_dir / "metrics.csv")

        io.save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            best_val_loss,
            best_epoch,
            best_metrics
        )

    io.save_summary({
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **best_metrics,
    }, exp_dir / "summary.json")

    if checkpoint_path.exists():
        print("Training complete. Removing checkpoint.")
        checkpoint_path.unlink()

    return model, history

def train_from_file(cfg):
    cfg.NUM_WORKERS = 2

    cfg_path = cfg.ROOT / "tmp_config.pkl"
    io.save_config_pickle(cfg, cfg_path)

    cmd = [
        sys.executable,
        "-u",  # unbuffered output
        str(cfg.ROOT / "src/S2/training/train.py"),
        "--path",
        str(cfg_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training script failed with return code {process.returncode}"
        )

    return process