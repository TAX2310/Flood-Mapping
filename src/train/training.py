import torch
import torch.optim as optim
from tqdm.auto import tqdm
import subprocess
import sys

import src.data.dataloader as DataLoader
import src.losses.losses as losses
import src.models.models as models

from src.util.metrics import metrics_from_logits
import src.util.io as io

def is_best(val_metrics, best_metrics, eps=1e-6):
    if not best_metrics:
        return True

    current_iou = val_metrics["iou"]
    best_iou = best_metrics["iou"]

    current_f1 = val_metrics["f1"]
    best_f1 = best_metrics["f1"]

    if current_iou > best_iou + eps:
        return True

    if abs(current_iou - best_iou) <= eps and current_f1 > best_f1 + eps:
        return True

    return False

def run_epoch(model, dataloader, loss_fn, optimizer=None, device="cpu", fusion=False):
    is_train = optimizer is not None

    print("")
    print("#"*20)
    if is_train:
        print("Running training epoch...")
    else:        
        print("Running validation epoch...")
    print("#"*20)
    print("")

    model.train() if is_train else model.eval()

    epoch_loss = 0.0
    all_metrics = []

    progress_bar = tqdm(
        dataloader,
        desc="Train" if is_train else "Val",
        leave=False
    )

    for batch in progress_bar:
        if fusion == True:
            s1_images = batch["s1_image"].to(device)
            s2_images = batch["s2_image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = s1_images.size(0)
        else:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad()

            if fusion == True:
                outputs = model(s1_images, s2_images)   
            else:
                outputs = model(images)
            loss = loss_fn(outputs, masks)

            if is_train:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item() * batch_size

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
    
    print("")
    print("#"*100)
    print("Starting training with configuration:")
    for key, value in vars(cfg).items():
        print(f"{key}: {value}")
    print("")
    print(f"Experiment directory: {exp_dir}")
    print("#"*100)
    print("")

    if (exp_dir / "summary.json").exists():
        print("Experiment already completed. Skipping training.")
        return 
    io.save_config(cfg, exp_dir / "config.json")
    io.save_config_pickle(cfg, exp_dir / "config.pkl")

    model = models.get_model(cfg).to(cfg.DEVICE)

    if cfg.DATA_TYPE == "Sentinel1_SAR":
        train_loader, val_loader, _ = DataLoader.make_s1_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        train_loader, val_loader, _ = DataLoader.make_s2_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        train_loader, val_loader, _ = DataLoader.make_fusion_dataloaders(cfg)
    else:
        raise ValueError(f"Unsupported DATA_TYPE: {cfg.DATA_TYPE}")

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
    best_epoch = 0
    best_metrics = {}

    patience = cfg.PATIENCE
    epochs_without_improvement = 0
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
            epochs_without_improvement = checkpoint["epochs_without_improvement"]

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
            device=cfg.DEVICE,
            fusion=True if cfg.DATA_TYPE == "Fusion_SAR_Optical" else False
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer=None,
            device=cfg.DEVICE,
            fusion=True if cfg.DATA_TYPE == "Fusion_SAR_Optical" else False
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val IoU: {val_metrics['iou']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}"
        )


        if is_best(val_metrics, best_metrics):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            io.save_model(model, exp_dir / "best_model.pth")
            print(f"New best model saved at epoch {best_epoch} with IoU: {best_metrics['iou']:.4f}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"No IoU improvement for {patience} epochs."
            )
            break

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
            best_metrics,
            epochs_without_improvement
        )

    io.save_summary({
        "title": f"{cfg.LR:.0e} LR, {cfg.BATCH_SIZE} BS, {cfg.WEIGHT_DECAY:.0e} WD, {cfg.DROPOUT_RATE:.1f} DR",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **best_metrics,
    }, exp_dir / "summary.json")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("")
    print("#"*20)
    print("Training complete. Removing checkpoint.")
    print("#"*20)
    print("")

    return model, history

def train_from_file(cfg, num_workers=2):
    cfg.NUM_WORKERS = num_workers

    cfg_path = cfg.ROOT / "tmp_config.pkl"
    io.save_config_pickle(cfg, cfg_path)

    cmd = [
        sys.executable,
        "-u",  # unbuffered output
        str(cfg.ROOT / "src/train/train_from_file.py"),
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