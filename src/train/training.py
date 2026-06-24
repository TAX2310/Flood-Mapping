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
    """
    Determine if the current validation metrics are better than the best metrics so far.
    """
    # If there are no best metrics yet, the current metrics are considered the best.
    if not best_metrics:
        return True

    current_iou = val_metrics["iou"]
    best_iou = best_metrics["iou"]

    current_f1 = val_metrics["f1"]
    best_f1 = best_metrics["f1"]

    # Check if the current IoU is better than the best IoU
    if current_iou > best_iou + eps:
        return True
    
    #  If the IoU is the same (within a small epsilon), check if the F1 score is better
    if abs(current_iou - best_iou) <= eps and current_f1 > best_f1 + eps:
        return True

    return False

def run_epoch(model, dataloader, loss_fn, optimizer=None, device="cpu", fusion=False, threshold=0.5):
    """
    Run a single epoch of training or validation.
    """

    # Determine if this is a training or validation epoch based on the presence of an optimizer.
    is_train = optimizer is not None

    print("")
    print("#"*20)
    if is_train:
        print("Running training epoch...")
    else:        
        print("Running validation epoch...")
    print("#"*20)
    print("")

    # Set the model to training or evaluation mode based on the epoch type.
    model.train() if is_train else model.eval()

    epoch_loss = 0.0
    all_metrics = []

    # Use tqdm to create a progress bar for the dataloader.
    progress_bar = tqdm(
        dataloader,
        desc="Train" if is_train else "Val",
        leave=False
    )

    # Iterate over batches in the dataloader.
    for batch in progress_bar:
        # Move the data to the specified device (CPU or GPU).
        if fusion == True:
            s1_images = batch["s1_image"].to(device)
            s2_images = batch["s2_image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = s1_images.size(0)
        else:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

        # Use torch.set_grad_enabled to enable or disable gradient computation based on the epoch type.
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

        # Update the epoch loss by accumulating the loss for the current batch, scaled by the batch size.
        epoch_loss += loss.item() * batch_size

        # If this is a validation epoch, compute metrics from the model outputs and masks, and store them in all_metrics.
        if not is_train:
            batch_metrics = metrics_from_logits(outputs, masks, threshold=threshold)
            all_metrics.append(batch_metrics)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    # Calculate the average loss for the epoch by dividing the total epoch loss by the number of samples in the dataloader.
    avg_loss = epoch_loss / len(dataloader.dataset)

    # If this is a training epoch, return the average loss. Otherwise, calculate the average metrics across all batches and return both the average loss and average metrics.
    if is_train:
        return avg_loss
    
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    return avg_loss, avg_metrics

def train_model(cfg):
    """
    Train a model based on the provided configuration.
    """
    # Determine the experiment directory based on the configuration.
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

    # Check if the experiment has already been completed by looking for a summary.json file in the experiment directory. If it exists, skip training.
    if (exp_dir / "summary.json").exists():
        print("Experiment already completed. Skipping training.")
        return 
    # Save the configuration to the experiment directory in both JSON and pickle formats for future reference.
    io.save_config(cfg, exp_dir / "config.json")
    io.save_config_pickle(cfg, exp_dir / "config.pkl")

    # Initialize the model based on the configuration and move it to the specified device (CPU or GPU).
    model = models.get_model(cfg).to(cfg.DEVICE)

    # Create data loaders for training and validation based on the specified data type in the configuration.
    if cfg.DATA_TYPE == "Sentinel1_SAR":
        train_loader, val_loader, _ = DataLoader.make_s1_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        train_loader, val_loader, _ = DataLoader.make_s2_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        train_loader, val_loader, _ = DataLoader.make_fusion_dataloaders(cfg)
    else:
        raise ValueError(f"Unsupported DATA_TYPE: {cfg.DATA_TYPE}")

    # Set up the optimizer (AdamW) and learning rate scheduler (ReduceLROnPlateau) based on the configuration.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Set up a learning rate scheduler that reduces the learning rate when a metric has stopped improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # Define the loss function to be used during training and validation. In this case, a combination of binary cross-entropy and Dice loss is used.
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
        
        # If the checkpoint epoch is less than the total number of epochs specified in the configuration, resume training from the next epoch.
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

    # Run training and validation for the specified number of epochs, starting from the last checkpoint if available.
    for epoch in range(start_epoch, cfg.EPOCHS):
        # Run a training epoch and compute the training loss.
        train_loss = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device=cfg.DEVICE,
            fusion=True if cfg.DATA_TYPE == "Fusion_SAR_Optical" else False,
            threshold=cfg.THRESHOLD
        )

        # Run a validation epoch and compute the validation loss and metrics.
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer=None,
            device=cfg.DEVICE,
            fusion=True if cfg.DATA_TYPE == "Fusion_SAR_Optical" else False,
            threshold=cfg.THRESHOLD
        )

        # Update the learning rate scheduler based on the validation loss.
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val IoU: {val_metrics['iou']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Check if the current validation metrics are better than the best metrics so far. If they are, update the best metrics and save the model.
        if is_best(val_metrics, best_metrics):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            io.save_model(model, exp_dir / "best_model.pth")
            print(f"New best model saved at epoch {best_epoch} with IoU: {best_metrics['iou']:.4f}")

        else:
            epochs_without_improvement += 1

        # Implement early stopping if the number of epochs without improvement exceeds the patience threshold.
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"No IoU improvement for {patience} epochs."
            )
            break

        # Append the current epoch's metrics to the training history for later analysis and visualization.
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics
        })

        # Save the current model, training history, and checkpoint to the experiment directory for later use or resumption of training.
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

    # Save the final summary of the training process, including the best epoch, best validation loss, and best metrics, to a JSON file in the experiment directory.
    io.save_summary({
        "title": f"{cfg.LR:.0e} LR, {cfg.BATCH_SIZE} BS, {cfg.WEIGHT_DECAY:.0e} WD, {cfg.DROPOUT_RATE:.1f} DR",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **best_metrics,
    }, exp_dir / "summary.json")

    # Remove the checkpoint file after training is complete to clean up the experiment directory.
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("")
    print("#"*20)
    print("Training complete. Removing checkpoint.")
    print("#"*20)
    print("")

    return model, history

def train_from_file(cfg, num_workers=2):
    """
    train from python file using subprocess to avoid memory issues with Jupyter Notebook. This function saves the configuration to a temporary pickle file and then runs the training script in a separate process.
    """
    # Set the number of workers for data loading in the configuration.
    cfg.NUM_WORKERS = num_workers

    # Save the configuration to a temporary pickle file in the root directory of the project. This allows the training script to access the configuration without requiring it to be passed directly as an argument.
    cfg_path = cfg.ROOT / "tmp_config.pkl"
    io.save_config_pickle(cfg, cfg_path)

    # Construct the command to run the training script in a separate process.
    cmd = [
        sys.executable,
        "-u",  # unbuffered output
        str(cfg.ROOT / "src/train/train_from_file.py"),
        "--path",
        str(cfg_path),
    ]

    # Use subprocess.Popen to run the training script in a separate process. 
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