from xml.parsers.expat import model
from pathlib import Path

import torch
from tqdm.auto import tqdm
import subprocess
import sys

import src.data.dataloader as DataLoader
import src.losses.losses as losses
import src.models.models as models

import src.util.metrics as metrics 
import src.util.io as io

def run_epoch(model, dataloader, loss_fn, device="cpu", fusion=False, threshold=0.5):
    """
    Run a single epoch of testing.
    """
    # Set the model to evaluation mode
    model.eval()

    epoch_loss = 0.0
    all_metrics = []

    # Use tqdm for progress bar
    progress_bar = tqdm(
        dataloader,
        desc="Test",
        leave=False
    )

    # Iterate over the dataloader and perform inference
    for batch in progress_bar:
        with torch.no_grad():
            # Perform forward pass through the model
            if fusion == True:
                s1_images = batch["s1_image"].to(device)
                s2_images = batch["s2_image"].to(device)
                masks = batch["mask"].to(device)
                batch_size = s1_images.size(0)
                outputs = model(s1_images, s2_images)
            else:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                batch_size = images.size(0)
                outputs = model(images)

            # Compute the loss for the batch
            loss = loss_fn(outputs, masks)

            # Update the epoch loss by adding the batch loss multiplied by the batch size
            epoch_loss += loss.item() * batch_size

            # Compute metrics based on the outputs and masks
            batch_metrics = metrics.metrics_from_logits(outputs, masks, threshold=threshold)
            
            # Append the batch metrics to the list of all metrics
            all_metrics.append(batch_metrics)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    # Calculate the average loss and metrics for the entire epoch
    avg_loss = epoch_loss / len(dataloader.dataset)
    
    # Calculate the average metrics by averaging each metric across all batches
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    return avg_loss, avg_metrics

def test_model(cfg, model_dir):
    """
    Test a trained model using the provided configuration and model directory.
    """
    model_dir = Path(model_dir)
    print(f"Model directory: {model_dir}")

    checkpoint_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.pkl"

    if not checkpoint_path.exists() or not config_path.exists():
        print("Model checkpoint or config not found.")
        return

    # Load the configuration from the pickle file
    cfg = io.load_config_pickle(config_path)

    # Set the device for testing (GPU if available, otherwise CPU)
    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load the model and its weights from the checkpoint
    model = models.get_model(cfg).to(cfg.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint)

    # Create the test dataloader based on the data type specified in the configuration
    if cfg.DATA_TYPE == "Sentinel1_SAR":
        _, _, test_loader = DataLoader.make_s1_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        _, _, test_loader = DataLoader.make_s2_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        _, _, test_loader = DataLoader.make_fusion_dataloaders(cfg)

    # Define the loss function to be used for testing
    loss_fn = losses.bce_dice

    # Run a single epoch of testing and compute the test loss and metrics
    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        loss_fn,
        device=cfg.DEVICE,
        fusion=True if cfg.DATA_TYPE == "Fusion_SAR_Optical" else False,
        threshold=cfg.THRESHOLD
    )

    print(f"Test Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save the test loss and metrics to a summary JSON file in the model directory
    test_summary = {
        "test_loss": float(test_loss),
        **{
            f"test_{key}": float(value)
            for key, value in test_metrics.items()
        }
    }
    io.update_summary(
        model_dir / "summary.json",
        test_summary,
        section="test_metrics"
    )

def select_model_to_test(cfg):
    """
    Select a trained model to test using a GUI file dialog.
    """
    io.select_model(
        cfg,
        on_select=lambda model_dir: test_model(cfg, model_dir),
        button_text="Test Model"
    )