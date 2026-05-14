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

def run_epoch(model, dataloader, loss_fn, device="cpu", fusion=False):

    model.eval()

    epoch_loss = 0.0
    all_metrics = []

    progress_bar = tqdm(
        dataloader,
        desc="Test",
        leave=False
    )

    for batch in progress_bar:
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

        loss = loss_fn(outputs, masks)

        epoch_loss += loss.item() * batch_size

        batch_metrics = metrics.metrics_from_logits(outputs, masks)
        all_metrics.append(batch_metrics)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_loss = epoch_loss / len(dataloader.dataset)
    
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    return avg_loss, avg_metrics

def test_model(model_dir):

    model_dir = Path(model_dir)
    print(f"Model directory: {model_dir}")

    checkpoint_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.pkl"

    if not checkpoint_path.exists() or not config_path.exists():
        print("Model checkpoint or config not found.")
        return

    cfg = io.load_config_pickle(config_path)
        
    model = models.get_model(cfg).to(cfg.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint)

    if cfg.DATA_TYPE == "Sentinel1_SAR":
        _, _, test_loader = DataLoader.make_s1_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        _, _, test_loader = DataLoader.make_s2_dataloaders(cfg)
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        _, _, test_loader = DataLoader.make_fusion_dataloaders(cfg)

    loss_fn = losses.bce_dice

    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        loss_fn,
        device=cfg.DEVICE,
        fusion=True if cfg.DATA_TYPE == "Fusion_SAR_Optical" else False
    )

    print(f"Test Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")