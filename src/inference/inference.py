import torch
from pathlib import Path

import src.util.io as io
import src.data.dataloader as dataloader
import src.models.models as models

def run_inference(model, dataloader, device="cpu", fusion=False, threshold=0.5):
    model.eval()

    results = []

    for sample in dataloader:
        with torch.no_grad():
            if fusion:
                s1_image = sample["s1_image"].to(device)
                s2_image = sample["s2_image"].to(device)
                logits = model(s1_image, s2_image)
                ref_path = sample["s2_image_path"][0]
            else:
                image = sample["image"].to(device)
                logits = model(image)
                ref_path = sample["image_path"][0]

            prob = torch.sigmoid(logits)
            pred = (prob >= threshold).to(torch.uint8)

        results.append({
            "id": sample["id"].item() if torch.is_tensor(sample["id"]) else sample["id"][0],
            "sample_id": sample["sample_id"][0],
            "prob": prob.squeeze().cpu(),
            "pred": pred.squeeze().cpu(),
            "reference_path": ref_path,
            "mask_path": sample["mask_path"][0],
        })

    return results

def inference(model_dir, samples, export):
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

    inference_loader = dataloader.make_inference_dataloader(cfg, samples)

    results = run_inference(model, inference_loader, device=cfg.DEVICE, fusion=True if cfg.DATA_TYPE=="Fusion" else False)
    
    if export:
        io.export_prediction_tifs(results)

    return results

def select_model_for_inference(cfg, samples, export=False):
    io.select_model(
        cfg,
        on_select=lambda model_dir: inference(model_dir, samples, export),
        button_text="Run Inference"
    )