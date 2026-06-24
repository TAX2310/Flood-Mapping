import torch
from pathlib import Path
import matplotlib.pyplot as plt

import src.util.io as io
import src.util.metrics as metrics 
import src.data.dataloader as dataloader
import src.models.models as models

def run_inference(model, dataloader, device="cpu", fusion=False, threshold=0.5):
    """
    Run inference on a given model and dataloader.
    """
    # Set the model to evaluation mode
    model.eval()

    results = []

    # Iterate over the dataloader and perform inference
    for sample in dataloader:
        with torch.no_grad():
            # Perform forward pass through the model
            if fusion:
                s1_image = sample["s1_image"].to(device)
                s2_image = sample["s2_image"].to(device)
                mask = sample["mask"].to(device)
                logits = model(s1_image, s2_image)
                ref_path = sample["s2_image_path"][0]
            else:
                image = sample["image"].to(device)
                mask = sample["mask"].to(device)
                logits = model(image)
                ref_path = sample["image_path"][0]

            # Compute probabilities and predictions
            prob = torch.sigmoid(logits)
            pred = (prob >= threshold).to(torch.uint8)
            # Compute metrics based on the logits and mask
            metric = metrics.metrics_from_logits(logits, mask, threshold=threshold)

        # Append the results for this sample to the results list
        if fusion:
            results.append({
                "id": sample["id"],
                "sample_id": sample["sample_id"][0],
                "s1_image": s1_image.cpu(),
                "s2_image": s2_image.cpu(),
                "mask": mask.cpu(),
                "prob": prob.squeeze().cpu(),
                "pred": pred.squeeze().cpu(),
                "reference_path": ref_path,
                "mask_path": sample["mask_path"][0],
                **metric,
            })
        else:
            results.append({
                "id": sample["id"],
                "sample_id": sample["sample_id"][0],
                "image": image.cpu(),
                "mask": mask.cpu(),
                "prob": prob.squeeze().cpu(),
                "pred": pred.squeeze().cpu(),
                "reference_path": ref_path,
                "mask_path": sample["mask_path"][0],
                **metric,
            })

    return results

def inference(cfg, model_dir, samples=None, export=False):
    """
    Run inference on a trained model using the provided configuration and model directory.
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

    # Set the device for inference (GPU if available, otherwise CPU)
    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load the model and its weights from the checkpoint
    model = models.get_model(cfg).to(cfg.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint)

    # Create the inference dataloader using the provided samples
    inference_loader = dataloader.make_inference_dataloader(cfg, samples)

    # Run inference using the model and dataloader
    results = run_inference(model, 
                            inference_loader, 
                            device=cfg.DEVICE, 
                            fusion=True if cfg.DATA_TYPE=="Fusion_SAR_Optical" else False, 
                            threshold=cfg.THRESHOLD)
    
    # Export the prediction results as TIFF files if the export flag is set
    if export:
        io.export_prediction_tifs(results, cfg.EXPORT_DIR)

    return results








