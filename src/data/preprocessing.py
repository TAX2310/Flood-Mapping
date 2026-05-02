import torch

def remap_mask_to_binary(mask, water_classes=(1,2,3,4,5), ignore_classes=(99,)):
    """
    mask: torch.Tensor [H,W] or [B,H,W]
    """
    out = torch.zeros_like(mask, dtype=torch.long)

    water_classes = torch.tensor(water_classes, device=mask.device)
    ignore_classes = torch.tensor(ignore_classes, device=mask.device)

    out[torch.isin(mask, water_classes)] = 1
    out[torch.isin(mask, ignore_classes)] = 255

    return out