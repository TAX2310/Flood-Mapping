import torch

def remap_mask_to_binary(mask, water_classes=(1,2,3,4,5), ignore_classes=(99,)):
    """
    Remap a mask to binary classes: water (1) and non-water (0), with an optional ignore class (255).
    """
    out = torch.zeros_like(mask, dtype=torch.long)

    water_classes = torch.tensor(water_classes, device=mask.device)
    ignore_classes = torch.tensor(ignore_classes, device=mask.device)

    out[torch.isin(mask, water_classes)] = 1
    out[torch.isin(mask, ignore_classes)] = 255

    return out