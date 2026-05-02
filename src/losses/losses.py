from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss(mode="binary")

def bce_dice(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice(pred, target)