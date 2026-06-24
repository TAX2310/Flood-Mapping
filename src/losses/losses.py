from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn

# Define the loss functions
bce = nn.BCEWithLogitsLoss()
dice = DiceLoss(mode="binary")

def bce_dice(pred, target):
    # Combine Binary Cross-Entropy (BCE) loss and Dice loss.
    return 0.5 * bce(pred, target) + 0.5 * dice(pred, target)