from src.config import CFG

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Fusion_CFG(CFG):

    DATA_TYPE: str = "Fusion_SAR_Optical"

    # input setup
    IMAGE_SIZE: int = 128
    S1_CHANNELS: int = 2
    S2_CHANNELS: int = 9

    # SAR dataloader
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    SHUFFLE_TRAIN: bool = True

    # data augmentation
    USE_ROTATIONS: bool = True

    # training
    EPOCHS: int = 20
    LR: float = 1e-3

    # model
    MODEL: str = "late_fusion_unet_resnet34"
    ENCODER_WEIGHTS: str = None

    # optimizer
    WEIGHT_DECAY: float = 1e-5