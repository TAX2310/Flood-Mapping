from src.config import CFG

from dataclasses import dataclass
from pathlib import Path

@dataclass
class S2_CFG(CFG):

    DATA_TYPE: str = "Sentinel2_Optical"

    # input setup
    IMAGE_SIZE: int = 128
    CHANNELS: int = 9

    # Optical dataloader
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
    MODEL: str = "unet_optical"
    DROPOUT_RATE: float = 0.0

    # optimizer
    WEIGHT_DECAY: float = 0.0