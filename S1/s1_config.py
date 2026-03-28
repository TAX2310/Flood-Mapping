from config import CFG

from dataclasses import dataclass
from pathlib import Path

@dataclass
class S1_CFG(CFG):

    # input setup
    IMAGE_SIZE: int = 128
    INPUT_CHANNELS: int = 2
    ADD_RATIO_CHANNEL: bool = False

    # STURM SAR normalisation
    S1_MIN_DB: float = -30.0
    S1_MAX_DB: float = 10.0

    # SAR dataloader
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    SHUFFLE_TRAIN: bool = True
