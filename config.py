from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # base path
    ROOT: Path = Path("./")

    # download
    ZIP_URL: str = "https://zenodo.org/records/12748983/files/Dataset.zip?download=1"

    # split
    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    RANDOM_SEED: int = 42
    N_TRAIN_EVENTS: int = 37
    N_VAL_EVENTS: int = 5
    N_TEST_EVENTS: int = 5

    # input setup
    IMAGE_SIZE: int = 128
    INPUT_CHANNELS: int = 2
    ADD_RATIO_CHANNEL: bool = False

    # STURM normalisation
    S1_MIN_DB: float = -30.0
    S1_MAX_DB: float = 10.0

    # masks
    BINARY_MASK: bool = True
    WATER_CLASSES: tuple = (1, 2, 3, 4, 5)
    IGNORE_CLASSES: tuple = (99,)

    # dataloader
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    SHUFFLE_TRAIN: bool = True

    @property
    def SAR_PATH(self) -> Path:
        return self.ROOT / "SAR"

    @property
    def OPTICAL_PATH(self) -> Path:
        return self.ROOT / "Optical"

    @property
    def FUSION_PATH(self) -> Path:
        return self.ROOT / "Fusion"

    @property
    def DATA_PATH(self) -> Path:
        return self.ROOT / "Dataset"

    @property
    def ZIP_PATH(self) -> Path:
        return self.ROOT / "Dataset.zip"

    @property
    def S1_PATH(self) -> Path:
        return self.DATA_PATH / "Sentinel1"

    @property
    def S1_IMAGE_PATH(self) -> Path:
        return self.S1_PATH / "S1"

    @property
    def S1_MASK_PATH(self) -> Path:
        return self.S1_PATH / "Floodmaps"

    # debug
    LIMIT_SAMPLES: int | None = None