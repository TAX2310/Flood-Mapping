from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # base path
    ROOT: Path = Path("./")

    # download
    ZIP_URL: str = "https://zenodo.org/records/12748983/files/Dataset.zip?download=1"

    # Random Seed 
    RANDOM_SEED: int = 42

    # split
    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10

    N_TRAIN_EVENTS: int = 37
    N_VAL_EVENTS: int = 5
    N_TEST_EVENTS: int = 5

    # masks
    BINARY_MASK: bool = True
    WATER_CLASSES: tuple = (1, 2, 3, 4, 5)
    IGNORE_CLASSES: tuple = (99,)

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
