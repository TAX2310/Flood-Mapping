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
    TRAIN_EVENTS = {"EMSR470", "EMSR467", "EMSR416", "EMSR438", "EMSR407"}
    VAL_EVENTS   = {"EMSR292", "EMSR479", "EMSR496"}
    TEST_EVENTS  = {"EMSR419", "EMSR629", "EMSR501", "EMSR279"}

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
    
    @property
    def S1_METADATA_CSV(self) -> Path:
        return self.DATA_PATH / "Sentinel1_metadata.csv"
    
    @property
    def S2_METADATA_CSV(self) -> Path:
        return self.DATA_PATH / "Sentinel2_metadata.csv"
