from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # base path
    ROOT: Path = Path("./")

    # Dataset
    DATASET: str = "STURM-fusion-24"

    # download
    ZIP_URL: str = "https://huggingface.co/datasets/tax2310/STURM-fusion-24/resolve/main/Dataset.zip"

    # Random Seed 
    RANDOM_SEED: int = 42

    # split
    TRAIN_EVENTS = {"EMSR470", "EMSR479", "EMSR419", "EMSR570", "EMSR492", "EMSR279", "EMSR629"}
    VAL_EVENTS   = {"EMSR554", "EMSR424", "EMSR441"}
    TEST_EVENTS  = {"EMSR292", "EMSR407", "EMSR416", "EMSR422", "EMSR465", "EMSR567"}

    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.10 
    TEST_SPLIT: float = 0.10

    SPLIT_METHOD: str = "random" #'by_event' or 'random'

    # masks
    BINARY_MASK: bool = True
    WATER_CLASSES: tuple = (1, 2, 3, 4, 5)
    IGNORE_CLASSES: tuple = (99,)

    @property
    def DATA_PATH(self) -> Path:
        return self.ROOT / "Dataset"

    @property
    def ZIP_PATH(self) -> Path:
        return self.ROOT / "Dataset.zip"

    @property
    def S1_PATH(self) -> Path:
        return self.DATA_PATH / "S1"

    @property
    def S2_PATH(self) -> Path:
        return self.DATA_PATH / "S2"
    
    @property
    def FUSION_PATH(self) -> Path:
        return self.ROOT / "Fusion"

    @property
    def MASK_PATH(self) -> Path:
        return self.DATA_PATH / "floodmaps"
    
    @property
    def METADATA_PATH(self) -> Path:
        return self.DATA_PATH / "metadata"
    
    @property
    def METADATA_CSV(self) -> Path:
        return self.METADATA_PATH / "metadata.csv"
    
    @property
    def EXP_DIR(self) -> Path:
        return self.ROOT / "experiments"