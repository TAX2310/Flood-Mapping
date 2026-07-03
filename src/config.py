from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # base path
    ROOT: Path = Path("./")

    # Dataset
    DATASET: str = "STURM-fusion-24"

    # download
    DATASET_ZIP_URL: str = "https://huggingface.co/datasets/tax2310/STURM-fusion-24/resolve/main/Dataset.zip"
    RESULTS_ZIP_URL: str = "https://huggingface.co/datasets/tax2310/STURM-fusion-24/resolve/main/Results.zip"

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

    # Training
    EPOCHS: int = 50
    PATIENCE: int = 8
    THRESHOLD: float = 0.5

    # hyperparameter search space
    LEARNING_RATES: tuple = (1e-3, 1e-4)
    BATCH_SIZES: tuple = (32, 64)
    WEIGHT_DECAYS: tuple = (0.0, 1e-5)
    DROPOUT_RATES: tuple = (0.0, 0.2)

    # data augmentation
    USE_ROTATIONS: bool = False

    @property
    def DATA_PATH(self) -> Path:
        return self.ROOT / "Dataset"

    @property
    def DATASET_ZIP_PATH(self) -> Path:
        return self.ROOT / "Dataset.zip"
    
    @property
    def RESULTS_ZIP_PATH(self) -> Path:
        return self.ROOT / "Results.zip"

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
    
    @property
    def TEST_RESULTS_DIR(self) -> Path:
        return self.ROOT / "test_results"
    
    @property
    def S1_MODEL(self) -> Path:
        return self.EXP_DIR / "STURM-fusion-24/Sentinel1_SAR/lr_0.001/bs_32/wd_1e-05/dr_0.0"
    
    @property
    def S2_MODEL(self) -> Path:
        return self.EXP_DIR / "STURM-fusion-24/Sentinel2_Optical/lr_0.001/bs_64/wd_1e-05/dr_0.2"
    
    @property
    def FUSION_MODEL(self) -> Path:
        return self.EXP_DIR / "STURM-fusion-24/Fusion_SAR_Optical/lr_0.001/bs_64/wd_1e-05/dr_0.2"

    @property
    def S1_TEST_RESULTS_CSV(self) -> Path:
        return self.ROOT / "test_results/s1_test_results.csv"

    @property
    def S2_TEST_RESULTS_CSV(self) -> Path:
        return self.ROOT / "test_results/s2_test_results.csv"

    @property
    def FUSION_TEST_RESULTS_CSV(self) -> Path:
        return self.ROOT / "test_results/fusion_test_results.csv"
    
    @property
    def FIG_EXPORTS_DIR(self) -> Path:
        return self.ROOT / "fig_exports"

@dataclass
class S1_CFG(CFG):

    DATA_TYPE: str = "Sentinel1_SAR"

    # input setup
    IMAGE_SIZE: int = 128
    CHANNELS: int = 2

    # SAR dataloader
    BATCH_SIZE: int = 0
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    SHUFFLE_TRAIN: bool = True

    # training
    LR: float = 0

    # model
    MODEL: str = "unet_resnet34_sar"
    DROPOUT_RATE: float = 0.0

    # optimizer
    WEIGHT_DECAY: float = 0.0

@dataclass
class S2_CFG(CFG):

    DATA_TYPE: str = "Sentinel2_Optical"

    # input setup
    IMAGE_SIZE: int = 128
    CHANNELS: int = 9

    # Optical dataloader
    BATCH_SIZE: int = 0
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    SHUFFLE_TRAIN: bool = True

    # training
    LR: float = 0

    # model
    MODEL: str = "unet_optical"
    DROPOUT_RATE: float = 0.0

    # optimizer
    WEIGHT_DECAY: float = 0.0

@dataclass
class Fusion_CFG(CFG):

    DATA_TYPE: str = "Fusion_SAR_Optical"

    # input setup
    IMAGE_SIZE: int = 128
    S1_CHANNELS: int = 2
    S2_CHANNELS: int = 9

    # SAR dataloader
    BATCH_SIZE: int = 0
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    SHUFFLE_TRAIN: bool = True

    # training
    LR: float = 0

    # model
    MODEL: str = "unet_resnet34_fusion"
    ENCODER_WEIGHTS: str = None
    DROPOUT_RATE: float = 0.0

    # optimizer
    WEIGHT_DECAY: float = 0.0