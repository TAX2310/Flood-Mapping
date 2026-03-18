import torch
from torch.utils.data import Dataset

from S1.util.io import *
from S1.util.preprocessing import *

class SturmS1Dataset(Dataset):
    def __init__(self, pairs, cfg, is_train=False):
        self.pairs = pairs
        self.cfg = cfg
        self.is_train = is_train

    def get_events(self):
        return sorted({p["event_id"] for p in self.pairs})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        image = read_image_tif(item["image_path"])   # torch.Tensor [C,H,W]
        mask = read_mask_tif(item["mask_path"])      # torch.Tensor [H,W]

        # sanity checks
        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape}"
        assert image.shape[1] == self.cfg.IMAGE_SIZE and image.shape[2] == self.cfg.IMAGE_SIZE
        assert mask.shape[0] == self.cfg.IMAGE_SIZE and mask.shape[1] == self.cfg.IMAGE_SIZE

        image = minmax_normalise_s1(
            image,
            min_db=self.cfg.S1_MIN_DB,
            max_db=self.cfg.S1_MAX_DB
        )

        if self.cfg.BINARY_MASK:
            mask = remap_mask_to_binary(
                mask,
                water_classes=self.cfg.WATER_CLASSES,
                ignore_classes=self.cfg.IGNORE_CLASSES
            )

        return {
            "image": image.float(),
            "mask": mask.long(),
            "id": item["id"],
        }