import torch
from torch.utils.data import Dataset

from S1.util.io import *
from S1.util.preprocessing import *

import torch
from torch.utils.data import Dataset

class SturmS1Dataset(Dataset):
    def __init__(self, pairs, cfg, is_train=False, use_rotation=False):
        self.pairs = pairs
        self.cfg = cfg
        self.is_train = is_train
        self.use_rotation = use_rotation

    def __len__(self):
        return len(self.pairs) * self.multiplier

    def __getitem__(self, idx):
        base_idx = idx % len(self.pairs)
        rot_idx = idx // len(self.pairs)  # 0,1,2,3

        item = self.pairs[base_idx]

        image = read_image_tif(item["image_path"])   # [C,H,W]
        mask = read_mask_tif(item["mask_path"])      # [H,W]

        # sanity checks
        assert image.shape[0] == 2
        assert image.shape[1] == self.cfg.IMAGE_SIZE and image.shape[2] == self.cfg.IMAGE_SIZE
        assert mask.shape[0] == self.cfg.IMAGE_SIZE and mask.shape[1] == self.cfg.IMAGE_SIZE

        # normalisation
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

        # 🔁 Apply rotation if enabled
        if self.use_rotation and rot_idx > 0:
            image = torch.rot90(image, k=rot_idx, dims=[1, 2])
            mask  = torch.rot90(mask,  k=rot_idx, dims=[0, 1])

        return {
            "image": image.float(),
            "mask": mask.long(),
            "id": item["id"],
            "event_id": item["event_id"],
        }