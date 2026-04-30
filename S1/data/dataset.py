from S1.util.io import *
from S1.util.preprocessing import *

import torch
from torch.utils.data import Dataset

class SturmS1Dataset(Dataset):
    def __init__(self, samples, cfg, is_train=False, use_rotation=False):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train
        self.use_rotation = use_rotation

    def __len__(self):
        return len(self.samples) * 4 if self.use_rotation else len(self.samples)


    def __getitem__(self, idx):
        if self.use_rotation:
            base_idx = idx % len(self.samples)
            rot_idx = idx // len(self.samples)
        else:
            base_idx = idx
            rot_idx = 0

        item = self.samples[base_idx]

        image = read_image_tif(item["image_path"])
        mask = read_mask_tif(item["mask_path"])

        # checks
        assert image.shape[0] == 2
        assert image.shape[1] == self.cfg.IMAGE_SIZE
        assert image.shape[2] == self.cfg.IMAGE_SIZE

        # normalise
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

        # rotation
        if rot_idx > 0:
            image = torch.rot90(image, k=rot_idx, dims=[1, 2])
            mask  = torch.rot90(mask,  k=rot_idx, dims=[0, 1])

        return {
            "image": image.float(),
            "mask": mask.long(),
            "id": item["id"],
            "event_id": item["event_id"],
        }