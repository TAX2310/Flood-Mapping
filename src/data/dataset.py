from src.util.io import *
from src.data.preprocessing import *

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
            "mask": mask.unsqueeze(0).float(),
            "id": item["id"],
            "event_id": item["event_id"],
        }
    
class SturmS2Dataset(Dataset):
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
            "mask": mask.unsqueeze(0).float(),
            "id": item["id"],
            "event_id": item["event_id"],
        }