import src.util.seed as seed
import src.data.split as split
import src.data.dataset as dataset
import torch
from torch.utils.data import DataLoader

def make_s1_dataloaders(cfg):
    seed.set_seed(cfg.RANDOM_SEED)
    
    samples = split.build_s1_index(cfg)

    if cfg.SPLIT_METHOD == "by_event":
        train_samples, val_samples, test_samples = split.split_by_event(samples, cfg)
    elif cfg.SPLIT_METHOD == "random":
        train_samples, val_samples, test_samples = split.split_random(samples, cfg)

    train_ds = dataset.SturmS1Dataset(train_samples, cfg, is_train=True, use_rotation=cfg.USE_ROTATIONS)
    val_ds = dataset.SturmS1Dataset(val_samples, cfg, is_train=False, use_rotation=False)
    test_ds = dataset.SturmS1Dataset(test_samples, cfg, is_train=False, use_rotation=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE_TRAIN,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


def make_s2_dataloaders(cfg):
    seed.set_seed(cfg.RANDOM_SEED)
    
    samples = split.build_s2_index(cfg)
    if cfg.SPLIT_METHOD == "by_event":
        train_samples, val_samples, test_samples = split.split_by_event(samples, cfg)
    elif cfg.SPLIT_METHOD == "random":
        train_samples, val_samples, test_samples = split.split_random(samples, cfg)

    train_ds = dataset.SturmS2Dataset(train_samples, cfg, is_train=True, use_rotation=cfg.USE_ROTATIONS)
    val_ds = dataset.SturmS2Dataset(val_samples, cfg, is_train=False, use_rotation=False)
    test_ds = dataset.SturmS2Dataset(test_samples, cfg, is_train=False, use_rotation=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE_TRAIN,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader

def make_fusion_dataloaders(cfg):
    seed.set_seed(cfg.RANDOM_SEED)
    
    samples = split.build_fusion_index(cfg)
    if cfg.SPLIT_METHOD == "by_event":
        train_samples, val_samples, test_samples = split.split_by_event(samples, cfg)
    elif cfg.SPLIT_METHOD == "random":
        train_samples, val_samples, test_samples = split.split_random(samples, cfg)

    train_ds = dataset.SturmFusionDataset(train_samples, cfg, is_train=True, use_rotation=cfg.USE_ROTATIONS)
    val_ds = dataset.SturmFusionDataset(val_samples, cfg, is_train=False, use_rotation=False)
    test_ds = dataset.SturmFusionDataset(test_samples, cfg, is_train=False, use_rotation=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE_TRAIN,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader

def make_inference_dataloader(cfg, samples=None):
    seed.set_seed(cfg.RANDOM_SEED)

    if cfg.DATA_TYPE == "Sentinel1_SAR":
        all_samples = split.build_s1_index(cfg)
        if samples is not None:
            samples = [s for s in all_samples if s["sample_id"] in samples]
        else:
            if cfg.SPLIT_METHOD == "by_event":
                _, _, samples = split.split_by_event(all_samples, cfg)
            elif cfg.SPLIT_METHOD == "random":
                _, _, samples = split.split_random(all_samples, cfg)
        ds = dataset.SturmS1Dataset(samples, cfg, is_train=False, use_rotation=False)
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        all_samples = split.build_s2_index(cfg)
        if samples is not None:
            samples = [s for s in all_samples if s["sample_id"] in samples]
        else:
            if cfg.SPLIT_METHOD == "by_event":
                _, _, samples = split.split_by_event(all_samples, cfg)
            elif cfg.SPLIT_METHOD == "random":
                _, _, samples = split.split_random(all_samples, cfg)
        ds = dataset.SturmS2Dataset(samples, cfg, is_train=False, use_rotation=False)
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        all_samples = split.build_fusion_index(cfg)
        if samples is not None:
            samples = [s for s in all_samples if s["sample_id"] in samples]
        else:
            if cfg.SPLIT_METHOD == "by_event":
                _, _, samples = split.split_by_event(all_samples, cfg)
            elif cfg.SPLIT_METHOD == "random":
                _, _, samples = split.split_random(all_samples, cfg)
        ds = dataset.SturmFusionDataset(samples, cfg, is_train=False, use_rotation=False)
    else:
        raise ValueError(f"Unsupported DATA_TYPE: {cfg.DATA_TYPE}")
    

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return loader
    
    
