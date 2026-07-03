import src.util.seed as seed
import src.data.split as split
import src.data.dataset as dataset
import torch
from torch.utils.data import DataLoader

def make_s1_dataloaders(cfg):
    # Set random seed for reproducibility
    seed.set_seed(cfg.RANDOM_SEED)
    
    # Build the index of samples for Sentinel-1 data
    samples = split.build_s1_index(cfg)

    # Split the samples into training, validation, and test sets based on the split method
    if cfg.SPLIT_METHOD == "by_event":
        train_samples, val_samples, test_samples = split.split_by_event(samples, cfg)
    elif cfg.SPLIT_METHOD == "random":
        train_samples, val_samples, test_samples = split.split_random(samples, cfg)

    # Create datasets for each split
    train_ds = dataset.SturmS1Dataset(train_samples, cfg, is_train=True, use_rotation=cfg.USE_ROTATIONS)
    val_ds = dataset.SturmS1Dataset(val_samples, cfg, is_train=False, use_rotation=False)
    test_ds = dataset.SturmS1Dataset(test_samples, cfg, is_train=False, use_rotation=False)

    # Create DataLoaders for each dataset
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
    # Set random seed for reproducibility
    seed.set_seed(cfg.RANDOM_SEED)
    
    # Build the index of samples for Sentinel-2 data
    samples = split.build_s2_index(cfg)
    
    # Split the samples into training, validation, and test sets based on the split method
    if cfg.SPLIT_METHOD == "by_event":
        train_samples, val_samples, test_samples = split.split_by_event(samples, cfg)
    elif cfg.SPLIT_METHOD == "random":
        train_samples, val_samples, test_samples = split.split_random(samples, cfg)

    # Create datasets for each split
    train_ds = dataset.SturmS2Dataset(train_samples, cfg, is_train=True, use_rotation=cfg.USE_ROTATIONS)
    val_ds = dataset.SturmS2Dataset(val_samples, cfg, is_train=False, use_rotation=False)
    test_ds = dataset.SturmS2Dataset(test_samples, cfg, is_train=False, use_rotation=False)

    # Create DataLoaders for each dataset
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
    # Set random seed for reproducibility
    seed.set_seed(cfg.RANDOM_SEED)
    
    # Build the index of samples for fusion data (Sentinel-1 and Sentinel-2)
    samples = split.build_fusion_index(cfg)
    if cfg.SPLIT_METHOD == "by_event":
        train_samples, val_samples, test_samples = split.split_by_event(samples, cfg)
    elif cfg.SPLIT_METHOD == "random":
        train_samples, val_samples, test_samples = split.split_random(samples, cfg)

    # Create datasets for each split
    train_ds = dataset.SturmFusionDataset(train_samples, cfg, is_train=True, use_rotation=cfg.USE_ROTATIONS)
    val_ds = dataset.SturmFusionDataset(val_samples, cfg, is_train=False, use_rotation=False)
    test_ds = dataset.SturmFusionDataset(test_samples, cfg, is_train=False, use_rotation=False)

    # Create DataLoaders for each dataset
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
    # Set random seed for reproducibility
    seed.set_seed(cfg.RANDOM_SEED)

    # Build the index of samples based on the data type and split method
    if cfg.DATA_TYPE == "Sentinel1_SAR":
        # Build the index of samples for Sentinel-1 data
        all_samples = split.build_s1_index(cfg)
        if samples is not None:
            # Filter the samples based on the provided sample IDs
            samples = [s for s in all_samples if s["sample_id"] in samples]
        else:
            if cfg.SPLIT_METHOD == "by_event":
                _, _, samples = split.split_by_event(all_samples, cfg)
            elif cfg.SPLIT_METHOD == "random":
                _, _, samples = split.split_random(all_samples, cfg)
        # Create the dataset for inference
        ds = dataset.SturmS1Dataset(samples, cfg, is_train=False, use_rotation=False)
    elif cfg.DATA_TYPE == "Sentinel2_Optical":
        # Build the index of samples for Sentinel-2 data
        all_samples = split.build_s2_index(cfg)
        if samples is not None:
            # Filter the samples based on the provided sample IDs
            samples = [s for s in all_samples if s["sample_id"] in samples]
        else:
            if cfg.SPLIT_METHOD == "by_event":
                _, _, samples = split.split_by_event(all_samples, cfg)
            elif cfg.SPLIT_METHOD == "random":
                _, _, samples = split.split_random(all_samples, cfg)
        # Create the dataset for inference
        ds = dataset.SturmS2Dataset(samples, cfg, is_train=False, use_rotation=False)
    elif cfg.DATA_TYPE == "Fusion_SAR_Optical":
        # Build the index of samples for fusion data (Sentinel-1 and Sentinel-2)
        all_samples = split.build_fusion_index(cfg)
        if samples is not None:
            # Filter the samples based on the provided sample IDs
            samples = [s for s in all_samples if s["sample_id"] in samples]
        else:
            if cfg.SPLIT_METHOD == "by_event":
                _, _, samples = split.split_by_event(all_samples, cfg)
            elif cfg.SPLIT_METHOD == "random":
                _, _, samples = split.split_random(all_samples, cfg)
        # Create the dataset for inference
        ds = dataset.SturmFusionDataset(samples, cfg, is_train=False, use_rotation=False)
    else:
        raise ValueError(f"Unsupported DATA_TYPE: {cfg.DATA_TYPE}")
    
    # Create a DataLoader for the inference dataset
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return loader
    
    
