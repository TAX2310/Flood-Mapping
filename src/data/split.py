import random
import pandas as pd
from collections import defaultdict

def build_s1_index(cfg):
    """
    Build an index of samples for S1 data.
    """
    image_dir = cfg.S1_PATH
    mask_dir = cfg.MASK_PATH

    # Read metadata CSV
    df = pd.read_csv(cfg.METADATA_CSV)

    samples = []
    # Iterate over each row in the DataFrame to create sample entries
    for i, row in df.iterrows():
        sample_id = row["tile_id"]
        event_id = row["ems_code"]
        img_path = image_dir / row["tile_id"]
        mask_path = mask_dir / row["tile_id"]

        samples.append({
            "id": i,
            "sample_id": sample_id,
            "event_id": event_id,
            "image_path": img_path,
            "mask_path": mask_path,
        })

    return samples

def build_s2_index(cfg):
    """
    Build an index of samples for S2 data.
    """
    image_dir = cfg.S2_PATH
    mask_dir = cfg.MASK_PATH

    # Read metadata CSV
    df = pd.read_csv(cfg.METADATA_CSV)

    samples = []
    # Iterate over each row in the DataFrame to create sample entries
    for i, row in df.iterrows():
        sample_id = row["tile_id"]
        event_id = row["ems_code"]
        img_path = image_dir / row["tile_id"]
        mask_path = mask_dir / row["tile_id"]

        samples.append({
            "id": i,
            "sample_id": sample_id,
            "event_id": event_id,
            "image_path": img_path,
            "mask_path": mask_path,
        })

    return samples

def build_fusion_index(cfg):
    """
    Build an index of samples for fused S1 and S2 data.
    """
    s1_image_dir = cfg.S1_PATH
    s2_image_dir = cfg.S2_PATH
    mask_dir = cfg.MASK_PATH

    # Read metadata CSV
    df = pd.read_csv(cfg.METADATA_CSV)

    samples = []
    # Iterate over each row in the DataFrame to create sample entries
    for i, row in df.iterrows():
        sample_id = row["tile_id"]
        event_id = row["ems_code"]
        s1_img_path = s1_image_dir / row["tile_id"]
        s2_img_path = s2_image_dir / row["tile_id"]
        mask_path = mask_dir / row["tile_id"]

        samples.append({
            "id": i,
            "sample_id": sample_id,
            "event_id": event_id,
            "s1_image_path": s1_img_path,
            "s2_image_path": s2_img_path,
            "mask_path": mask_path,
        })

    return samples

def split_by_event(samples, cfg):
    """
    Split samples into train, validation, and test sets based on event IDs.
    """
    # Group tiles by event
    groups = defaultdict(list)
    for s in samples:
        event_id = s["event_id"]
        groups[event_id].append(s)

    train_events = set(cfg.TRAIN_EVENTS)
    val_events   = set(cfg.VAL_EVENTS)
    test_events  = set(cfg.TEST_EVENTS)

    # Safety checks
    overlap = (train_events & val_events) | (train_events & test_events) | (val_events & test_events)
    assert not overlap, f"Overlapping events found across splits: {overlap}"

    # Flatten back to tile-level
    train_samples = [s for e in train_events for s in groups[e]]
    val_samples   = [s for e in val_events   for s in groups[e]]
    test_samples  = [s for e in test_events  for s in groups[e]]

    return train_samples, val_samples, test_samples


def split_random(samples, cfg):
    """
    Randomly split samples into train, validation, and test sets based on specified ratios.
    """
    assert len(samples) > 0, "Sample list is empty. Cannot perform random split."

    # Ensure that the split ratios sum to 1
    total_split = cfg.TRAIN_SPLIT + cfg.VAL_SPLIT + cfg.TEST_SPLIT
    assert abs(total_split - 1.0) < 1e-6, f"Splits must sum to 1. Got {total_split}"

    # Make a copy so the original list is not changed
    samples = list(samples)
    
    # Shuffle the samples to ensure randomness
    random.shuffle(samples)

    total = len(samples)

    # Calculate split sizes
    n_train = int(total * cfg.TRAIN_SPLIT)
    n_val = int(total * cfg.VAL_SPLIT)

    # Test gets everything left over
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    return train_samples, val_samples, test_samples