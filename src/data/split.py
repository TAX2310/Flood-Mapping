import random
import pandas as pd
from collections import defaultdict

def build_s1_index(cfg):
    image_dir = cfg.S1_PATH
    mask_dir = cfg.MASK_PATH

    df = pd.read_csv(cfg.METADATA_CSV)

    samples = []
    for i, row in df.iterrows():
        event_id = row["ems_code"]
        img_path = image_dir / row["tile_id"]
        mask_path = mask_dir / row["tile_id"]

        samples.append({
            "id": i,
            "event_id": event_id,
            "image_path": img_path,
            "mask_path": mask_path,
        })

    return samples

def build_s2_index(cfg):
    image_dir = cfg.S2_PATH
    mask_dir = cfg.MASK_PATH

    df = pd.read_csv(cfg.METADATA_CSV)

    samples = []
    for i, row in df.iterrows():
        event_id = row["ems_code"]
        img_path = image_dir / row["tile_id"]
        mask_path = mask_dir / row["tile_id"]

        samples.append({
            "id": i,
            "event_id": event_id,
            "image_path": img_path,
            "mask_path": mask_path,
        })

    return samples

def split_by_event(samples, cfg):
    # 1. Group tiles by event
    groups = defaultdict(list)
    for s in samples:
        event_id = s["event_id"]
        groups[event_id].append(s)

    train_events = set(cfg.TRAIN_EVENTS)
    val_events   = set(cfg.VAL_EVENTS)
    test_events  = set(cfg.TEST_EVENTS)

    # 2. Safety checks
    overlap = (train_events & val_events) | (train_events & test_events) | (val_events & test_events)
    assert not overlap, f"Overlapping events found across splits: {overlap}"

    # 3. Flatten back to tile-level
    train_samples = [s for e in train_events for s in groups[e]]
    val_samples   = [s for e in val_events   for s in groups[e]]
    test_samples  = [s for e in test_events  for s in groups[e]]

    return train_samples, val_samples, test_samples