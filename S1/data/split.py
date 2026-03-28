import random
from collections import defaultdict
from S1.util.path import *
from data.intersection import get_intersection_ids

def build_s1_index(cfg):
    img_dir, mask_dir = get_data_dirs(cfg)

    image_paths = sorted(img_dir.glob("*.tif"))
    mask_paths = sorted(mask_dir.glob("*.tif"))

    mask_map = {p.name: p for p in mask_paths}

    pairs = []
    for img_path in image_paths:
        mask_path = mask_map.get(img_path.name)
        if mask_path is None:
            continue

        sample_id = img_path.stem
        event_id = sample_id.split("_")[0]

        pairs.append({
            "id": sample_id,
            "event_id": event_id,
            "image_path": img_path,
            "mask_path": mask_path,
        })

    intersection_ids = get_intersection_ids(
        cfg.S1_METADATA_CSV,
        cfg.S2_METADATA_CSV,
        key="tile_id"
    )
    pairs = [p for p in pairs if p["id"] in intersection_ids]

    return pairs

def split_pairs(pairs, cfg):
    rng = random.Random(cfg.RANDOM_SEED)
    pairs = pairs.copy()
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_val = int(n * cfg.VAL_RATIO)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs

def split_by_event(pairs, cfg):
    # 1. Group tiles by event
    groups = defaultdict(list)
    for p in pairs:
        event_id = p["event_id"]
        groups[event_id].append(p)

    all_events = set(groups.keys())

    train_events = set(cfg.TRAIN_EVENTS)
    val_events   = set(cfg.VAL_EVENTS)
    test_events  = set(cfg.TEST_EVENTS)

    # 2. Safety checks
    overlap = (train_events & val_events) | (train_events & test_events) | (val_events & test_events)
    assert not overlap, f"Overlapping events found across splits: {overlap}"

    assigned_events = train_events | val_events | test_events
    missing_events = all_events - assigned_events
    extra_events = assigned_events - all_events

    assert not missing_events, f"Some events in pairs are not assigned to a split: {missing_events}"
    assert not extra_events, f"Some cfg split events are not present in pairs: {extra_events}"

    # 3. Flatten back to tile-level
    train_pairs = [p for e in train_events for p in groups[e]]
    val_pairs   = [p for e in val_events   for p in groups[e]]
    test_pairs  = [p for e in test_events  for p in groups[e]]

    return train_pairs, val_pairs, test_pairs