import random
from collections import defaultdict
from S1.util.path import *

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

def group_by_event(pairs):
    groups = defaultdict(list)

    for p in pairs:
        event_id = p["id"].split("_")[0]
        groups[event_id].append(p)

    return groups

def split_by_event(pairs, cfg):
    # 1. Group tiles by event
    groups = defaultdict(list)
    for p in pairs:
        event_id = p["event_id"] if "event_id" in p else p["id"].split("_")[0]
        groups[event_id].append(p)

    event_ids = list(groups.keys())

    # 2. Shuffle events (reproducible)
    rng = random.Random(cfg.RANDOM_SEED)
    rng.shuffle(event_ids)

    # 3. Fixed split
    n_train = cfg.N_TRAIN_EVENTS
    n_val = cfg.N_VAL_EVENTS
    n_test = cfg.N_TEST_EVENTS

    assert len(event_ids) == (n_train + n_val + n_test), \
        f"Expected 47 events, got {len(event_ids)}"

    train_events = event_ids[:n_train]
    val_events   = event_ids[n_train:n_train + n_val]
    test_events  = event_ids[n_train + n_val:]

    # 4. Flatten back to tile-level
    train_pairs = [p for e in train_events for p in groups[e]]
    val_pairs   = [p for e in val_events   for p in groups[e]]
    test_pairs  = [p for e in test_events  for p in groups[e]]

    return train_pairs, val_pairs, test_pairs