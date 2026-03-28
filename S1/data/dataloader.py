from util.seed import *
from S1.data.split import *
from S1.data.dataset import *
import torch
from torch.utils.data import DataLoader

def make_dataloaders(cfg):
    set_seed(cfg.RANDOM_SEED)

    pairs = build_s1_index(cfg)
    train_pairs, val_pairs, test_pairs = split_by_event(pairs, cfg)
    #train_pairs, val_pairs, test_pairs = split_pairs(pairs, cfg)

    train_ds = SturmS1Dataset(train_pairs, cfg, is_train=True)
    val_ds = SturmS1Dataset(val_pairs, cfg, is_train=False)
    test_ds = SturmS1Dataset(test_pairs, cfg, is_train=False)

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