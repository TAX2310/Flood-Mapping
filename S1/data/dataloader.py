import util.seed as seed
import S1.data.split as S1Split
from S1.data.dataset import SturmS1Dataset
import torch
from torch.utils.data import DataLoader

def make_dataloaders(cfg):
    seed.set_seed(cfg.RANDOM_SEED)

    
    samples = S1Split.build_s1_index(cfg)
    train_samples, val_samples, test_samples = S1Split.split_by_event(samples, cfg)

    train_ds = SturmS1Dataset(train_samples, cfg, is_train=True)
    val_ds = SturmS1Dataset(val_samples, cfg, is_train=False)
    test_ds = SturmS1Dataset(test_samples, cfg, is_train=False)

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