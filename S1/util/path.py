from pathlib import Path

def get_data_dirs(cfg):
    root = Path(cfg.DATA_PATH)
    img_dir = root / cfg.S1_IMAGE_PATH
    mask_dir = root / cfg.S1_MASK_PATH
    return img_dir, mask_dir
