from src.fusion.models.LateFusionUNetResNet34 import LateFusionUNetResNet34

def get_model(cfg):
    if cfg.MODEL == "unet_resnet34_fusion":
        return UNet_ResNet34_fusion(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL}")

def UNet_ResNet34_fusion(cfg):
    
    model = LateFusionUNetResNet34(
        s1_channels=cfg.S1_CHANNELS,
        s2_channels=cfg.S2_CHANNELS,
        encoder_weights=cfg.ENCODER_WEIGHTS
    )
    return model