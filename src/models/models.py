import segmentation_models_pytorch as smp
from src.models.LateFusionUNetResNet34 import LateFusionUNetResNet34

def get_model(cfg):
    model_name = cfg.MODEL
    if model_name == "unet_sar":
        return unet_sar(cfg)
    elif model_name == "unet_resnet34_sar":
        return unet_resnet34_sar(cfg)
    elif model_name == "unet++_sar":
        return unet_plus_plus_sar(cfg)
    elif model_name == "unet++_resnet34_sar":
        return unet_plus_plus_resnet34_sar(cfg)
    elif model_name == "unet_optical":
        return unet_optical(cfg)
    elif model_name == "unet_resnet34_optical":
        return unet_resnet34_optical(cfg)
    elif model_name == "unet_resnet34_fusion":
        return unet_resnet34_fusion(cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    

def unet_sar(cfg):

    model = smp.Unet(
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=cfg.CHANNELS,    # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_resnet34_sar(cfg):

    model = smp.Unet(
        encoder_name="resnet34",        # ResNet34 encoder
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=cfg.CHANNELS,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_plus_plus_sar(cfg):

    model = smp.UnetPlusPlus(
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=cfg.CHANNELS,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_plus_plus_resnet34_sar(cfg):

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        # ResNet34 encoder
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=cfg.CHANNELS,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_optical(cfg):

    model = smp.Unet(
        encoder_weights=None,
        in_channels=cfg.CHANNELS,
        classes=1,
        decoder_dropout=cfg.DROPOUT_RATE
    )
    return model

def unet_resnet34_optical(cfg):

    model = smp.Unet(
        encoder_name="resnet34",        # ResNet34 encoder
        encoder_weights=None,           # None for optical (not RGB ImageNet)
        in_channels=cfg.CHANNELS,                  # Your S2 input (RGB)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_resnet34_fusion(cfg):
    
    model = LateFusionUNetResNet34(
        s1_channels=cfg.S1_CHANNELS,
        s2_channels=cfg.S2_CHANNELS,
        encoder_weights=cfg.ENCODER_WEIGHTS
    )
    return model