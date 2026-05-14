import segmentation_models_pytorch as smp

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