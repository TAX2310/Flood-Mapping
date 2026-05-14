import segmentation_models_pytorch as smp

def get_model(cfg):
    model_name = cfg.MODEL
    if model_name == "unet_optical":
        return unet_optical(cfg)
    elif model_name == "unet_resnet34_optical":
        return unet_resnet34_optical(cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
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