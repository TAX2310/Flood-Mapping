import segmentation_models_pytorch as smp

def get_model(model_name):
    if model_name == "unet_sar":
        return unet_sar()
    elif model_name == "unet_resnet34_sar":
        return unet_resnet34_sar()
    elif model_name == "unet_plus_plus_sar":
        return unet_plus_plus_sar()
    elif model_name == "unet_plus_plus_resnet34_sar":
        return unet_plus_plus_resnet34_sar()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def unet_sar():

    model = smp.Unet(
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=2,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_resnet34_sar():

    model = smp.Unet(
        encoder_name="resnet34",        # ResNet34 encoder
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=2,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_plus_plus_sar():

    model = smp.UnetPlusPlus(
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=2,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model

def unet_plus_plus_resnet34_sar():

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        # ResNet34 encoder
        encoder_weights=None,           # None for SAR (not RGB ImageNet)
        in_channels=2,                  # Your S1 input (VV, VH)
        classes=1                       # Binary segmentation output
    )
    return model