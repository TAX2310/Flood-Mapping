import segmentation_models_pytorch as smp
from src.models.LateFusionUNetResNet34 import LateFusionUNetResNet34

def get_model(cfg):
    """
    Returns the model based on the configuration.
    """
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
    """
    Returns a U-Net model for SAR data.
    """
    model = smp.Unet(
        encoder_weights=None,           
        in_channels=cfg.CHANNELS,    
        classes=1                       
    )
    return model

def unet_resnet34_sar(cfg):
    """
    Returns a U-Net model with ResNet34 encoder for SAR data.
    """
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None,           
        in_channels=cfg.CHANNELS,                 
        classes=1                       
    )
    return model

def unet_plus_plus_sar(cfg):
    """
    Returns a U-Net++ model for SAR data.
    """
    model = smp.UnetPlusPlus(
        encoder_weights=None,           
        in_channels=cfg.CHANNELS,      
        classes=1                       
    )
    return model

def unet_plus_plus_resnet34_sar(cfg):
    """
    Returns a U-Net++ model with ResNet34 encoder for SAR data.
    """
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights=None,           
        in_channels=cfg.CHANNELS,                  
        classes=1                       
    )
    return model

def unet_optical(cfg):
    """
    Returns a U-Net model for optical data.
    """
    model = smp.Unet(
        encoder_weights=None,
        in_channels=cfg.CHANNELS,
        classes=1,
        decoder_dropout=cfg.DROPOUT_RATE
    )
    return model

def unet_resnet34_optical(cfg):
    """
    Returns a U-Net model with ResNet34 encoder for optical data.
    """
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None,           
        in_channels=cfg.CHANNELS,                  
        classes=1                       
    )
    return model

def unet_resnet34_fusion(cfg):
    """
    Returns a U-Net model with ResNet34 encoder for fusion of SAR and optical data.
    """
    model = LateFusionUNetResNet34(
        s1_channels=cfg.S1_CHANNELS,
        s2_channels=cfg.S2_CHANNELS,
        encoder_weights=cfg.ENCODER_WEIGHTS
    )
    return model