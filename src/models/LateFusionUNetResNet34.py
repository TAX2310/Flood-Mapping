# src/fusion/models/fusion_unet_resnet34.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class LateFusionUNetResNet34(nn.Module):
    """
    A late fusion U-Net model with ResNet34 encoders for two input modalities (S1 and S2).
    """

    def __init__(
        self,
        s1_channels=2,
        s2_channels=9,
        encoder_weights=None,
    ):
        super().__init__()

        # S1 encoder
        self.s1_encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=s1_channels,
            depth=5,
            weights=encoder_weights,
        )

        # S2 encoder
        self.s2_encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=s2_channels,
            depth=5,
            weights=encoder_weights,
        )

        # ResNet34 encoder channels:
        # [in_channels, 64, 64, 128, 256, 512]
        s1_encoder_channels = self.s1_encoder.out_channels
        s2_encoder_channels = self.s2_encoder.out_channels
        encoder_channels = self.s1_encoder.out_channels


        # 1x1 convolutions to reduce concatenated channels back to normal U-Net decoder size.
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1 + c2, cf, kernel_size=1),
                nn.BatchNorm2d(cf),
                nn.ReLU(inplace=True),
            )
            for c1, c2, cf in zip(
                s1_encoder_channels,
                s2_encoder_channels,
                encoder_channels
            )
        ])

        # Shared U-Net decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None,
        )

        # Final segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=1,
            activation=None,
            kernel_size=3,
        )

    def forward(self, s1, s2):
        # Extract encoder features
        s1_features = self.s1_encoder(s1)
        s2_features = self.s2_encoder(s2)

        # Fuse features at every encoder level
        fused_features = []
        for f1, f2, fuse in zip(s1_features, s2_features, self.fusion_blocks):
            fused = torch.cat([f1, f2], dim=1)
            fused = fuse(fused)
            fused_features.append(fused)

        # Decode fused features
        decoder_output = self.decoder(fused_features)
        logits = self.segmentation_head(decoder_output)

        return logits