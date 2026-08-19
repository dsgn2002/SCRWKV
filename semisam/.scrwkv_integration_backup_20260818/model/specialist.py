"""2D U-Net specialist with ResNet encoder.

Ported from SemiSAM+ networks/unet_3D.py — 3D U-Net → 2D U-Net.
Uses torchvision ResNet-34 as encoder (ImageNet pretrained).
Decoder is standard U-Net upsample blocks with skip connections.

Later additions (Tasks 8-9): variance head + UPFM module.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from data.transforms import scale_crop_zoom, scale_pad_zoom, align_prediction


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU (×2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsample → ConvBlock with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd-size mismatches from upsampling
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# specialist
# ---------------------------------------------------------------------------

class Specialist(nn.Module):
    """2D U-Net with ResNet encoder for binary crack segmentation.

    Args:
        encoder_name: torchvision backbone (default: 'resnet34')
        decoder_channels: list of decoder output channels per block
        in_channels: input channels (3 for RGB)
        pretrained: load ImageNet weights
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        decoder_channels: tuple = (256, 128, 64, 32),
        in_channels: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_channels = decoder_channels

        # --- encoder ---
        backbone = getattr(torchvision.models, encoder_name)(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        # Handle 3-channel first conv (for RGB input, standard case)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool0 = backbone.maxpool  # stride-2 pooling
        self.encoder1 = backbone.layer1  # 64 ch,  H/4
        self.encoder2 = backbone.layer2  # 128 ch, H/8
        self.encoder3 = backbone.layer3  # 256 ch, H/16
        self.encoder4 = backbone.layer4  # 512 ch, H/32

        encoder_out_ch = self._encoder_channels(backbone)

        # --- center ---
        self.center = ConvBlock(encoder_out_ch[4], encoder_out_ch[4] * 2)

        # --- decoder ---
        center_ch = encoder_out_ch[4] * 2
        dec_ch = decoder_channels

        self.dec3 = DecoderBlock(center_ch, encoder_out_ch[3], dec_ch[0])  # 1024+256→256
        self.dec2 = DecoderBlock(dec_ch[0], encoder_out_ch[2], dec_ch[1])   # 256+128→128
        self.dec1 = DecoderBlock(dec_ch[1], encoder_out_ch[1], dec_ch[2])   # 128+64→64
        self.dec0 = DecoderBlock(dec_ch[2], encoder_out_ch[0], dec_ch[3])   # 64+64→32

        # --- heads ---
        self._final_ch = dec_ch[3]
        self.mask_head = nn.Conv2d(self._final_ch, 1, kernel_size=1)

        # ponytail: auxiliary modules attached lazily
        self.structure_risks: nn.Module | None = None
        self.consultation_predictor: nn.Module | None = None

        self._init_weights()

    @staticmethod
    def _encoder_channels(backbone) -> dict:
        """Return channel counts at each encoder level."""
        return {
            0: 64,    # after conv1+bn+relu (before maxpool)
            1: 64,    # layer1 out
            2: 128,   # layer2 out
            3: 256,   # layer3 out
            4: 512,   # layer4 out
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with:
            'mask_logits': raw logits (B, 1, H, W)
            'log_sigma2':  variance logits (B, 1, H, W), None if no variance head
            'features':    pre-head feature map (B, C, H, W), for UPFM input later
        """
        # --- encode ---
        e0 = self.encoder0(x)          # (B, 64, H/2, W/2)
        e0p = self.pool0(e0)           # (B, 64, H/4, W/4)
        e1 = self.encoder1(e0p)        # (B, 64, H/4, W/4)
        e2 = self.encoder2(e1)         # (B, 128, H/8, W/8)
        e3 = self.encoder3(e2)         # (B, 256, H/16, W/16)
        e4 = self.encoder4(e3)         # (B, 512, H/32, W/32)

        # --- center ---
        c = self.center(e4)            # (B, 1024, H/32, W/32)

        # --- decode ---
        d3 = self.dec3(c, e3)          # (B, 256, H/16, W/16)
        d2 = self.dec2(d3, e2)         # (B, 128, H/8, W/8)
        d1 = self.dec1(d2, e1)         # (B, 64, H/4, W/4)
        d0 = self.dec0(d1, e0)         # (B, 32, H/2, W/2)

        features = d0  # penultimate features for UPFM

        # --- heads ---
        mask_logits = self.mask_head(features)

        # Structure risk maps (topology, morphology, appearance)
        risks = {}
        if self.structure_risks is not None:
            risks = self.structure_risks(features)
            # Upsample risk maps to match input size if needed
            if mask_logits.shape[2:] != x.shape[2:]:
                risks = {
                    k: F.interpolate(v, size=x.shape[2:], mode="bilinear", align_corners=True)
                    for k, v in risks.items()
                }

        # ponytail: upsample to match input spatial size
        if mask_logits.shape[2:] != x.shape[2:]:
            mask_logits = F.interpolate(
                mask_logits, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        result = {
            "mask_logits": mask_logits,
            "features": features,
        }
        result.update(risks)  # adds R_topo, R_morph, R_app if available

        return result

    def add_structure_risks(self, risk_module: nn.Module):
        """Attach structure-aware risk heads (topology calibration)."""
        self.structure_risks = risk_module

    def add_consultation_predictor(self, predictor: nn.Module):
        """Attach consultation-value predictor MLP."""
        self.consultation_predictor = predictor

    def get_teacher(self) -> "Specialist":
        """Return an EMA-friendly copy with detached params (Mean Teacher init)."""
        import copy
        teacher = copy.deepcopy(self)
        for p in teacher.parameters():
            p.detach_()
        return teacher
