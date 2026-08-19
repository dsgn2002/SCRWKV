"""Central specialist factory shared by training and evaluation."""

from __future__ import annotations

from model.specialist import Specialist
from model.specialist_segformer import SpecialistSegFormer
from model.specialist_scrwkv import DEFAULT_SCRWKV_ROOT, SpecialistSCRWKV


def build_specialist(
    encoder_name: str,
    decoder_channels: tuple = (256, 128, 64, 32),
    in_channels: int = 3,
    pretrained: bool = True,
    scrwkv_root: str = DEFAULT_SCRWKV_ROOT,
    scrwkv_risk_scale: float = 0.25,
):
    """Build a ResNet-UNet, SegFormer, or SCRWKV specialist."""
    normalized = encoder_name.lower().replace("-", "_")
    if normalized == "scrwkv":
        return SpecialistSCRWKV(
            encoder_name="scrwkv",
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            pretrained=pretrained,
            scrwkv_root=scrwkv_root,
            risk_scale=scrwkv_risk_scale,
        )
    if normalized.startswith("mit_b"):
        return SpecialistSegFormer(
            encoder_name, decoder_channels, in_channels, pretrained
        )
    return Specialist(encoder_name, decoder_channels, in_channels, pretrained)


def uses_adamw(encoder_name: str) -> bool:
    normalized = encoder_name.lower().replace("-", "_")
    return normalized == "scrwkv" or normalized.startswith("mit_b")
