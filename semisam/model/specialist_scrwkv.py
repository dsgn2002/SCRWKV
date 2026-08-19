"""SCRWKV specialist adapter for the SemiSAM training interface."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SCRWKV_ROOT = "/home/guest/dinu/SCRWKV/SCRWKV"


def _load_scrwkv(root: str):
    root_path = Path(root).expanduser().resolve()
    adapter_path = root_path / "models" / "semisam_adapter.py"
    if not adapter_path.is_file():
        raise FileNotFoundError(
            "SCRWKV SemiSAM adapter was not found at "
            f"{adapter_path}. Check model.scrwkv_root and use the "
            "semisam-integration branch of dsgn2002/SCRWKV."
        )
    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module = importlib.import_module("models.semisam_adapter")
    loaded_path = Path(module.__file__).resolve()
    if root_path not in loaded_path.parents:
        raise ImportError(
            f"Imported SCRWKV adapter from unexpected location: {loaded_path}"
        )
    return module.SemiSAMSCRWKV


class SpecialistSCRWKV(nn.Module):
    """SCRWKV specialist with SemiSAM normalization and attachment hooks.

    SemiSAM supplies ImageNet-normalized RGB tensors. SCRWKV was designed for
    channel-wise normalization with mean/std 0.5, so the input is converted
    internally without changing the shared dataset or SAM denormalization.
    """

    def __init__(
        self,
        encoder_name: str = "scrwkv",
        decoder_channels: tuple = (256, 128, 64, 32),
        in_channels: int = 3,
        pretrained: bool = False,
        scrwkv_root: str = DEFAULT_SCRWKV_ROOT,
        drop_path_rate: float = 0.2,
        risk_scale: float = 0.25,
    ):
        super().__init__()
        del decoder_channels, pretrained
        if encoder_name.lower() != "scrwkv":
            raise ValueError(f"Expected encoder_name='scrwkv', got {encoder_name!r}")
        if in_channels != 3:
            raise ValueError("SCRWKV currently supports three-channel RGB input only")
        if not 0 < risk_scale <= 1:
            raise ValueError("risk_scale must be in (0, 1]")

        base_class = _load_scrwkv(scrwkv_root)
        self.scrwkv = base_class(drop_path_rate=drop_path_rate)
        self.encoder_name = encoder_name
        self.scrwkv_root = str(Path(scrwkv_root).expanduser().resolve())
        self.risk_scale = risk_scale
        self._final_ch = self.scrwkv._final_ch
        self.structure_risks: nn.Module | None = None
        self.consultation_predictor: nn.Module | None = None

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _to_scrwkv_normalization(self, image: torch.Tensor) -> torch.Tensor:
        image_01 = image * self.imagenet_std + self.imagenet_mean
        return image_01.mul(2.0).sub(1.0)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.scrwkv(self._to_scrwkv_normalization(image))

        if self.structure_risks is not None:
            risk_features = output["features"]
            if self.risk_scale < 1:
                risk_features = F.interpolate(
                    risk_features,
                    scale_factor=self.risk_scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
            risks = self.structure_risks(risk_features)
            input_size = image.shape[-2:]
            output.update({
                key: F.interpolate(
                    value, size=input_size, mode="bilinear", align_corners=False
                )
                for key, value in risks.items()
            })
        return output

    def add_structure_risks(self, risk_module: nn.Module):
        self.structure_risks = risk_module

    def add_consultation_predictor(self, predictor: nn.Module):
        self.consultation_predictor = predictor

    def get_teacher(self) -> "SpecialistSCRWKV":
        import copy

        teacher = copy.deepcopy(self)
        for parameter in teacher.parameters():
            parameter.detach_()
        return teacher

