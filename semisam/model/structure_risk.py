"""Structure-aware risk heads for crack analysis.

- Topo head: DeformConv2d — kernel follows local crack curvature (snake-like)
- Morph head: Dilated conv — wider boundary context
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class StructureRiskHeads(nn.Module):
    """Two architecturally distinct per-pixel risk predictors."""

    def __init__(self, in_channels: int = 32, mid_channels: int = 32):
        super().__init__()

        # --- Topo head: deformable → follows crack curvature ---
        self.topo_offset = nn.Conv2d(in_channels, 2 * 9, 3, padding=1)  # (dx,dy) × 9 kernel pts
        self.topo_deform = DeformConv2d(in_channels, mid_channels, 3, padding=1)
        self.topo_out = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(mid_channels, 1, 1), nn.Sigmoid())

        # --- Morph head: dilated → wider boundary context ---
        self.morph_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
            nn.Sigmoid(),
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DeformConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        offset = self.topo_offset(features)
        topo_feat = self.topo_deform(features, offset)

        return {
            "R_topo": self.topo_out(topo_feat),
            "R_morph": self.morph_conv(features),
        }
