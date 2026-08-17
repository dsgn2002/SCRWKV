"""Standalone Structure-Field Encoder for SCRWKV.

This module preserves the architecture released by Zhang et al. while removing
the model-time dependency on the vendored MMCV/MMClassification tree.  The
original implementation remains available under ``mmcls/SFE_dev`` for
reference.  Pyramid sizes are derived from the input image instead of being
fixed to the paper's 512 x 512 training resolution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from models.AMCM import AMCM
from models.SCIU import Block


class ConvPatchEmbed(nn.Module):
    """The convolutional patch stem used by the released SFE implementation."""

    def __init__(self, in_channels=3, embed_dims=256, num_convs=2,
                 patch_size=4):
        super().__init__()
        if patch_size != 4:
            raise ValueError("The released SCRWKV architecture expects patch_size=4")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.GroupNorm(num_channels=64, num_groups=4),
            nn.ReLU(True),
        )
        convs = []
        for _ in range(num_convs):
            convs.extend([
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.GroupNorm(num_channels=64, num_groups=4),
                nn.ReLU(True),
            ])
        self.convs = nn.Sequential(*convs) if convs else nn.Identity()
        self.projection = nn.Conv2d(
            64, embed_dims, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        x = self.projection(self.convs(self.stem(x)))
        resolution = tuple(x.shape[-2:])
        return x.flatten(2).transpose(1, 2), resolution


class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            mid_channels, mid_channels, kernel_size, stride, padding,
            groups=mid_channels, bias=False,
        )
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise_2(self.depthwise(self.pointwise_1(x)))


class SFE(nn.Module):
    """Structure-Field Encoder with a four-level dynamic feature pyramid."""

    arch_zoo = {
        "Crack": {
            "patch_size": 4,
            "embed_dims": 256,
            "num_layers": 4,
            "num_convs_patch_embed": 2,
        }
    }

    def __init__(self, img_size=224, in_channels=3, arch="Crack",
                 out_indices=(0, 1, 2, 3), drop_rate=0.0,
                 drop_path_rate=0.0, final_norm=True,
                 interpolate_mode="bicubic", **_kwargs):
        super().__init__()
        if arch not in self.arch_zoo:
            raise ValueError(f"Unsupported SFE architecture: {arch}")
        cfg = self.arch_zoo[arch]
        self.embed_dims = cfg["embed_dims"]
        self.num_layers = cfg["num_layers"]
        self.patch_size = cfg["patch_size"]
        self.interpolate_mode = interpolate_mode
        self.out_indices = tuple(out_indices)
        if any(i < 0 or i >= self.num_layers for i in self.out_indices):
            raise ValueError(
                f"out_indices must be within [0, {self.num_layers - 1}]"
            )

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            num_convs=cfg["num_convs_patch_embed"],
            patch_size=self.patch_size,
        )
        base_h = int(np.ceil(img_size / self.patch_size))
        base_w = base_h
        self.base_resolution = (base_h, base_w)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, base_h * base_w, self.embed_dims)
        )
        trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = nn.ModuleList([
            Block(
                n_embd=self.embed_dims,
                n_layer=self.num_layers,
                layer_id=i,
                channel_gamma=1 / 4,
                shift_pixel=1,
                shift_mode1="GBST",
                hidden_rate=4,
                drop_path=dpr[i],
                init_mode="fancy",
                init_values=None,
                post_norm=False,
                key_norm=False,
                with_cp=False,
            )
            for i in range(self.num_layers)
        ])
        self.output_norms = nn.ModuleDict({
            str(i): (
                nn.LayerNorm(self.embed_dims)
                if i != self.num_layers - 1 else nn.Identity()
            )
            for i in self.out_indices
        })
        self.final_norm = final_norm

        self.projections = nn.ModuleList([
            BottConv(256, 128, 64, kernel_size=1),
            BottConv(256, 64, 32, kernel_size=1),
            BottConv(256, 32, 16, kernel_size=1),
            BottConv(256, 16, 8, kernel_size=1),
        ])
        self.projection_norms = nn.ModuleList([
            nn.GroupNorm(8, 128),
            nn.GroupNorm(4, 64),
            nn.GroupNorm(2, 32),
            nn.GroupNorm(2, 16),
        ])
        self.AMCM = AMCM(self.embed_dims)

    def _resize_pos_embed(self, dst_shape):
        if dst_shape == self.base_resolution:
            return self.pos_embed
        src_h, src_w = self.base_resolution
        pos = self.pos_embed.reshape(
            1, src_h, src_w, self.embed_dims
        ).permute(0, 3, 1, 2)
        pos = F.interpolate(
            pos, size=dst_shape, mode=self.interpolate_mode,
            align_corners=False,
        )
        return pos.flatten(2).transpose(1, 2)

    @staticmethod
    def _pyramid_sizes(input_size):
        height, width = input_size
        return tuple(
            (max(1, int(np.ceil(height / divisor))),
             max(1, int(np.ceil(width / divisor))))
            for divisor in (8, 4, 2, 1)
        )

    def forward(self, image):
        input_size = tuple(image.shape[-2:])
        tokens, patch_resolution = self.patch_embed(image)
        tokens = self.drop_after_pos(
            tokens + self._resize_pos_embed(patch_resolution)
        )

        batch, _, channels = tokens.shape
        height, width = patch_resolution
        spatial = tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        tokens = self.AMCM(spatial).flatten(2).transpose(1, 2)

        raw_outputs = []
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens, patch_resolution)
            if index in self.out_indices:
                normalized = self.output_norms[str(index)](tokens)
                feature = normalized.reshape(
                    batch, height, width, channels
                ).permute(0, 3, 1, 2).contiguous()
                raw_outputs.append(feature)

        if len(raw_outputs) != 4:
            raise RuntimeError(
                f"CSHF requires four SFE outputs, received {len(raw_outputs)}"
            )

        pyramid = []
        for feature, projection, norm, size in zip(
                raw_outputs, self.projections, self.projection_norms,
                self._pyramid_sizes(input_size)):
            feature = norm(projection(feature))
            feature = F.interpolate(
                feature, size=size, mode="bilinear", align_corners=False
            )
            pyramid.append(feature)
        return pyramid
