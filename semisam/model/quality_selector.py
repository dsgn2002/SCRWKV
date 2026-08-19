"""Learned per-pixel pseudo-label quality selector (replaces hand-crafted 1−U).

Cross-attention: SAM prob-map tokens query EMA prob-map tokens, both
conditioned on decoder features via small conv encoders. Output w(p) ∈ [0,1]
per pixel — probability that the pseudo-label is correct.

Supervised densely on labeled images: w* = 1[pseudo == GT].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PatchEncoder(nn.Module):
    """1×1-conv + depthwise 3×3: embeds a (H,W) prob map to (d,H,W) tokens."""

    def __init__(self, extra_ch: int, d: int = 64):
        super().__init__()
        in_ch = 1 + extra_ch  # prob map + optional feature channels
        self.proj = nn.Conv2d(in_ch, d, 1)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)

    def forward(self, prob: torch.Tensor, feat: torch.Tensor | None) -> torch.Tensor:
        x = prob if feat is None else torch.cat([prob, feat], dim=1)
        return self.dw(self.proj(x))  # (B, d, H, W)


class QualitySelector(nn.Module):
    """w(p) = P(pseudo-label correct at pixel p).

    SAM tokens (Q) attend to EMA tokens (K,V); both encoders also see the
    shared decoder features so attention has structure, not just two 1-ch maps.
    """

    def __init__(self, feat_ch: int, d: int = 64, n_heads: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.sam_enc = _PatchEncoder(feat_ch, d)
        self.ema_enc = _PatchEncoder(feat_ch, d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(d, 1, 1),
        )

    def forward(
        self,
        sam_prob: torch.Tensor,   # (B, 1, H, W)
        ema_prob: torch.Tensor,   # (B, 1, H, W)
        feat: torch.Tensor,       # (B, C, H, W) decoder features (detached)
    ) -> torch.Tensor:
        B, _, H, W = sam_prob.shape
        # ponytail: features at half-res of mask are fine — resize once
        if feat.shape[2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear",
                                 align_corners=False)

        sam_t = self.sam_enc(sam_prob, feat.detach())   # (B, d, H, W)
        ema_t = self.ema_enc(ema_prob, feat.detach())

        # (B, d, H, W) → (B, HW, d): SAM queries, EMA keys/values
        q = sam_t.flatten(2).transpose(1, 2)
        kv = ema_t.flatten(2).transpose(1, 2)
        fused, _ = self.attn(q, kv, kv)                 # (B, HW, d)

        w = self.head(fused.transpose(1, 2).reshape(B, -1, H, W))
        return torch.sigmoid(w)                          # (B, 1, H, W)
