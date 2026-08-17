'''
Author: Zhang Hanxu
Github: github.com/zhxhzy/SCRWKV
'''

import torch.nn as nn
import torch
from torch.nn import functional as F
from models.DySample import DySample


class BottConv(nn.Module):
    """Pointwise-depthwise-pointwise bottleneck used by the original CSHF."""

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


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_format = data_format
        if self.channel_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.channel_format == "channels_last":  # [N, H, W, C]
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        elif self.channel_format == "channels_first":  # [N, C, H, W]
            chan_mean = x.mean(1, keepdim=True)                          # μ_c
            chan_var = (x - chan_mean).pow(2).mean(1, keepdim=True)      # σ_c^2
            x_norm = (x - chan_mean) / torch.sqrt(chan_var + self.eps)
            x_out = self.gamma[:, None, None] * x_norm + self.beta[:, None, None]
            return x_out
class ScaleAwareAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_attn = nn.Conv2d(dim * 4, 4, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.expand = nn.Conv2d(dim, dim * 4, kernel_size=1)
        #scale embedding
        # shape: (4, dim)
        self.scale_embed = nn.Parameter(torch.randn(4, dim))

    def forward(self, c4, c3, c2, c1, return_attention=False):
        B, C, H, W = c4.size()

        # --- 加入 scale embedding ---
        c4 = c4 + self.scale_embed[0].view(1, C, 1, 1)
        c3 = c3 + self.scale_embed[1].view(1, C, 1, 1)
        c2 = c2 + self.scale_embed[2].view(1, C, 1, 1)
        c1 = c1 + self.scale_embed[3].view(1, C, 1, 1)


        x = torch.cat([c4, c3, c2, c1], dim=1)       # (B, 4C, H, W)
        attn = self.softmax(self.conv_attn(x))       # (B, 4, H, W)

        out = (
            attn[:, 0:1] * c4 +
            attn[:, 1:2] * c3 +
            attn[:, 2:3] * c2 +
            attn[:, 3:4] * c1
        )
        fused_expanded = self.expand(out)

        if return_attention:
            return fused_expanded, attn
        return fused_expanded
class CSHF(nn.Module):
    def __init__(self, embedding_dim):
        super(CSHF, self).__init__()

        self.embedding_dim = embedding_dim

        self.linear_c4 = nn.Conv2d(128, out_channels=embedding_dim,kernel_size=1)
        self.linear_c3 = nn.Conv2d(64, out_channels=embedding_dim,kernel_size=1)
        self.linear_c2 = nn.Conv2d(32, out_channels=embedding_dim,kernel_size=1)
        self.linear_c1 = nn.Conv2d(16, out_channels=embedding_dim,kernel_size=1)


        self.linear_fuse1 = BottConv(embedding_dim*4, embedding_dim, embedding_dim//4, kernel_size=1, padding=0, stride=1)
        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        # self.sig=nn.Sigmoid()



        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)
        # self.DySample_C_16 = DySample(embedding_dim, scale=16)

        self.norm = LayerNorm(embedding_dim*4 , eps=1e-6, data_format="channels_first")

        self.fusion = ScaleAwareAttentionFusion(embedding_dim)


    def forward(self, inputs, return_features=False):
        c4, c3, c2, c1 = inputs


        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(c4).reshape(b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4)

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(c3).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3)

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(c2).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2)

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1).reshape(b, self.embedding_dim, h, w)

        # DySample preserves the paper's learned upsampling.  Explicit final
        # alignment also supports non-square inputs and dimensions that are
        # not exact powers-of-two multiples.
        target_size = out_c1.shape[-2:]
        aligned = []
        for feature in (out_c4, out_c3, out_c2):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feature)
        out_c4, out_c3, out_c2 = aligned

        # ============================================================
        # ScaleAwareAttentionFusion
        # ============================================================
        scale_fused, scale_attention = self.fusion(
            out_c4, out_c3, out_c2, out_c1, return_attention=True
        )

        # ============================================================


        out_c = torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1)  # 4c

        harmonic_features = self.norm(scale_fused * out_c)

        decoder_features = self.dropout(self.linear_fuse1(harmonic_features))

        logits = self.linear_pred_1(self.linear_pred(decoder_features))
        if not return_features:
            return logits
        return {
            "logits": logits,
            "decoder_features": decoder_features,
            "harmonic_features": harmonic_features,
            "aligned_features": (out_c4, out_c3, out_c2, out_c1),
            "scale_attention": scale_attention,
        }
