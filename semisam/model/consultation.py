"""Consultation-value predictor — predicts TopoSAM benefit per component.

Two variants:
  ConsultationPredictor   — MLP on scalar features (ablation baseline).
  CrossAttentionConsultation — morph queries attend to topo tokens,
                               learning feature interactions the MLP misses.

Dropout is the only regularizer (49 labeled images → few hundred components).
Attach to specialist via add_consultation_predictor().
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConsultationPredictor(nn.Module):
    """MLP baseline: risk vector + component features → Δ̂_j.

    Input: [R_topo, R_morph, confidence, log(area),
            log(length), log(width), n_endpoints, bbox_fill_ratio]  — 8 features.

    Output: Δ̂_j ∈ [-1, 1] — positive = SAM expected to help.
    """

    def __init__(self, in_features: int = 8, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, component_features: torch.Tensor) -> torch.Tensor:
        """(N, 8) → (N, 1) Δ̂_j in [-1, 1]."""
        return self.mlp(component_features)


class CrossAttentionConsultation(nn.Module):
    """Cross-attention fusion of morphological and topological features.

    Morph tokens (boundary, contrast) attend to topo tokens (skeleton,
    endpoints). Learns interactions like "thin + fragmented + uncertain
    boundary → query SAM" that a concatenated MLP cannot.

    Input per component:
      topo_feats:  (D_t,) — pooled skeleton/endpoint features
      morph_feats: (D_m,) — boundary/contrast features

    Output: Δ̂_j ∈ [-1, 1].
    """

    def __init__(self, topo_dim: int, morph_dim: int, d_model: int = 64,
                 n_heads: int = 1, dropout: float = 0.3):
        super().__init__()
        self.topo_proj = nn.Linear(topo_dim, d_model)
        self.morph_proj = nn.Linear(morph_dim, d_model)
        # ponytail: 1-head attention, learnable query — minimal overfit surface
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(self, topo_feats: torch.Tensor, morph_feats: torch.Tensor
                ) -> torch.Tensor:
        """(N, D_t) + (N, D_m) → (N, 1) Δ̂_j in [-1, 1]."""
        N = topo_feats.shape[0]
        t = self.topo_proj(topo_feats).unsqueeze(1)   # (N, 1, d) topo token
        m = self.morph_proj(morph_feats).unsqueeze(1) # (N, 1, d) morph token
        tokens = torch.cat([t, m], dim=1)              # (N, 2, d)

        q = self.query.expand(N, -1, -1)                # (N, 1, d)
        fused, _ = self.attn(q, tokens, tokens)         # (N, 1, d)
        return self.head(fused.squeeze(1))               # (N, 1)
