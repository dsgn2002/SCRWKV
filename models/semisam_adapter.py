"""SemiSAM-compatible wrapper around the SCRWKV specialist."""

from __future__ import annotations

import torch.nn as nn

from models.decoder import Decoder
from models.SFE import SFE


class SemiSAMSCRWKV(nn.Module):
    """Return the dictionary interface expected by SemiSAM's training loop.

    SAM remains a training-time teacher.  This adapter exposes SCRWKV's mask
    logits and feature hierarchy without adding SAM masks to the inference
    input.
    """

    def __init__(self, drop_path_rate=0.2):
        super().__init__()
        backbone = SFE(
            arch="Crack",
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
            final_norm=True,
        )
        self.network = Decoder(backbone)
        self._final_ch = self.network.CSHF.embedding_dim
        self.structure_risks = None
        self.consultation_predictor = None

    def add_structure_risks(self, module):
        self.structure_risks = module

    def add_consultation_predictor(self, module):
        self.consultation_predictor = module

    def forward(self, image):
        features = self.network(image, return_features=True)
        output = {
            "mask_logits": features["logits"],
            "features": features["decoder_features"],
            "multiscale_features": features["pyramid_features"],
            "aligned_multiscale_features": features["aligned_features"],
            "harmonic_features": features["harmonic_features"],
            "scale_attention": features["scale_attention"],
        }
        if self.structure_risks is not None:
            risks = self.structure_risks(output["features"])
            if isinstance(risks, dict):
                output.update(risks)
            else:
                output["structure_risks"] = risks
        return output


def build_semisam_scrwkv(**kwargs):
    return SemiSAMSCRWKV(**kwargs)

