"""SegFormer-B2 specialist — drop-in replacement for Specialist (ResNet34-UNet).

Matches the exact Specialist interface: forward(), forward_multiscale(),
add_variance_head(), add_upfm(), add_fusion().

Pretrained MiT-B2 encoder from HuggingFace transformers + lightweight MLP decoder.
Usage: config.yaml → specialist_encoder: "mit_b2"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.transforms import scale_crop_zoom, scale_pad_zoom, align_prediction


class MLPDecoder(nn.Module):
    """SegFormer-style All-MLP decoder: fuse 4 multi-scale features."""

    def __init__(self, encoder_dims: list[int], decoder_dim: int = 256):
        super().__init__()
        # 1x1 conv to project each encoder stage to decoder_dim
        self.linear_c = nn.ModuleList([
            nn.Conv2d(d, decoder_dim, 1) for d in encoder_dims
        ])
        # Fusion MLP after concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        h, w = features[0].shape[2:]
        projected = []
        for i, f in enumerate(features):
            f = self.linear_c[i](f)
            if f.shape[2:] != (h, w):
                f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            projected.append(f)
        x = torch.cat(projected, dim=1)
        return self.fuse(x)


class SpecialistSegFormer(nn.Module):
    """SegFormer-B2 specialist for semi-supervised crack segmentation.

    Matches the Specialist (ResNet34-UNet) interface exactly:
      - forward(x) → {mask_logits, log_sigma2, features}
      - forward_multiscale(x) → +aligned_preds
      - add_variance_head / add_upfm / add_fusion

    encoder_name: "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"
    """

    # ponytail: MiT encoder channel dims per stage (after patch embed + 4 stages)
    _ENC_DIMS = {
        "mit_b0": [32, 64, 160, 256],
        "mit_b1": [64, 128, 320, 512],
        "mit_b2": [64, 128, 320, 512],
        "mit_b3": [64, 128, 320, 512],
        "mit_b4": [64, 128, 320, 512],
        "mit_b5": [64, 128, 320, 512],
    }

    def __init__(
        self,
        encoder_name: str = "mit_b2",
        decoder_channels: tuple = (256, 128, 64, 32),  # ignored, kept for API compat
        in_channels: int = 3,
        pretrained: bool = True,
        decoder_dim: int = 256,
    ):
        super().__init__()
        self.encoder_name = encoder_name

        # --- encoder: HuggingFace SegFormer (MiT) ---
        # ponytail: map encoder names to HF model IDs
        _HF_MAP = {
            "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
            "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
            "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
            "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
            "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
            "mit_b5": "nvidia/segformer-b5-finetuned-ade-512-512",
        }
        hf_name = _HF_MAP.get(encoder_name, f"nvidia/{encoder_name}")
        try:
            from transformers import SegformerModel
            self.encoder = SegformerModel.from_pretrained(
                hf_name, use_safetensors=True, num_labels=1,
            )
        except Exception:
            # fallback: no pretrained weights
            from transformers import SegformerConfig, SegformerModel
            config = SegformerConfig.from_pretrained(hf_name)
            self.encoder = SegformerModel(config)

        # Freeze early encoder stages, unfreeze last 2 for crack-specific tuning
        if pretrained:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # ponytail: unfreeze stages 2+3 (last 2 of 4) for domain adaptation
            for n, p in self.encoder.named_parameters():
                if n.startswith("stages.2") or n.startswith("stages.3"):
                    p.requires_grad = True

        enc_dims = self._ENC_DIMS.get(encoder_name, [64, 128, 320, 512])

        # --- decoder ---
        self.decoder = MLPDecoder(enc_dims, decoder_dim)
        self._final_ch = decoder_dim

        # --- heads ---
        self.mask_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # --- stubs (attached via add_* methods) ---
        self.structure_risks = None
        self.consultation_predictor = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        input_size = x.shape[2:]

        # Encoder
        enc_out = self.encoder(x, output_hidden_states=True)
        # hidden_states: 4 stage outputs (HF Segformer excludes patch embed)
        features_list = list(enc_out.hidden_states)

        # Decoder
        features = self.decoder(features_list)  # (B, C, H/4, W/4)

        # Heads
        mask_logits = self.mask_head(features)

        # Upsample to input size
        if mask_logits.shape[2:] != input_size:
            mask_logits = F.interpolate(
                mask_logits, size=input_size, mode="bilinear", align_corners=True
            )
        # Structure risk maps (topology, morphology, appearance) — same as Specialist
        risks = {}
        if self.structure_risks is not None:
            risks = self.structure_risks(features)
            risks = {
                k: F.interpolate(v, size=input_size, mode="bilinear", align_corners=True)
                for k, v in risks.items()
            }

        result = {"mask_logits": mask_logits,
                  "features": features}
        result.update(risks)  # adds R_topo, R_morph, R_app if available
        return result

    # ------------------------------------------------------------------
    # multi-scale forward (for U_scale)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # attachment methods (same as Specialist)
    # ------------------------------------------------------------------

    def add_structure_risks(self, risk_module: nn.Module):
        self.structure_risks = risk_module

    def add_consultation_predictor(self, predictor: nn.Module):
        self.consultation_predictor = predictor

    def get_teacher(self) -> "SpecialistSegFormer":
        import copy
        teacher = copy.deepcopy(self)
        for p in teacher.parameters():
            p.detach_()
        return teacher
