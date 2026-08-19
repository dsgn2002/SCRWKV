"""Configuration dataclass + YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    labeled_image_dir: str = "data/train_labeled/images/"
    labeled_mask_dir: str = "data/train_labeled/masks/"
    unlabeled_dir: str = "data/train_unlabeled/images/"
    val_image_dir: str = "data/val/images/"
    val_mask_dir: str = "data/val/masks/"
    img_size: List[int] = field(default_factory=lambda: [512, 512])
    labeled_num: int = 5
    num_classes: int = 2


@dataclass
class ModelConfig:
    specialist_encoder: str = "resnet34"
    specialist_decoder_channels: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32]
    )
    generalist_type: str = "vit_b"
    generalist_checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth"


@dataclass
class PromptingConfig:
    prompt_type: str = "box"  # box | point | mask | unc
    box_margin_px: int = 10
    box_merge_dist_px: int = 30
    box_jitter_scale: float = 0.1
    box_jitter_translate: int = 15
    num_prompt_perturbations: int = 4
    num_clicks: int = 10
    # Skeleton + point sampling (framework steps 5-7)
    skeleton_enabled: bool = True
    cc_min_area: int = 50
    cc_conf_threshold: float = 0.3
    prune_spur_px: int = 20
    pos_spacing_px: int = 15
    neg_offset_px: int = 20
    neg_subsample: int = 3
    point_subset_frac: float = 0.7


@dataclass
class UncertaintyConfig:
    upfm_enabled: bool = True
    boundary_band_width_px: int = 5
    w_boundary: float = 2.0
    w_interior: float = 1.0
    # Scale-equivariant uncertainty (Scale_equivariant.md Mod 2)
    scale_enabled: bool = False
    scale_list: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    scale_alpha: float = 1.0  # U_scale weight in fusion: w_ema = exp(-(entropy + alpha * U_scale))
    # T4 ablation flags
    use_detach: bool = True
    use_inverted_gating: bool = True
    use_learnable_fusion: bool = True


@dataclass
class TopologyConfig:
    enabled: bool = True
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    cl_dice_weight: float = 0.1
    cl_dice_interval: int = 1
    cl_dice_warmup: int = 200
    max_sam_queries: int = 5
    consultation_threshold: float = 0.0
    acceptance_u_geo: float = 0.3
    topo_alpha: float = 1.0
    topo_beta: float = 0.5
    morph_a: float = 1.0
    morph_b: float = 0.5


@dataclass
class TrainingConfig:
    # --- general ---
    max_iterations: int = 30000
    batch_size: int = 4
    labeled_bs: int = 2
    base_lr: float = 0.01
    ssl_lr: float = 1e-5  # lower LR for SSL stage (UnCoL convention)
    ema_decay: float = 0.99
    consistency: float = 0.1
    consistency_rampup: float = 200.0
    beta_sam: float = 0.1
    seed: int = 1337

    # --- two-stage (UnCoL) ---
    stage1_iterations: int = 6000  # foundation KD pretraining steps
    pretrain_checkpoint: str = ""  # path to stage-1 checkpoint (auto-saved if empty)

    # --- dual-teacher (UnCoL) ---
    dual_teacher_enabled: bool = True  # use dual-teacher fusion (SAM + EMA)
    # uncertainty fusion threshold: only trust pixels where u < threshold
    # threshold ramps from 0.75 → 1.0 over training
    fusion_uncertainty_threshold: float = 0.75

    # --- BCP CutMix (UnCoL) ---
    bcp_enabled: bool = True  # Bidirectional Copy-Paste mixing
    bcp_mask_ratio: float = 0.66  # CutMix mask proportion (2/3 in UnCoL)
    bcp_u_weight: float = 0.5  # weight of unlabeled pixels in mix loss


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            prompting=PromptingConfig(**raw.get("prompting", {})),
            uncertainty=UncertaintyConfig(**raw.get("uncertainty", {})),
            topology=TopologyConfig(**raw.get("topology", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save config back to YAML."""
        d = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "prompting": self.prompting.__dict__,
            "uncertainty": self.uncertainty.__dict__,
            "topology": self.topology.__dict__,
            "training": self.training.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)
