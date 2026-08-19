"""U_geo: box-perturbation disagreement uncertainty (CLAUDE.md §5)."""

from __future__ import annotations

import numpy as np
import torch


def compute_u_geo(
    generalist,
    image_np: np.ndarray,
    jittered_boxes: list[list[list[float]]],
    device: str = "cuda",
) -> torch.Tensor:
    """Compute U_geo from SAM predictions across perturbed box prompts.

    Ported from compute_epistemic_uncertainty in semisam_plus.py,
    adapted for box perturbations instead of iterative point clicks.

    Args:
        generalist: GeneralistWrapper instance.
        image_np: (H, W, 3) numpy uint8 image.
        jittered_boxes: list of n box-lists from jitter_boxes().
        device: target device for output tensor.

    Returns:
        (1, 1, H, W) float tensor, pixel-wise uncertainty [0, 1].
    """
    final_mask, unc_np = generalist.predict_unc(image_np, jittered_boxes)
    return torch.from_numpy(unc_np).unsqueeze(0).unsqueeze(0).float().to(device)
