"""Boundary band mask computation (CLAUDE.md §5, Modification 2, step 5)."""

import cv2
import numpy as np
import torch


def boundary_band_mask(
    pred_mask: torch.Tensor,
    width_k: int = 5,
) -> torch.Tensor:
    """Compute boundary band = dilate(contour, width_k) as a binary mask.

    Args:
        pred_mask: (B, 1, H, W) float tensor (logits or probs, thresholded at 0.5).
        width_k: boundary band half-width in pixels.

    Returns:
        (B, 1, H, W) float tensor {0, 1}: 1 = boundary region.
    """
    batch_size = pred_mask.shape[0]
    device = pred_mask.device
    bands = []

    for b in range(batch_size):
        mask_np = (pred_mask[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if mask_np.sum() == 0:
            bands.append(torch.zeros_like(pred_mask[b:b + 1]))
            continue

        # Find contours
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contour on empty canvas
        boundary = np.zeros_like(mask_np, dtype=np.uint8)
        cv2.drawContours(boundary, contours, -1, 1, thickness=1)

        # Dilate to create band
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * width_k + 1, 2 * width_k + 1)
        )
        band = cv2.dilate(boundary, kernel, iterations=1)

        bands.append(
            torch.from_numpy(band.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    return torch.cat(bands, dim=0)
