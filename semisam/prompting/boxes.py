"""Box prompt generation from specialist coarse masks (Modification 1).

Implements CLAUDE.md §4:
  boxes_from_mask() → connected-component bounding boxes
  jitter_boxes()    → n perturbed box variants for uncertainty estimation
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch


def boxes_from_mask(
    pred_mask: torch.Tensor,
    min_area: int = 50,
    margin_px: int = 10,
    merge_dist_px: int = 30,
    conf_threshold: float = 0.3,
) -> list[list[float]]:
    """Extract bounding boxes from binary mask via connected-component analysis.

    Args:
        pred_mask: (B, 1, H, W) tensor of logits or probabilities.
        min_area: minimum component area in pixels (ignore noise).
        margin_px: pixels to dilate each box.
        merge_dist_px: merge boxes closer than this distance.
        conf_threshold: discard CCs with mean confidence below this.

    Returns:
        List of boxes as [x1, y1, x2, y2] in pixel coordinates.
    """
    # Threshold to binary, keep probability for confidence filtering
    prob = torch.sigmoid(pred_mask).detach().cpu().squeeze().numpy()
    binary = (prob > 0.5).astype(np.uint8)
    _, h, w = (prob.shape[0] if prob.ndim == 3 else 1, prob.shape[0], prob.shape[1])
    if prob.ndim == 3:  # batch: take first
        binary = binary[0]
        prob = prob[0]
        h, w = binary.shape

    if binary.sum() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        # Mean confidence within this CC
        cc_mask = labels == i
        mean_conf = prob[cc_mask].mean()
        if mean_conf < conf_threshold:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(w, x + bw + margin_px)
        y2 = min(h, y + bh + margin_px)

        boxes.append([float(x1), float(y1), float(x2), float(y2)])

    boxes = _merge_boxes(boxes, merge_dist_px)
    return boxes


def jitter_boxes(
    boxes: list[list[float]],
    n: int = 5,
    scale_jitter: float = 0.1,
    translate_jitter: int = 15,
    img_size: tuple[int, int] = (512, 512),
) -> list[list[list[float]]]:
    """Generate `n` perturbed box sets for uncertainty estimation.

    Args:
        boxes: base boxes from boxes_from_mask().
        n: number of perturbed variants.
        scale_jitter: random scale variation fraction (e.g., 0.1 = ±10%).
        translate_jitter: max pixel translation per box.
        img_size: (H, W) for clamping.

    Returns:
        List of `n` box-lists, each with same structure as input `boxes`.
        Shape: [n][n_boxes][4]
    """
    if len(boxes) == 0:
        return [[[]] for _ in range(n)]

    h, w = img_size
    jittered = []

    for _ in range(n):
        variant = []
        for (x1, y1, x2, y2) in boxes:
            bw = x2 - x1
            bh = y2 - y1

            # Random scale
            scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            new_bw = bw * scale
            new_bh = bh * scale

            # Random translation
            dx = random.randint(-translate_jitter, translate_jitter)
            dy = random.randint(-translate_jitter, translate_jitter)

            nx1 = cx - new_bw / 2.0 + dx
            ny1 = cy - new_bh / 2.0 + dy
            nx2 = cx + new_bw / 2.0 + dx
            ny2 = cy + new_bh / 2.0 + dy

            # Clamp
            nx1 = max(0.0, min(float(w), nx1))
            ny1 = max(0.0, min(float(h), ny1))
            nx2 = max(0.0, min(float(w), nx2))
            ny2 = max(0.0, min(float(h), ny2))

            # ponytail: prevent zero-area boxes (collapsed at image edge)
            if nx2 - nx1 < 1.0:
                nx2 = min(float(w), nx1 + 1.0)
                nx1 = max(0.0, nx2 - 1.0)
            if ny2 - ny1 < 1.0:
                ny2 = min(float(h), ny1 + 1.0)
                ny1 = max(0.0, ny2 - 1.0)

            variant.append([nx1, ny1, nx2, ny2])

        jittered.append(variant)

    return jittered


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _merge_boxes(
    boxes: list[list[float]], merge_dist: int
) -> list[list[float]]:
    """Merge boxes that are within `merge_dist` pixels of each other."""
    if len(boxes) <= 1:
        return boxes

    # Simple greedy merge: sort by x1, merge overlapping/enclosed-with-margin
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    current = sorted_boxes[0][:]

    for box in sorted_boxes[1:]:
        x1, y1, x2, y2 = box
        cx1, cy1, cx2, cy2 = current

        # Check if boxes overlap (with margin)
        if (
            x1 <= cx2 + merge_dist
            and cx1 <= x2 + merge_dist
            and y1 <= cy2 + merge_dist
            and cy1 <= y2 + merge_dist
        ):
            # Merge: union of both boxes
            current[0] = min(cx1, x1)
            current[1] = min(cy1, y1)
            current[2] = max(cx2, x2)
            current[3] = max(cy2, y2)
        else:
            merged.append(current)
            current = box[:]

    merged.append(current)
    return merged
