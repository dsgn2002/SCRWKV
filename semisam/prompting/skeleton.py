"""Skeleton extraction + positive/negative point sampling from coarse masks.

Framework steps 5-7: skeletonize each CC, prune spurs, sample points.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize


def skeletonize_mask(
    mask: np.ndarray,
    prune_spur_px: int = 20,
) -> list[np.ndarray]:
    """Skeletonize a binary mask per CC, prune short spurs.

    Args:
        mask: (H, W) binary uint8 mask.
        prune_spur_px: remove skeleton branches shorter than this.

    Returns:
        List of (N_i, 2) arrays of (y, x) skeleton pixel coords per CC.
    """
    skel = skeletonize(mask.astype(bool))
    # ponytail: prune by iteratively removing endpoints shorter than prune_spur_px
    skel = _prune_spurs(skel, prune_spur_px)

    # Label individual CCs on the skeleton
    from scipy.ndimage import label as nd_label

    labeled, num = nd_label(skel)
    skeletons = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)  # (N, 2) in (y, x)
        if len(coords) >= 2:
            skeletons.append(coords)
    return skeletons


def sample_positive_points(
    skeletons: list[np.ndarray],
    spacing_px: int = 15,
) -> list[np.ndarray]:
    """Sample foreground points along skeleton arc length.

    Args:
        skeletons: list of (N_i, 2) skeleton coords per CC.
        spacing_px: sample every N pixels along the arc.

    Returns:
        List of (M_i, 2) arrays of (y, x) positive point coords per CC.
    """
    all_points = []
    for skel in skeletons:
        if len(skel) < 2:
            if len(skel) == 1:
                all_points.append(skel)
            continue

        # ponytail: greedy arc-length sampling
        sampled = [skel[0]]
        last = skel[0]
        dist_acc = 0.0
        for pt in skel[1:]:
            dist_acc += np.linalg.norm(pt.astype(float) - last.astype(float))
            if dist_acc >= spacing_px:
                sampled.append(pt)
                dist_acc = 0.0
            last = pt
        # Always include the last point
        if not np.array_equal(sampled[-1], skel[-1]):
            sampled.append(skel[-1])
        all_points.append(np.array(sampled))
    return all_points


def sample_negative_points(
    pos_points: list[np.ndarray],
    skeletons: list[np.ndarray],
    offset_px: int = 20,
    mask_shape: tuple[int, int] | None = None,
    pos_mask: np.ndarray | None = None,
    subsample: int = 3,
) -> list[np.ndarray]:
    """Sample background points perpendicular to skeleton at positive points.

    Args:
        pos_points: per-CC list of (M_i, 2) positive point coords.
        skeletons: per-CC skeleton coords (for tangent estimation).
        offset_px: perpendicular offset distance.
        mask_shape: (H, W) for bounds clamping.
        pos_mask: (H, W) binary mask of all CC regions to avoid.
        subsample: only compute negatives every N positives (speed).

    Returns:
        List of (K_i, 2) arrays of (y, x) negative point coords per CC.
    """
    all_neg = []
    for cc_idx, (pp, skel) in enumerate(zip(pos_points, skeletons)):
        negs = []
        for i, pt in enumerate(pp):
            if i % subsample != 0:
                continue
            tangent = _local_tangent(skel, pt)
            if tangent is None:
                continue
            normal = np.array([-tangent[0], tangent[1]])  # perpendicular

            for sign in [-1, 1]:
                neg = pt.astype(float) + sign * offset_px * normal
                neg = neg.astype(int)
                if mask_shape is not None:
                    neg[0] = np.clip(neg[0], 0, mask_shape[0] - 1)
                    neg[1] = np.clip(neg[1], 0, mask_shape[1] - 1)
                # Reject if inside any positive CC
                if pos_mask is not None and pos_mask[neg[0], neg[1]]:
                    continue
                negs.append(neg)
        if negs:
            all_neg.append(np.array(negs))
        else:
            all_neg.append(np.zeros((0, 2), dtype=int))
    return all_neg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prune_spurs(skel: np.ndarray, min_length: int) -> np.ndarray:
    """Iteratively remove skeleton branches shorter than min_length pixels."""
    import cv2

    skel_u8 = skel.astype(np.uint8)

    for _ in range(5):  # ponytail: fixed iterations, enough for crack spurs
        # Find endpoints (pixels with exactly 1 neighbor)
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbors = cv2.filter2D(skel_u8, -1, kernel) * skel_u8
        endpoints = (neighbors == 2) & (skel_u8 > 0)  # 1 neighbor + self = 2

        if not endpoints.any():
            break

        # Trace from each endpoint, prune if branch < min_length
        endpoint_coords = np.argwhere(endpoints)
        for ey, ex in endpoint_coords:
            branch = _trace_branch(skel_u8, ey, ex)
            if 0 < len(branch) <= min_length:
                for by, bx in branch:
                    skel_u8[by, bx] = 0

    return skel_u8.astype(bool)


def _trace_branch(skel: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    """Trace from an endpoint to the nearest junction. Returns list of (y, x)."""
    h, w = skel.shape
    branch = [(y, x)]
    visited = {(y, x)}
    cy, cx = y, x
    for _ in range(200):  # max trace length
        nbrs = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    nbrs.append((ny, nx))
        unvisited = [n for n in nbrs if n not in visited]
        if len(unvisited) == 1:
            cy, cx = unvisited[0]
            visited.add((cy, cx))
            branch.append((cy, cx))
        else:
            break  # junction or dead end
    return branch


def _local_tangent(
    skeleton: np.ndarray, pt: np.ndarray, window: int = 10
) -> np.ndarray | None:
    """Estimate local tangent at pt on skeleton using a window of nearby points."""
    dists = np.linalg.norm(skeleton.astype(float) - pt.astype(float), axis=1)
    nearby = skeleton[dists < window]
    if len(nearby) < 3:
        return None
    # ponytail: PCA on nearby points for tangent direction
    centered = nearby.astype(float) - nearby.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]  # first principal component
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    return tangent
