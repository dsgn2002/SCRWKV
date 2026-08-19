"""Evaluation metrics for 2D crack segmentation.

Ported from SemiSAM+ code_semisam+/utils/metrics.py and test_3D_util.py.
medpy dependency removed — using scipy for Hausdorff distance.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """Binary Dice coefficient."""
    pred = pred.astype(np.float32).reshape(-1)
    gt = gt.astype(np.float32).reshape(-1)
    intersection = (pred * gt).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """Binary IoU / Jaccard."""
    pred = pred.astype(np.float32).reshape(-1)
    gt = gt.astype(np.float32).reshape(-1)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile Hausdorff distance (in pixels)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0 if pred.sum() == gt.sum() else float("inf")

    # Distance from pred boundary to gt boundary
    pred_border = pred ^ _erode(pred)
    gt_border = gt ^ _erode(gt)

    dt_pred_to_gt = distance_transform_edt(~gt_border)
    dt_gt_to_pred = distance_transform_edt(~pred_border)

    dist_pred = dt_pred_to_gt[pred_border]
    dist_gt = dt_gt_to_pred[gt_border]
    all_dists = np.concatenate([dist_pred, dist_gt])

    return float(np.percentile(all_dists, 95))


def asd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Surface Distance."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0 if pred.sum() == gt.sum() else float("inf")

    pred_border = pred ^ _erode(pred)
    gt_border = gt ^ _erode(gt)

    dt_pred_to_gt = distance_transform_edt(~gt_border)
    dt_gt_to_pred = distance_transform_edt(~pred_border)

    return float(
        (dt_pred_to_gt[pred_border].mean() + dt_gt_to_pred[gt_border].mean()) / 2.0
    )


def boundary_iou(pred: np.ndarray, gt: np.ndarray, width: int = 5) -> float:
    """Boundary IoU: IoU computed only within a boundary band of `width` pixels."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    gt_boundary = _boundary_band(gt, width)
    if gt_boundary.sum() == 0:
        return 1.0

    intersection = (pred & gt & gt_boundary).sum()
    union = ((pred | gt) & gt_boundary).sum()
    return float(intersection / union) if union > 0 else 0.0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, width: int = 5) -> float:
    """Boundary F-score: F1 computed on boundary pixels."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    gt_boundary = _boundary_band(gt, width)
    if gt_boundary.sum() == 0:
        return 1.0

    tp = (pred & gt & gt_boundary).sum()
    fp = (pred & ~gt & gt_boundary).sum()
    fn = (~pred & gt & gt_boundary).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return [dice, iou, hd95, asd] for a single case."""
    return np.array(
        [
            dice_score(pred, gt),
            iou_score(pred, gt),
            hd95(pred, gt),
            asd(pred, gt),
        ]
    )


# ---------------------------------------------------------------------------
# clDice — topology-aware metric for crack detection
# ---------------------------------------------------------------------------

def cl_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """Centerline Dice: topology-preservation metric.

    clDice = 2 * (T_prec * T_sens) / (T_prec + T_sens)

    where:
      T_prec(pred, gt) = |pred ∩ skeleton(gt)| / |pred|     (topology precision)
      T_sens(pred, gt) = |skeleton(pred) ∩ gt| / |skeleton(pred)|  (topology sensitivity)

    Primary metric for OmniCrack30k protocol.
    """
    from skimage.morphology import skeletonize

    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    # Topology precision: fraction of pred that lies on GT skeleton
    gt_skel = skeletonize(gt_bin)
    t_prec = (pred_bin & gt_skel).sum() / (pred_bin.sum() + smooth)

    # Topology sensitivity: fraction of pred skeleton that lies on GT
    pred_skel = skeletonize(pred_bin)
    t_sens = (pred_skel & gt_bin).sum() / (pred_skel.sum() + smooth)

    return float(2.0 * t_prec * t_sens / (t_prec + t_sens + smooth))


# --- helpers ---

def _erode(mask: np.ndarray) -> np.ndarray:
    """Single-pixel binary erosion."""
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask)


def _dilate(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Binary dilation by `width` pixels."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=width)


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    """Boundary band = dilation(mask, width) - erosion(mask, width)."""
    outer = _dilate(mask, width)
    inner = _erode(mask)  # single-pixel erosion for inner contour
    # ponytail: close is good enough — per-pixel refinement if needed
    return outer.astype(np.uint8) ^ inner.astype(np.uint8)
