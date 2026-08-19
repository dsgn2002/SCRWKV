"""Topology metrics for crack segmentation evaluation."""
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import label, distance_transform_edt


def skeleton_precision(pred: np.ndarray, gt: np.ndarray, width: int = 2) -> float:
    """Fraction of pred skeleton within `width` pixels of GT skeleton."""
    pred_skel = skeletonize(pred > 0.5)
    gt_skel = skeletonize(gt > 0.5)
    if pred_skel.sum() == 0:
        return 1.0 if gt_skel.sum() == 0 else 0.0
    dist = distance_transform_edt(~gt_skel)
    matched = (pred_skel & (dist <= width)).sum()
    return float(matched / pred_skel.sum())


def skeleton_recall(pred: np.ndarray, gt: np.ndarray, width: int = 2) -> float:
    """Fraction of GT skeleton within `width` pixels of pred skeleton."""
    pred_skel = skeletonize(pred > 0.5)
    gt_skel = skeletonize(gt > 0.5)
    if gt_skel.sum() == 0:
        return 1.0
    dist = distance_transform_edt(~pred_skel)
    matched = (gt_skel & (dist <= width)).sum()
    return float(matched / gt_skel.sum())


def endpoint_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean distance from each pred endpoint to nearest GT endpoint."""
    pred_skel = skeletonize(pred > 0.5)
    gt_skel = skeletonize(gt > 0.5)

    # Find endpoints: pixels with exactly 1 neighbor
    def _endpoints(skel):
        from scipy.ndimage import convolve
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
        return (skel & (neighbors == 1))

    pred_eps = _endpoints(pred_skel)
    gt_eps = _endpoints(gt_skel)

    if pred_eps.sum() == 0 or gt_eps.sum() == 0:
        return 0.0

    pred_coords = np.argwhere(pred_eps)
    gt_coords = np.argwhere(gt_eps)

    distances = []
    for pc in pred_coords:
        d = np.min(np.sqrt(((gt_coords - pc) ** 2).sum(axis=1)))
        distances.append(d)

    return float(np.mean(distances))


def fragmentation_count(mask: np.ndarray) -> int:
    """Number of connected components in the prediction."""
    _, n = label(mask > 0.5)
    return n
