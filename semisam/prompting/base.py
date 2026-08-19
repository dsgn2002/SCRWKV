"""Point and mask prompting for SAM generalist.

Ported from SemiSAM+ utils/click_method.py — 3D EDT-based → 2D.
These are baseline prompt types kept for ablation comparisons.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# point prompting (ported from click_method.py)
# ---------------------------------------------------------------------------

def get_next_click2d_torch(
    prev_seg: torch.Tensor, gt_semantic_seg: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Random point sampling from FN/FP regions (ported from get_next_click3D_torch)."""
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > mask_threshold
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):
        fn_points = torch.nonzero(fn_masks[i, 0], as_tuple=False)
        fp_points = torch.nonzero(fp_masks[i, 0], as_tuple=False)

        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        else:
            _, h, w = fn_masks[i].shape
            point = torch.tensor([np.random.randint(h), np.random.randint(w)])
            is_positive = False

        bp = point.clone().detach().reshape(1, 1, 2).float()
        bl = torch.tensor([[int(is_positive)]])
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click2d_torch_ritm(
    prev_seg: torch.Tensor, gt_semantic_seg: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """EDT-based click: pick point at max distance from error boundary.

    Ported from get_next_click3D_torch_ritm — 3D EDT → 2D EDT.
    """
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > mask_threshold
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):
        fn_np = fn_masks[i, 0].cpu().numpy().astype(np.uint8)
        fp_np = fp_masks[i, 0].cpu().numpy().astype(np.uint8)

        # Pad for proper EDT at borders
        fn_padded = np.pad(fn_np, 1, mode="constant")
        fp_padded = np.pad(fp_np, 1, mode="constant")

        fn_dt = distance_transform_edt(1 - fn_padded)[1:-1, 1:-1]
        fp_dt = distance_transform_edt(1 - fp_padded)[1:-1, 1:-1]

        fn_max = fn_dt.max() if fn_np.any() else 0.0
        fp_max = fp_dt.max() if fp_np.any() else 0.0

        is_positive = fn_max > fp_max
        dt = fn_dt if is_positive else fp_dt
        threshold = max(fn_max, fp_max) / 2.0
        to_point_mask = dt > threshold

        points = np.argwhere(to_point_mask)
        if len(points) == 0:
            points = np.argwhere(dt >= dt.max())

        pt = points[np.random.randint(len(points))]  # (y, x)

        if fn_np[pt[0], pt[1]]:
            is_positive = True
        else:
            is_positive = False

        bp = torch.tensor([[pt[1], pt[0]]], dtype=torch.float32).reshape(1, 1, 2)
        bl = torch.tensor([[int(is_positive)]])
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click2d_torch_2(
    prev_seg: torch.Tensor, gt_semantic_seg: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Random click from any error region (ported from get_next_click3D_torch_2)."""
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > mask_threshold
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.nonzero(to_point_mask[i, 0], as_tuple=False)
        if len(points) == 0:
            _, h, w = fn_masks[i].shape
            pt = torch.tensor([np.random.randint(h), np.random.randint(w)])
        else:
            pt = points[np.random.randint(len(points))]

        if fn_masks[i, 0, pt[0], pt[1]]:
            is_positive = True
        else:
            is_positive = False

        bp = pt.clone().detach().reshape(1, 1, 2).float()
        bl = torch.tensor([[int(is_positive)]])
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels


# ---------------------------------------------------------------------------
# multi-click iterative prompting (ported from semisam_plus.py)
# ---------------------------------------------------------------------------

def prompt_sam_iterative(
    generalist,
    image_np: np.ndarray,
    prev_mask: torch.Tensor,
    gt_mask: torch.Tensor | None,
    num_clicks: int = 10,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Iterative point-click prompting for SAM.

    Ported from finetune_model_predict3D_unc / finetune_model_predict3D_point.

    Returns:
        (final_mask, uncertainty_map)
        uncertainty_map is None if gt_mask is None (not enough info for error sampling).
    """
    import torch

    all_preds = []
    seg_prob = torch.sigmoid(prev_mask)

    # Use gt for error-based click placement
    if gt_mask is not None:
        for num_click in range(num_clicks):
            if num_click == 0:
                click_fn = get_next_click2d_torch_ritm
            else:
                click_fn = get_next_click2d_torch_2

            batch_points, batch_labels = click_fn(seg_prob, gt_mask)

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            points_list = points_co.cpu().squeeze(1).numpy().tolist()
            labels_list = points_la.cpu().squeeze(1).numpy().tolist()

            mask = generalist.predict(image_np, points=points_list, point_labels=labels_list)
            seg_prob = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            all_preds.append(seg_prob.clone())

        final_mask = (seg_prob > 0.5).float()

        if len(all_preds) > 1:
            stacked = torch.stack([p.squeeze() for p in all_preds])  # (K, H, W)
            uncertainty = stacked.var(dim=0).cpu().numpy()  # (H, W)
        else:
            uncertainty = np.zeros(image_np.shape[:2], dtype=np.float32)

        return final_mask.squeeze().cpu().numpy().astype(np.uint8), uncertainty

    else:
        # No gt — just run one-pass with mask prompt
        mask_np = (torch.sigmoid(prev_mask).squeeze().cpu().numpy() > 0.5).astype(np.float32)
        final_mask = generalist.predict(image_np, mask_prompt=mask_np)
        return final_mask, None
