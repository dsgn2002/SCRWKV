"""Topology-aware losses: soft clDice computed directly from mask probabilities."""
import torch
import torch.nn.functional as F


def soft_skel(pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Differentiable skeleton via min-pool: ReLU(pred - minpool(pred))."""
    mp = -F.max_pool2d(-pred, kernel_size=3, stride=1, padding=1)
    skel = torch.relu(pred - mp)
    return skel


def soft_cl_dice_loss(
    pred_mask: torch.Tensor,   # (B, 1, H, W) sigmoid probabilities
    gt_mask: torch.Tensor,     # (B, 1, H, W) binary
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Differentiable clDice loss from mask_prob and gt_mask directly.

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    Tprec = |soft_skel(pred) ∩ gt| / |soft_skel(pred)|
    Tsens = |pred ∩ soft_skel(gt)| / |soft_skel(gt)|
    """
    pred_skel_soft = soft_skel(pred_mask)
    gt_skel = soft_skel(gt_mask)  # gt is binary, soft_skel ≈ hard skeleton

    tprec = (pred_skel_soft * gt_mask).sum(dim=(1, 2, 3))
    tprec = tprec / (pred_skel_soft.sum(dim=(1, 2, 3)) + smooth)

    tsens = (pred_mask * gt_skel).sum(dim=(1, 2, 3))
    tsens = tsens / (gt_skel.sum(dim=(1, 2, 3)) + smooth)

    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return 1.0 - cl_dice.mean()
