"""Supervised losses for segmentation.

Ported from SemiSAM+ utils/losses.py + new heteroscedastic loss (CLAUDE.md §5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ported from original
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Multi-class Dice loss — ported verbatim from SemiSAM+ utils/losses.py."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), (
            f"predict {inputs.shape} & target {target.shape} shape do not match"
        )
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def dice_loss_binary(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary Dice loss (for single-channel crack masks).

    score: (B, 1, H, W) logits or probs
    target: (B, 1, H, W) binary
    """
    score = torch.sigmoid(score)
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1.0 - dice


def bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE + Dice for binary segmentation. Returns (loss_bce, loss_dice)."""
    loss_bce = F.binary_cross_entropy_with_logits(logits, target)
    loss_dice = dice_loss_binary(logits, target)
    return loss_bce, loss_dice


# ---------------------------------------------------------------------------
# heteroscedastic loss (CLAUDE.md §5, Modification 2, step 2)
# ---------------------------------------------------------------------------

def heteroscedastic_loss(
    mask_logits: torch.Tensor,
    mask_gt: torch.Tensor,
    log_sigma2: torch.Tensor,
) -> torch.Tensor:
    """Kendall & Gal style heteroscedastic regression loss.

    L = 0.5 * exp(-log_sigma2) * L_mask  +  0.5 * log_sigma2

    Where L_mask = BCE + Dice (per-pixel before spatial reduction).
    The uncertainty term log_sigma2 gates the mask loss: high uncertainty
    pixels contribute less to the supervised loss, and the 0.5*log_sigma2
    term prevents the trivial solution of sigma2 → +inf.

    Args:
        mask_logits: (B, 1, H, W) raw logits from specialist.
        mask_gt: (B, 1, H, W) binary ground truth.
        log_sigma2: (B, 1, H, W) predicted log-variance.

    Returns:
        Scalar loss (mean over batch and spatial dims).
    """
    # Per-pixel BCE
    bce_per_pixel = F.binary_cross_entropy_with_logits(
        mask_logits, mask_gt, reduction="none"
    )  # (B, 1, H, W)

    # Per-pixel "soft Dice" — use sigmoid probs
    probs = torch.sigmoid(mask_logits)
    smooth = 1e-5
    kernel_size = 7
    avg_probs = F.avg_pool2d(probs, kernel_size, stride=1, padding=kernel_size // 2)
    avg_gt = F.avg_pool2d(mask_gt, kernel_size, stride=1, padding=kernel_size // 2)
    intersect = avg_probs * mask_gt
    union = avg_probs + mask_gt - avg_probs * mask_gt
    dice_per_pixel = 1.0 - (2.0 * intersect + smooth) / (union + avg_gt + smooth)

    # Combined mask loss per pixel
    mask_loss = (bce_per_pixel + dice_per_pixel) / 2.0  # (B, 1, H, W)

    # Heteroscedastic weighting — clamp log_sigma2 to prevent exp explosion
    log_sigma2 = torch.clamp(log_sigma2, min=-10.0, max=10.0)
    inv_unc = torch.exp(-log_sigma2)  # precision, now safe
    loss = 0.5 * inv_unc * mask_loss + 0.5 * log_sigma2

    return loss.mean()


# ---------------------------------------------------------------------------
# Focal Loss (ported)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Ported verbatim from SemiSAM+ utils/losses.py."""

    def __init__(self, gamma=2, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.detach()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
