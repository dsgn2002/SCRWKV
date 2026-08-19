"""Dual-teacher pseudo-label fusion and BCP CutMix (UnCoL / SemiSAM-Crack).

Adapted from:
  https://github.com/VivienLu/UnCoL (train_2d.py, utils/BCP_utils.py)
  Lu et al., "Harmonizing Generalization and Specialization:
  Uncertainty-Informed Collaborative Learning for Semi-supervised
  Medical Image Segmentation", IEEE TMI, 2026.

Key components:
  1. entropy() — per-pixel predictive entropy (both teachers)
  2. fuse_pseudo_labels() — uncertainty-weighted dual-teacher fusion
  3. generate_bcp_mask() — CutMix mask for BCP augmentation
  4. masked_mix_loss() — uncertainty-gated Dice+CE on mixed images
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# entropy-based uncertainty (UnCoL §3.3)
# ---------------------------------------------------------------------------

def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel predictive entropy, normalized to [0, 1].

    Args:
        probs: (B, C, H, W) softmax probabilities (or sigmoid for binary).
        eps: numerical stability.

    Returns:
        (B, 1, H, W) normalized entropy ∈ [0, 1].
    """
    C = probs.shape[1]
    if C == 1:
        # Binary: convert to 2-class [p, 1-p]
        p = torch.clamp(probs, eps, 1.0 - eps)
        entropy_map = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        entropy_max = math.log(2)
    else:
        p = torch.clamp(probs, eps, 1.0 - eps)
        entropy_map = -torch.sum(p * torch.log(p), dim=1, keepdim=True)
        entropy_max = math.log(C)
    return entropy_map / (entropy_max + eps)


def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy from raw logits (applies sigmoid for binary, softmax for multi-class).

    Args:
        logits: (B, C, H, W) raw logits.
        eps: numerical stability.

    Returns:
        (B, 1, H, W) normalized entropy ∈ [0, 1].
    """
    C = logits.shape[1]
    if C == 1:
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=1)
    return entropy(probs, eps=eps)


# ---------------------------------------------------------------------------
# dual-teacher pseudo-label fusion (UnCoL Eq. 5-7)
# ---------------------------------------------------------------------------

def fuse_pseudo_labels(
    p_foundation: torch.Tensor,
    p_teacher: torch.Tensor,
    u_foundation: torch.Tensor,
    u_teacher: torch.Tensor,
    uncertainty_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uncertainty-weighted fusion of foundation (SAM) and specialist (EMA) predictions.

    Adapted from UnCoL's fuse_pseudo_label_with_mask_fast().

    The fusion rule:
      - w_i = exp(-u_i)  (lower uncertainty → higher weight)
      - w_i /= (w_foundation + w_teacher)  (normalize)
      - fused = w_foundation * p_foundation + w_teacher * p_teacher
      - trust_mask = (u_foundation ≤ threshold) OR (u_teacher ≤ threshold)

    For binary segmentation (C=1), we fuse the probability maps directly.
    For multi-class (C>1), we fuse per-class and argmax for discrete labels.

    Args:
        p_foundation: (B, C, H, W) foundation model probabilities.
        p_teacher:    (B, C, H, W) specialist EMA probabilities.
        u_foundation: (B, 1, H, W) foundation uncertainty [0, 1].
        u_teacher:    (B, 1, H, W) specialist uncertainty [0, 1].
        uncertainty_threshold: pixels with u ≤ this are trusted.

    Returns:
        fused_probs: (B, C, H, W) fused probability map.
        trust_mask:  (B, 1, H, W) {0, 1} — 1 = at least one teacher confident.
    """
    # Trust masks
    trust_foundation = (u_foundation <= uncertainty_threshold).float()
    trust_teacher = (u_teacher <= uncertainty_threshold).float()
    trust_mask = ((trust_foundation + trust_teacher) > 0).float()

    # Uncertainty → weight via exp(-u)
    w_f = torch.exp(-u_foundation)
    w_t = torch.exp(-u_teacher)
    w_sum = w_f + w_t + 1e-8

    # Both-trust, only-teacher, only-foundation regions
    both_trust = trust_foundation * trust_teacher
    only_teacher = (1.0 - trust_foundation) * trust_teacher
    only_foundation = trust_foundation * (1.0 - trust_teacher)

    # Weighted average (normalized)
    w_f_norm = w_f / w_sum
    w_t_norm = 1.0 - w_f_norm
    fused = w_f_norm * p_foundation + w_t_norm * p_teacher

    # In regions where only one teacher is trusted, use that teacher directly
    pseudo_fused = (
        both_trust * fused
        + only_teacher * p_teacher
        + only_foundation * p_foundation
    )

    return pseudo_fused, trust_mask


def fuse_pseudo_labels_binary(
    p_sam: torch.Tensor,
    p_ema: torch.Tensor,
    u_sam: torch.Tensor,
    u_ema: torch.Tensor,
    uncertainty_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for binary (C=1) fusion — calls fuse_pseudo_labels.

    Args:
        p_sam: (B, 1, H, W) SAM probabilities.
        p_ema: (B, 1, H, W) EMA teacher probabilities.
        u_sam: (B, 1, H, W) SAM combined uncertainty [0, 1].
        u_ema: (B, 1, H, W) EMA entropy uncertainty [0, 1].
        uncertainty_threshold: pixels with u ≤ this are trusted.

    Returns:
        fused_probs: (B, 1, H, W) fused probability map.
        trust_mask:  (B, 1, H, W) {0, 1}.
    """
    return fuse_pseudo_labels(p_sam, p_ema, u_sam, u_ema, uncertainty_threshold)


# ---------------------------------------------------------------------------
# BCP CutMix augmentation (UnCoL / BCP)
# ---------------------------------------------------------------------------

def generate_bcp_mask(
    height: int, width: int, mask_ratio: float = 0.66
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random rectangular CutMix mask (2D version of UnCoL's generate_mask).

    Args:
        height, width: spatial dimensions.
        mask_ratio: proportion of image to mask out.

    Returns:
        mask:      (H, W) tensor, 1=keep, 0=replace.
        loss_mask: (1, H, W) tensor, 1=compute loss, 0=ignore.
    """
    patch_h = int(height * mask_ratio)
    patch_w = int(width * mask_ratio)
    w = np.random.randint(0, max(1, height - patch_h))
    h = np.random.randint(0, max(1, width - patch_w))

    mask = torch.ones(height, width)
    mask[w : w + patch_h, h : h + patch_w] = 0

    loss_mask = torch.ones(1, height, width)
    loss_mask[:, w : w + patch_h, h : h + patch_w] = 0

    return mask, loss_mask


# ---------------------------------------------------------------------------
# BCP masked mix loss (UnCoL / BCP)
# ---------------------------------------------------------------------------

def masked_mix_loss(
    student_output: torch.Tensor,
    img_label: torch.Tensor,
    patch_label: torch.Tensor,
    mask: torch.Tensor,
    trust_mask: torch.Tensor | None = None,
    u_weight: float = 0.5,
    unlab: bool = False,
) -> torch.Tensor:
    """Dice + CE/BCE loss on CutMix-mixed images, gated by trust mask.

    Adapted from UnCoL's masked_mix_loss() in utils/BCP_utils.py.
    Handles both binary (C=1) and multi-class (C>1) outputs.

    Args:
        student_output: (B, C, H, W) student logits.
        img_label:      (B, H, W) or (B, 1, H, W) long label for img region.
        patch_label:    (B, H, W) or (B, 1, H, W) long label for patch region.
        mask:           (H, W) or (B, H, W) CutMix mask, 1=img region.
        trust_mask:     (B, 1, H, W) optional trust mask {0, 1}.
        u_weight:       weight for unlabeled-side loss.
        unlab:          if True, the "img" side is unlabeled (lower weight).

    Returns:
        Scalar loss.
    """
    B, C, H, W = student_output.shape
    binary = C == 1

    # Ensure correct shapes
    img_label = img_label.long()
    patch_label = patch_label.long()
    if img_label.dim() == 4:
        img_label = img_label.squeeze(1)
    if patch_label.dim() == 4:
        patch_label = patch_label.squeeze(1)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).expand(B, H, W)
    patch_mask = 1.0 - mask

    # Apply trust mask if provided
    if trust_mask is not None:
        if trust_mask.shape[1] == 1:
            trust_mask_sq = trust_mask.squeeze(1)
        else:
            trust_mask_sq = trust_mask
        if unlab:
            mask = mask * trust_mask_sq
        else:
            patch_mask = patch_mask * trust_mask_sq

    img_weight, patch_weight = (1.0, u_weight)
    if unlab:
        img_weight, patch_weight = (u_weight, 1.0)

    # --- Dice loss (masked) ---
    if binary:
        dice_loss = _masked_dice_loss_binary(
            student_output, img_label, mask
        ) * img_weight
        dice_loss += _masked_dice_loss_binary(
            student_output, patch_label, patch_mask
        ) * patch_weight
    else:
        dice_loss = _masked_dice_loss(
            student_output, img_label, mask, n_classes=C
        ) * img_weight
        dice_loss += _masked_dice_loss(
            student_output, patch_label, patch_mask, n_classes=C
        ) * patch_weight

    # --- Pixel loss (masked) ---
    if binary:
        # Binary: BCEWithLogitsLoss per-pixel, masked
        bce = F.binary_cross_entropy_with_logits(
            student_output,
            img_label.unsqueeze(1).float(),
            reduction="none",
        )  # (B, 1, H, W)
        bce = bce.squeeze(1)  # (B, H, W)
        pixel_loss_img = (bce * mask).sum() / (mask.sum() + 1e-16)
        pixel_loss_patch = (bce * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    else:
        ce = F.cross_entropy(student_output, img_label, reduction="none")
        pixel_loss_img = (ce * mask).sum() / (mask.sum() + 1e-16)
        pixel_loss_patch = (ce * patch_mask).sum() / (patch_mask.sum() + 1e-16)

    loss_pixel = img_weight * pixel_loss_img + patch_weight * pixel_loss_patch

    return (dice_loss + loss_pixel) / 2.0


def _masked_dice_loss_binary(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Binary Dice loss on masked region.

    Args:
        logits: (B, 1, H, W) logits.
        target: (B, H, W) long {0, 1}.
        mask:   (B, H, W) float.

    Returns:
        Scalar dice loss.
    """
    probs = torch.sigmoid(logits)  # (B, 1, H, W)
    target_f = target.unsqueeze(1).float()  # (B, 1, H, W)
    m = mask.unsqueeze(1)  # (B, 1, H, W)
    smooth = 1e-5

    intersect = torch.sum(probs * target_f * m)
    p_sum = torch.sum(probs * m)
    t_sum = torch.sum(target_f * m)
    dice = (2.0 * intersect + smooth) / (p_sum + t_sum + smooth)
    return 1.0 - dice


def _masked_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Dice loss computed only on masked region.

    Args:
        logits: (B, C, H, W).
        target: (B, H, W) long.
        mask:   (B, H, W) float.
        n_classes: number of classes.

    Returns:
        Scalar dice loss.
    """
    probs = torch.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes).permute(
        0, 3, 1, 2
    ).float()  # (B, C, H, W)
    smooth = 1e-5

    mask = mask.unsqueeze(1)  # (B, 1, H, W)
    total_loss = 0.0
    for c in range(n_classes):
        p_c = probs[:, c : c + 1]
        t_c = target_onehot[:, c : c + 1]
        intersect = torch.sum(p_c * t_c * mask)
        p_sum = torch.sum(p_c * p_c * mask)
        t_sum = torch.sum(t_c * t_c * mask)
        dice_c = (2.0 * intersect + smooth) / (p_sum + t_sum + smooth)
        total_loss += 1.0 - dice_c

    return total_loss / n_classes
