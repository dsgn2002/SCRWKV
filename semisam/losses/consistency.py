"""Consistency losses for semi-supervised learning (Mean Teacher).

Ported from SemiSAM+ SemiSAM_train_MT_3D.py.
"""

import torch
import torch.nn.functional as F


def mse_consistency_loss(
    student_pred: torch.Tensor, teacher_pred: torch.Tensor
) -> torch.Tensor:
    """MSE consistency loss between student and teacher predictions.

    Args:
        student_pred: (B, C, H, W) student softmax/sigmoid probs.
        teacher_pred: (B, C, H, W) teacher softmax/sigmoid probs (detached).

    Returns:
        Scalar loss (mean over all dims).
    """
    return torch.mean((student_pred - teacher_pred) ** 2)


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss.

    Ported verbatim from SemiSAM+ utils/losses.py.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    return (input_softmax - target_softmax) ** 2
