"""Confidence-aware consistency loss (CLAUDE.md §5, Modification 2, step 5).

Boundary-split L_sam: consistency loss gated by uncertainty, with separate
boundary and interior weighting.
"""

import torch


def confidence_aware_loss(
    student_pred: torch.Tensor,
    sam_pseudo: torch.Tensor,
    u_geo: torch.Tensor,
    u_app: torch.Tensor | None,
    boundary_mask: torch.Tensor,
    w_boundary: float = 2.0,
    w_interior: float = 1.0,
    agreement_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Boundary-split, uncertainty-gated consistency loss vs SAM pseudo-labels.

    L_sam = w_b * mean[(1-U) * L_consistency]_{B}  +  w_i * mean[(1-U) * L_consistency]_{~B}

    Where:
      U = combined aleatoric uncertainty from U_geo and U_app
      B = boundary band mask
      L_consistency = per-pixel MSE between student and SAM pseudo-label
      agreement_weight = optional student-SAM agreement gate (elem-wise, [0,1])

    Args:
        student_pred: (B, 1, H, W) specialist probs.
        sam_pseudo:   (B, 1, H, W) SAM pseudo-label probs.
        u_geo:        (B, 1, H, W) box-perturbation uncertainty [0, 1].
        u_app:        (B, 1, H, W) heteroscedastic variance [0, 1], or None.
        boundary_mask:(B, 1, H, W) boundary band {0, 1}.
        w_boundary:   weight for boundary term.
        w_interior:   weight for interior term.
        agreement_weight: (B, 1, H, W) per-pixel student-SAM agreement [0, 1].

    Returns:
        Dict with 'total', 'boundary', 'interior' loss scalars.
    """
    # NaN guard: check inputs before computation
    for name, t in [
        ("student_pred", student_pred), ("sam_pseudo", sam_pseudo),
        ("u_geo", u_geo), ("boundary_mask", boundary_mask),
    ]:
        if not torch.isfinite(t).all():
            return {
                "total": torch.tensor(0.0, device=t.device),
                "boundary": torch.tensor(0.0, device=t.device),
                "interior": torch.tensor(0.0, device=t.device),
            }
    if u_app is not None and not torch.isfinite(u_app).all():
        u_app = None  # fall through to u_geo-only
    if agreement_weight is not None and not torch.isfinite(agreement_weight).all():
        agreement_weight = None

    # Consistency error per pixel
    per_pixel_err = (student_pred - sam_pseudo) ** 2  # (B, 1, H, W)

    # Combined uncertainty: average of geo and app, or just geo
    if u_app is not None:
        u_combined = (u_geo + u_app) / 2.0
    else:
        u_combined = u_geo

    # Refactor #2: inverted gate — high uncertainty → high weight.
    # Controlled by cfg.uncertainty.use_inverted_gating
    import os
    if os.environ.get("T4_USE_INVERTED_GATING", "1") == "1":
        confidence = u_combined  # up-weight uncertain pixels
    else:
        confidence = 1.0 - u_combined  # standard: suppress uncertain pixels

    # Student-SAM agreement gate (ponytail: SAM not crack-trained, moderate trust)
    if agreement_weight is not None:
        confidence = confidence * agreement_weight

    # Boundary term
    boundary_err = per_pixel_err * confidence * boundary_mask
    boundary_sum = boundary_mask.sum() + 1e-8
    loss_boundary = w_boundary * (boundary_err.sum() / boundary_sum)

    # Interior term
    interior_mask = 1.0 - boundary_mask
    interior_err = per_pixel_err * confidence * interior_mask
    interior_sum = interior_mask.sum() + 1e-8
    loss_interior = w_interior * (interior_err.sum() / interior_sum)

    loss_total = loss_boundary + loss_interior

    return {
        "total": loss_total,
        "boundary": loss_boundary,
        "interior": loss_interior,
    }
