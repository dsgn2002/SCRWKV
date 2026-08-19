"""Supervised risk targets for topology calibration.

Computes per-component ground-truth risk scores from EMA predictions
and GT masks on labeled data. Risk maps are back-projected spatially
for supervised training of StructureRiskHeads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from prompting.components import extract_components, CrackComponent


# ---------------------------------------------------------------------------
# per-component risk computation
# ---------------------------------------------------------------------------

def compute_risk_targets(
    mask_prob: np.ndarray,       # (H, W) EMA prediction probability
    gt_mask: np.ndarray,         # (H, W) GT binary mask
    gt_centerline: np.ndarray,   # (H, W) GT skeleton
    min_area: int = 30,
) -> dict[str, np.ndarray]:
    """Compute per-component risk scores and back-project to spatial risk maps.

    Returns risk maps matching mask_prob spatial size:
      {"R_topo_target": (H,W), "R_morph_target": (H,W), "R_app_target": (H,W)}
    All values in [0, 1].
    """
    H, W = mask_prob.shape
    pred_bin = (mask_prob > 0.5).astype(np.uint8)

    # Init risk maps to 0 (low risk for background regions)
    topo_map = np.zeros((H, W), dtype=np.float32)
    morph_map = np.zeros((H, W), dtype=np.float32)
    app_map = np.zeros((H, W), dtype=np.float32)

    if pred_bin.sum() == 0:
        # No prediction — risk is undefined, return zeros
        return {"R_topo_target": topo_map, "R_morph_target": morph_map, "R_app_target": app_map}

    components = extract_components(mask_prob, min_area=min_area)

    for comp in components:
        comp_mask = comp.mask.astype(bool)
        if comp_mask.sum() == 0:
            continue

        # --- Topology risk ---
        # High when: fragmentation (many components), many endpoints per length,
        # or skeleton has breaks near component edges.
        e_topo = _compute_topo_risk(comp, gt_mask, gt_centerline)

        # --- Morphology risk ---
        # High when: implausible width, poor boundary match, shape deviation.
        e_morph = _compute_morph_risk(comp, gt_mask)

        # --- Appearance risk ---
        # High when: low prediction confidence, distractor confusion.
        e_app = _compute_app_risk(comp, gt_mask, mask_prob)

        # Assign risk scores to component pixels
        topo_map[comp_mask] = np.clip(e_topo, 0.0, 1.0)
        morph_map[comp_mask] = np.clip(e_morph, 0.0, 1.0)
        app_map[comp_mask] = np.clip(e_app, 0.0, 1.0)

    return {"R_topo_target": topo_map, "R_morph_target": morph_map, "R_app_target": app_map}


# ---------------------------------------------------------------------------
# per-component heuristics (ponytail: lightweight, no learned model needed here)
# ---------------------------------------------------------------------------

def _compute_topo_risk(
    comp: CrackComponent,
    gt_mask: np.ndarray,
    gt_centerline: np.ndarray,
) -> float:
    """Topology risk from endpoint density, fragmentation, and GT skeleton mismatch.

    e_topo = α·endpoint_error + (1−α)·(1−clDice_approx)

    ponytail: simplified version — endpoint density + component count proxy.
    """
    comp_mask = comp.mask.astype(bool)
    n_endpoints = len(comp.endpoints)

    # Endpoint density: endpoints per 100px of skeleton length
    # Higher = more fragmentation in this component
    endpoint_density = min(n_endpoints / (comp.length + 1), 1.0) * 2.0  # cap at 2

    # Component count proxy: if many small components → fragmentation
    # Single component: low risk. Many endpoints on a single component: high risk.
    topo = 0.5 * min(endpoint_density, 1.0)

    # GT skeleton overlap: does the predicted component cover GT centerline?
    if gt_centerline is not None and gt_centerline.sum() > 0:
        gt_centerline_bin = gt_centerline.astype(bool)
        overlap = (comp_mask & gt_centerline_bin).sum() / (comp_mask.sum() + 1)
        topo += 0.5 * (1.0 - overlap)

    return float(np.clip(topo, 0.0, 1.0))


def _compute_morph_risk(
    comp: CrackComponent,
    gt_mask: np.ndarray,
) -> float:
    """Morphology risk from width variation and boundary accuracy.

    e_morph = a·boundary_error + b·width_deviation

    ponytail: width/length ratio + component shape as proxy.
    """
    # Width-to-length ratio: very thin or very thick = implausible
    # Typical crack: width ~1-10px, length >> width
    aspect = comp.width / (comp.length + 1)  # should be small for cracks

    # Width plausibility: if width > 50px or width/area suggests non-crack
    width_score = min(comp.width / 50.0, 1.0)  # > 50px wide = high risk

    # Shape: if component fills its bbox poorly → blob, not crack-like
    bbox_w = comp.bbox[2] - comp.bbox[0] + 1
    bbox_h = comp.bbox[3] - comp.bbox[1] + 1
    bbox_area = max(bbox_w * bbox_h, 1)
    fill_ratio = comp.area / bbox_area  # near 1 = blob, << 1 = thin

    # Blob penalty: fill ratio > 0.5 is non-crack-like
    blob_risk = max(0.0, (fill_ratio - 0.3) * 2.0)  # > 0.3 fill = increasingly blob-like

    # GT boundary overlap check
    if gt_mask is not None and gt_mask.sum() > 0:
        gt_mask_bin = gt_mask.astype(bool)
        gt_overlap = (comp.mask.astype(bool) & gt_mask_bin).sum() / (comp.area + 1)
    else:
        gt_overlap = 0.5

    morph = 0.3 * width_score + 0.3 * blob_risk + 0.4 * (1.0 - gt_overlap)
    return float(np.clip(morph, 0.0, 1.0))


def _compute_app_risk(
    comp: CrackComponent,
    gt_mask: np.ndarray,
    mask_prob: np.ndarray,
) -> float:
    """Appearance risk from prediction confidence and GT mismatch.

    e_app = 1 − confidence (distractor probability proxy)
    """
    # Lower confidence → higher appearance risk
    conf_risk = 1.0 - comp.confidence

    # Check if this component overlaps GT (is it a true crack or distractor?)
    if gt_mask is not None and gt_mask.sum() > 0:
        gt_bin = gt_mask.astype(bool)
        overlap = (comp.mask.astype(bool) & gt_bin).sum()
        # FP rate: predicted crack where GT says background
        fp_rate = (comp.area - overlap) / (comp.area + 1)
        app = 0.5 * conf_risk + 0.5 * fp_rate
    else:
        app = conf_risk

    return float(np.clip(app, 0.0, 1.0))


# ---------------------------------------------------------------------------
# consultation target: Δ_j = Dice(EMA) - Dice(SAM) for labeled components
# ---------------------------------------------------------------------------

def compute_consultation_target(
    ema_mask: np.ndarray,       # (H, W) EMA prediction binary
    sam_mask: np.ndarray,       # (H, W) SAM prediction binary
    gt_mask: np.ndarray,        # (H, W) GT binary
    component_mask: np.ndarray, # (H, W) component region binary
) -> float:
    """Compute consultation value target for one component.

    Δ_j = Dice(EMA_pred, GT) − Dice(SAM_pred, GT) within the component region.

    Positive Δ_j → SAM improves over EMA → consult SAM.
    Negative Δ_j → SAM degrades vs EMA → don't consult.
    """
    smooth = 1e-5
    comp = component_mask.astype(bool)

    ema_dice = _local_dice(ema_mask.astype(bool), gt_mask.astype(bool), comp)
    sam_dice = _local_dice(sam_mask.astype(bool), gt_mask.astype(bool), comp)

    return float(np.clip(ema_dice - sam_dice, -1.0, 1.0))


def _local_dice(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray
) -> float:
    """Dice score restricted to a local mask region."""
    p = pred & mask
    g = gt & mask
    smooth = 1e-5
    inter = (p & g).sum()
    return (2.0 * inter + smooth) / (p.sum() + g.sum() + smooth)
