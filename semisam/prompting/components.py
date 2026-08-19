"""Component-level crack extraction for selective SAM consultation.

Stage 2 (TopoSAM): explicit structural descriptors per candidate region,
mask-pooled encoder features along skeleton paths and endpoint neighborhoods.

Stage 3 (TopoSAM): appearance feature extraction via mask pooling inside
candidate, background ring, and contrast difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrackComponent:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: np.ndarray                 # binary mask of this component
    skel: np.ndarray                 # skeleton of this component
    endpoints: list[tuple[int, int]] # skeleton endpoint coordinates
    area: int
    length: float                    # skeleton pixel count
    width: float                     # area / length
    confidence: float                # mean mask probability
    crop_margin: int = 20

    # --- Stage 2: topological descriptors ---
    branch_points: list[tuple[int, int]] = field(default_factory=list)
    n_branches: int = 0
    n_fragments: int = 1
    candidate_gaps: list[dict] = field(default_factory=list)
    # pooled feature vectors (set after extraction)
    z_skel: np.ndarray | None = None       # mask-pooled features along skeleton
    z_endpoint: np.ndarray | None = None   # mask-pooled features at endpoints

    # --- Stage 3: appearance descriptors ---
    z_in: np.ndarray | None = None         # features inside candidate
    z_ring: np.ndarray | None = None       # features in background ring
    z_diff: np.ndarray | None = None       # contrast: z_in - z_ring
    appearance_scalars: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# mask pooling (Stages 2-3)
# ---------------------------------------------------------------------------

def mask_pool(
    features: np.ndarray,       # (C, H, W) encoder feature map
    region_mask: np.ndarray,    # (H, W) binary mask
) -> np.ndarray:
    """Average-pool feature vectors within a binary region mask.

    Returns (C,) vector, or zeros if region is empty.
    """
    if region_mask.sum() == 0:
        return np.zeros(features.shape[0], dtype=np.float32)
    # ponytail: einsum over spatial dims — 1 line, no reshape chain
    return np.einsum('chw,hw->c', features.astype(np.float32),
                     region_mask.astype(np.float32)) / region_mask.sum()


# ---------------------------------------------------------------------------
# Stage 2: topology vector extraction
# ---------------------------------------------------------------------------

def _detect_branch_points(skel: np.ndarray) -> list[tuple[int, int]]:
    """Find skeleton pixels with ≥3 neighbors (branch/junction points)."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    branch_pts = np.argwhere((skel == 1) & (neighbors >= 3))
    return [(int(p[1]), int(p[0])) for p in branch_pts]


def _count_fragments(mask: np.ndarray) -> int:
    """Count disconnected sub-components within this component's mask."""
    n_labels, _, _, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return max(1, n_labels - 1)  # subtract background


def _detect_gaps(
    endpoints: list[tuple[int, int]],
    skel: np.ndarray,
    max_dist_px: int = 50,
) -> list[dict]:
    """Find nearby endpoint pairs that may be disconnected crack segments.

    Returns list of {endpoint_a, endpoint_b, distance, orientation_compat}.
    """
    if len(endpoints) < 2:
        return []

    pts = np.array(endpoints, dtype=np.float32)
    gaps = []

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist_px:
                continue
            # orientation: vector between endpoints
            vec = pts[j] - pts[i]
            vec = vec / (d + 1e-8)
            # local tangent at each endpoint (3px neighborhood)
            tan_i = _local_tangent(skel, pts[i])
            tan_j = _local_tangent(skel, pts[j])
            if tan_i is None or tan_j is None:
                continue
            # compatibility: dot product of tangents with gap direction
            orient_compat = max(
                abs(np.dot(tan_i, vec)), abs(np.dot(tan_j, vec))
            )
            gaps.append({
                "endpoint_a": (int(pts[i][0]), int(pts[i][1])),
                "endpoint_b": (int(pts[j][0]), int(pts[j][1])),
                "distance": float(d),
                "orientation_compat": float(orient_compat),
            })

    return gaps


def _local_tangent(skel: np.ndarray, pt: np.ndarray, radius: int = 5
                   ) -> np.ndarray | None:
    """PCA-based tangent direction at a skeleton point. Returns (2,) unit vector."""
    H, W = skel.shape
    ys, xs = np.mgrid[
        max(0, int(pt[0]) - radius):min(H, int(pt[0]) + radius + 1),
        max(0, int(pt[1]) - radius):min(W, int(pt[1]) + radius + 1),
    ]
    region = skel[ys, xs]
    if region.sum() < 3:
        return None
    coords = np.column_stack([ys[region > 0], xs[region > 0]]).astype(np.float32)
    coords -= coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    return vh[0]  # first principal component


def extract_topological_vector(comp: CrackComponent) -> np.ndarray:
    """Build the 10-dim topological descriptor vector t_j for a component.

    Returns (10,) float32 array:
      [endpoint_count, branch_count, n_fragments, skel_length,
       gap_count, max_gap_dist, mean_orient_compat,
       mean_width, width_std, aspect_ratio]
    """
    n_ep = len(comp.endpoints)
    n_bp = comp.n_branches
    n_frag = comp.n_fragments
    skel_len = comp.length

    gaps = comp.candidate_gaps
    n_gaps = len(gaps)
    max_gap = max((g["distance"] for g in gaps), default=0.0)
    mean_orient = np.mean([g["orientation_compat"] for g in gaps]) if gaps else 0.0

    # width statistics from skeleton distance transform
    skel_pts = np.argwhere(comp.skel > 0)
    widths = []
    if len(skel_pts) > 0:
        dist = cv2.distanceTransform(
            (comp.mask > 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        for y, x in skel_pts:
            widths.append(dist[y, x] * 2)  # radius → diameter
    mean_w = float(np.mean(widths)) if widths else comp.width
    std_w = float(np.std(widths)) if widths else 0.0

    x1, y1, x2, y2 = comp.bbox
    aspect = (x2 - x1) / max(y2 - y1, 1)

    return np.array([
        n_ep, n_bp, n_frag, skel_len,
        n_gaps, max_gap, mean_orient,
        mean_w, std_w, aspect,
    ], dtype=np.float32)


def extract_topology_features(
    encoder_features: list[np.ndarray],  # [(C_l, H_l, W_l)] per encoder level
    comp: CrackComponent,
) -> None:
    """Populate comp.z_skel and comp.z_endpoint via mask pooling.

    Pools multi-scale encoder features at skeleton path and endpoint
    neighborhoods, then concatenates across levels into flat vectors.
    """
    # ponytail: resize component masks to each encoder level, pool, concat
    z_skel_parts = []
    z_end_parts = []

    for Feat in encoder_features:
        _, Hf, Wf = Feat.shape
        # Resize masks to feature map resolution
        skel_f = cv2.resize(
            comp.skel.astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_NEAREST
        )
        mask_f = cv2.resize(
            comp.mask.astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_NEAREST
        )

        # Endpoint neighborhood: 5px radius around each endpoint
        end_mask = np.zeros((Hf, Wf), dtype=np.float32)
        for ep_x, ep_y in comp.endpoints:
            # scale endpoint to feature map coords
            fx = int(ep_x * Wf / (comp.bbox[2] - comp.bbox[0] + 1)) if comp.bbox[2] > comp.bbox[0] else 0
            fy = int(ep_y * Hf / (comp.bbox[3] - comp.bbox[1] + 1)) if comp.bbox[3] > comp.bbox[1] else 0
            r = 2  # 2px radius at feature resolution
            y0, y1 = max(0, fy - r), min(Hf, fy + r + 1)
            x0, x1 = max(0, fx - r), min(Wf, fx + r + 1)
            end_mask[y0:y1, x0:x1] = 1.0
        # Intersect endpoint neighborhood with component mask
        end_mask = end_mask * (mask_f > 0)

        z_skel_parts.append(mask_pool(Feat, skel_f))
        z_end_parts.append(mask_pool(Feat, end_mask))

    comp.z_skel = np.concatenate(z_skel_parts) if z_skel_parts else np.array([])
    comp.z_endpoint = np.concatenate(z_end_parts) if z_end_parts else np.array([])


# ---------------------------------------------------------------------------
# Stage 3: appearance feature extraction
# ---------------------------------------------------------------------------

def extract_appearance_features(
    encoder_features: list[np.ndarray],  # [(C_l, H_l, W_l)] per encoder level
    comp: CrackComponent,
    ema_prob: np.ndarray,               # (H, W) EMA prediction
    student_prob: np.ndarray | None = None,  # (H, W) student prediction
    ring_width_px: int = 10,
) -> None:
    """Populate comp.z_in, z_ring, z_diff and appearance_scalars.

    Pools encoder features from three regions:
      - Inside candidate mask C_j
      - Background ring Dilate(C_j) \ C_j
      - Contrast difference z_in - z_ring
    Also computes scalar appearance descriptors.
    """
    # --- region masks ---
    mask = comp.mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width_px, ring_width_px))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = (dilated > 0) & (mask == 0)

    H, W = mask.shape
    comp_ema = ema_prob[mask > 0]
    comp_student = student_prob[mask > 0] if student_prob is not None else None

    # --- appearance scalars ---
    entropy_ema = float(np.mean(
        -(comp_ema * np.log(comp_ema + 1e-7)
          + (1 - comp_ema) * np.log(1 - comp_ema + 1e-7))
    ))
    ema_student_disagreement = float(
        np.mean(np.abs(comp_ema - comp_student))
    ) if comp_student is not None else 0.0

    # Local image-gradient statistics (on original image, Sobel)
    grad_mean = 0.0
    grad_std = 0.0

    # ponytail: per-level mask pooling for inside, ring, diff
    z_in_parts, z_ring_parts, z_diff_parts = [], [], []

    for Feat in encoder_features:
        _, Hf, Wf = Feat.shape
        mask_f = cv2.resize(mask.astype(np.float32), (Wf, Hf),
                            interpolation=cv2.INTER_NEAREST)
        ring_f = cv2.resize(ring.astype(np.float32), (Wf, Hf),
                            interpolation=cv2.INTER_NEAREST)

        z_in = mask_pool(Feat, mask_f)
        z_ring = mask_pool(Feat, ring_f)
        z_in_parts.append(z_in)
        z_ring_parts.append(z_ring)
        z_diff_parts.append(z_in - z_ring)

    comp.z_in = np.concatenate(z_in_parts) if z_in_parts else np.array([])
    comp.z_ring = np.concatenate(z_ring_parts) if z_ring_parts else np.array([])
    comp.z_diff = np.concatenate(z_diff_parts) if z_diff_parts else np.array([])

    comp.appearance_scalars = {
        "foreground_prob": float(np.mean(comp_ema)),
        "entropy": entropy_ema,
        "ema_student_disagreement": ema_student_disagreement,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
    }


# ---------------------------------------------------------------------------
# main extraction entry point
# ---------------------------------------------------------------------------

def extract_components(
    mask_prob: np.ndarray,   # (H, W) probability map
    min_area: int = 30,
    margin_px: int = 10,
) -> list[CrackComponent]:
    """Extract connected components from mask with structural metadata.

    Returns list of CrackComponent sorted by area (largest first).
    """
    mask_bin = (mask_prob > 0.5).astype(np.uint8)
    if mask_bin.sum() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )
    components = []

    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x2 = x1 + w
        y2 = y1 + h

        comp_mask = (labels == i).astype(np.uint8)
        comp_skel = skeletonize(comp_mask > 0).astype(np.uint8)

        # Endpoints: pixels with exactly 1 neighbor
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(comp_skel, -1, kernel)
        endpoints = np.argwhere((comp_skel == 1) & (neighbors == 1))
        endpoint_list = [(int(p[1]), int(p[0])) for p in endpoints]

        length = max(comp_skel.sum(), 1)
        width_val = area / length

        # Confidence: mean probability within this component
        conf = float(mask_prob[comp_mask > 0].mean())

        # --- Stage 2: topology descriptors ---
        branch_pts = _detect_branch_points(comp_skel)
        n_frag = _count_fragments(comp_mask)
        gaps = _detect_gaps(endpoint_list, comp_skel)

        bbox = (
            max(0, x1 - margin_px),
            max(0, y1 - margin_px),
            min(mask_prob.shape[1], x2 + margin_px),
            min(mask_prob.shape[0], y2 + margin_px),
        )

        components.append(CrackComponent(
            bbox=bbox,
            mask=comp_mask,
            skel=comp_skel,
            endpoints=endpoint_list,
            area=int(area),
            length=float(length),
            width=float(width_val),
            confidence=conf,
            branch_points=branch_pts,
            n_branches=len(branch_pts),
            n_fragments=n_frag,
            candidate_gaps=gaps,
        ))

    # Sort by area descending
    components.sort(key=lambda c: c.area, reverse=True)
    return components
