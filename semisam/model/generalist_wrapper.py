"""Frozen SAM generalist wrapper — crack-fine-tuned.

Loads SAM-1 ViT-B format (segment-anything) for crack-fine-tuned checkpoints.
Also supports SAM-3 Ultralytics format via auto-detection.
"""

from __future__ import annotations

import numpy as np
import torch


class GeneralistWrapper:
    """Frozen SAM generalist with crack-domain checkpoint.

    Auto-detects checkpoint format:
      - SAM-1 (segment-anything): loads via sam_model_registry + SamPredictor
      - SAM-3 (ultralytics):     loads via ultralytics.SAM

    Usage:
        gw = GeneralistWrapper(checkpoint="path/to/crack_sam.pth")
        mask = gw.predict(image_np, boxes=[[x1,y1,x2,y2], ...], points=..., point_labels=...)
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._backend = self._detect_backend(checkpoint)
        self._load(checkpoint)

    def _detect_backend(self, checkpoint: str) -> str:
        """Detect SAM-1 vs SAM-3 from checkpoint keys."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "image_encoder.pos_embed" in ckpt:
            return "sam1"
        return "sam3"

    def _load(self, checkpoint: str):
        if self._backend == "sam1":
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry["vit_b"](checkpoint=None)
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)

            # ponytail: crack checkpoint (3 multimask outputs) vs installed SAM (4).
            # Pad dim-0 of mismatched tensors to match.
            model_sd = sam.state_dict()
            for key in list(state.keys()):
                if key in model_sd and state[key].shape != model_sd[key].shape:
                    t = torch.zeros(model_sd[key].shape, dtype=state[key].dtype)
                    # Copy overlapping elements: checkpoint into first N slots
                    if state[key].ndim == 1:
                        t[:state[key].shape[0]] = state[key]
                    elif state[key].ndim == 2:
                        t[:state[key].shape[0], :state[key].shape[1]] = state[key]
                    state[key] = t
            sam.load_state_dict(state, strict=False)  # mlps.3 keys missing — unused 4th token
            sam.to(self.device)
            sam.eval()
            for p in sam.parameters():
                p.requires_grad = False
            self.sam = sam
            self.predictor = SamPredictor(sam)
        else:
            from ultralytics import SAM

            self.model = SAM(checkpoint)

    # ------------------------------------------------------------------
    def predict(
        self,
        image: np.ndarray,
        *,
        boxes: list[list[float]] | None = None,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
    ) -> np.ndarray:
        """Run SAM with box/point prompts, return binary mask (H, W) uint8."""
        if self._backend == "sam1":
            return self._predict_sam1(image, boxes, points, point_labels)
        return self._predict_sam3(image, boxes, points, point_labels)

    def _predict_sam1(self, image, boxes, points, point_labels):
        self.predictor.set_image(image)

        box_np = np.array(boxes) if boxes and len(boxes) > 0 else None
        pt_np = None
        pt_lbl_np = None
        if points and len(points) > 0:
            pt_np = np.array(points)
            pt_lbl_np = np.array(point_labels or [1] * len(points))

        masks, scores, _ = self.predictor.predict(
            point_coords=pt_np,
            point_labels=pt_lbl_np,
            box=box_np,
            multimask_output=False,
        )
        return (masks[0] > 0).astype(np.uint8) if masks.shape[0] > 0 else np.zeros(image.shape[:2], dtype=np.uint8)

    def _predict_sam3(self, image, boxes, points, point_labels):
        has_boxes = boxes and len(boxes) > 0
        if not has_boxes:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        if len(boxes) == 1:
            bbox = boxes[0]
        else:
            xs = [b[0] for b in boxes] + [b[2] for b in boxes]
            ys = [b[1] for b in boxes] + [b[3] for b in boxes]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        results = self.model.predict(
            source=image, bboxes=[bbox],
            points=[points] if points else None,
            labels=[point_labels] if point_labels else None,
            save=False, verbose=False,
        )
        if results and results[0].masks is not None and results[0].masks.data.shape[0] > 0:
            return (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # ------------------------------------------------------------------
    def predict_unc(
        self, image: np.ndarray, jittered_boxes: list[list[list[float]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """n perturbed box sets → mask (majority vote) + uncertainty entropy."""
        all_masks = [self.predict(image, boxes=b).astype(np.float32) for b in jittered_boxes]
        stacked = np.stack(all_masks, axis=0)
        mean = stacked.mean(axis=0)
        eps = 1e-7
        mean_c = np.clip(mean, eps, 1.0 - eps)
        entropy = -mean_c * np.log(mean_c) - (1 - mean_c) * np.log(1 - mean_c)
        uncertainty = entropy / np.log(2.0)
        final = (stacked.mean(axis=0) > 0.5).astype(np.uint8)
        return final, uncertainty.astype(np.float32)
