"""SemiSAM-Crack diagnostic visualizations.

Generates per-sample multi-panel figures showing:
  1. Original image + ground truth mask
  2. Coarse prediction (specialist output before SAM refinement)
  3. SAM prompts (box prompts overlaid on image, or point prompts)
  4. Final label (SAM-refined pseudo-label or specialist final)

Usage:
    python visualize.py --labeled-dir data/train_labeled --output-dir viz_output/
    python visualize.py --checkpoint checkpoints/best_model.pth --num-samples 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from prompting.boxes import boxes_from_mask

# ---------------------------------------------------------------------------
# color palette
# ---------------------------------------------------------------------------

CMAP_CRACK = plt.cm.Reds  # for ground truth / final masks
CMAP_COARSE = plt.cm.Blues  # for coarse predictions
BOX_COLOR = "#00FF00"  # green boxes
BOX_ALPHA = 0.6
GT_COLOR = np.array([220, 50, 50]) / 255.0  # red overlay
PRED_COLOR = np.array([50, 120, 220]) / 255.0  # blue overlay


# ---------------------------------------------------------------------------
# core visualization
# ---------------------------------------------------------------------------


def make_diagnostic_figure(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    coarse_pred: np.ndarray,
    boxes: list[list[float]],
    final_mask: np.ndarray,
    save_path: str,
    title: str = "",
):
    """Create a 2×3 or 2×2 diagnostic panel.

    Layout:
        Row 1: [Original+GT overlay] [Coarse prediction] [Coarse overlay]
        Row 2: [SAM box prompts]      [Final prediction]  [Final overlay]
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Normalize image for display
    img_disp = image_rgb.astype(np.float32) / 255.0

    # --- Row 1 ---
    # [0,0] Original image + GT overlay
    ax = axes[0, 0]
    ax.imshow(img_disp)
    if gt_mask is not None and gt_mask.any():
        overlay = np.zeros_like(img_disp)
        overlay[gt_mask > 0] = GT_COLOR
        ax.imshow(overlay, alpha=0.4)
    ax.set_title("Image + Ground Truth")
    ax.axis("off")

    # [0,1] Coarse prediction (specialist)
    ax = axes[0, 1]
    ax.imshow(coarse_pred, cmap=CMAP_COARSE, vmin=0, vmax=1)
    ax.set_title(f"Coarse Prediction\n(crack_px={int(coarse_pred.sum())})")
    ax.axis("off")

    # [0,2] Coarse prediction overlaid on image
    ax = axes[0, 2]
    ax.imshow(img_disp)
    if coarse_pred.any():
        overlay = np.zeros_like(img_disp)
        overlay[coarse_pred > 0] = PRED_COLOR
        ax.imshow(overlay, alpha=0.4)
    ax.set_title("Coarse Overlay")
    ax.axis("off")

    # --- Row 2 ---
    # [1,0] SAM box prompts
    ax = axes[1, 0]
    ax.imshow(img_disp)
    for (x1, y1, x2, y2) in boxes:
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=BOX_COLOR,
            facecolor="none",
            alpha=BOX_ALPHA,
        )
        ax.add_patch(rect)
    ax.set_title(f"SAM Box Prompts\n({len(boxes)} boxes)")
    ax.axis("off")

    # [1,1] Final prediction
    ax = axes[1, 1]
    ax.imshow(final_mask, cmap=CMAP_CRACK, vmin=0, vmax=1)
    ax.set_title(f"Final Prediction\n(crack_px={int(final_mask.sum())})")
    ax.axis("off")

    # [1,2] Final overlay on image
    ax = axes[1, 2]
    ax.imshow(img_disp)
    if final_mask.any():
        overlay = np.zeros_like(img_disp)
        overlay[final_mask > 0] = GT_COLOR
        ax.imshow(overlay, alpha=0.5)
    ax.set_title("Final Overlay")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# batch generation
# ---------------------------------------------------------------------------

def visualize_samples(
    dataset: CrackDataset,
    model: Specialist | None = None,
    generalist=None,
    num_samples: int = 8,
    output_dir: str = "viz_output",
    device: str = "cuda",
    box_margin_px: int = 10,
    box_merge_dist_px: int = 30,
):
    """Generate diagnostic plots for `num_samples` from the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    if model is not None:
        model.eval()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        image_t = sample["image"]  # (3, H, W) normalized tensor
        label_t = sample["label"]  # (1, H, W) float tensor

        # Denormalize image for display
        image_np = _denormalize_for_display(image_t)

        gt_np = (label_t.squeeze().numpy() > 0.5).astype(np.uint8)

        # --- coarse prediction ---
        if model is not None:
            with torch.no_grad():
                inp = image_t.unsqueeze(0).to(device)
                out = model(inp)
                coarse_logits = out["mask_logits"][0]
                coarse_prob = torch.sigmoid(coarse_logits).squeeze().cpu().numpy()
        else:
            # ponytail: random coarse mask for demo without model
            coarse_prob = (np.random.randn(*image_np.shape[:2]) * 0.5 + 0.5).clip(0, 1)

        coarse_bin = (coarse_prob > 0.5).astype(np.uint8)

        # --- box prompts ---
        coarse_t = torch.from_numpy(coarse_prob).unsqueeze(0).unsqueeze(0).to(device)
        boxes = boxes_from_mask(
            coarse_t,
            margin_px=box_margin_px,
            merge_dist_px=box_merge_dist_px,
        )

        # Scale boxes from model output size back to display size
        h_disp, w_disp = image_np.shape[:2]
        h_prob, w_prob = coarse_prob.shape
        scale_x = w_disp / max(w_prob, 1)
        scale_y = h_disp / max(h_prob, 1)
        boxes_scaled = [
            [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            for b in boxes
        ]

        # --- final prediction (try SAM if available, else use coarse) ---
        final_np = coarse_bin
        if generalist is not None and len(boxes) > 0:
            try:
                # SAM expects uint8 RGB
                img_for_sam = (image_np * 255).astype(np.uint8)
                sam_mask = generalist.predict(img_for_sam, boxes=boxes_scaled)
                if sam_mask is not None and sam_mask.sum() > 0:
                    final_np = sam_mask.astype(np.uint8)
            except Exception as e:
                print(f"  SAM failed for sample {idx}: {e}")

        # --- save ---
        save_path = os.path.join(output_dir, f"sample_{idx:03d}.png")
        make_diagnostic_figure(
            image_rgb=image_np,
            gt_mask=gt_np,
            coarse_pred=coarse_bin,
            boxes=boxes_scaled,
            final_mask=final_np,
            save_path=save_path,
            title=f"Sample {idx} | Boxes: {len(boxes)} | Crack px (GT/Coarse/Final): {gt_np.sum()}/{coarse_bin.sum()}/{final_np.sum()}",
        )

    print(f"Saved {len(indices)} diagnostic plots to {output_dir}/")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _denormalize_for_display(image_t: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization → uint8 RGB numpy (H, W, 3)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = image_t.permute(1, 2, 0).numpy()
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemiSAM-Crack Diagnostic Visualizations")
    parser.add_argument("--labeled-dir", type=str, default="data/train_labeled/images")
    parser.add_argument("--mask-dir", type=str, default="data/train_labeled/masks")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specialist model checkpoint")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="viz_output")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--box-margin", type=int, default=10)
    parser.add_argument("--box-merge", type=int, default=30)
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM (use coarse as final)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    ds = CrackDataset(
        image_dir=args.labeled_dir,
        mask_dir=args.mask_dir,
        transform=get_val_transform(tuple(args.img_size)),
    )

    # Model
    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        model = Specialist(pretrained=False).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Generalist (SAM-3)
    generalist = None
    if not args.no_sam:
        sam_ckpt = os.environ.get(
            "SAM_CKPT",
            "/home/guest/.cache/modelscope/hub/models/facebook/sam3/sam3.pt",
        )
        try:
            from model.generalist_wrapper import GeneralistWrapper
            generalist = GeneralistWrapper(checkpoint=sam_ckpt, device=device)
            print(f"SAM-3 generalist loaded from {sam_ckpt}")
        except Exception as e:
            print(f"SAM-3 not available: {e}")

    visualize_samples(
        dataset=ds,
        model=model,
        generalist=generalist,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=device,
        box_margin_px=args.box_margin,
        box_merge_dist_px=args.box_merge,
    )
