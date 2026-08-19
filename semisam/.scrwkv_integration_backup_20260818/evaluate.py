"""Evaluation and ablation for SemiSAM-Crack.

Ported from SemiSAM+ val_3D.py + test_3D_util.py — 3D sliding window → 2D direct.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from utils.metrics import (
    dice_score,
    iou_score,
    hd95,
    asd,
    boundary_iou,
    boundary_f1,
    cl_dice,
    calculate_metric_percase,
)


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """Run validation, return mean Dice score."""
    model.eval()
    dices = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            label = batch.get("label")
            if label is None:
                continue
            label = label.to(device)

            out = model(image)
            pred = (torch.sigmoid(out["mask_logits"]) > 0.5).float()

            for b in range(pred.shape[0]):
                d = dice_score(
                    pred[b, 0].cpu().numpy(),
                    label[b, 0].cpu().numpy(),
                )
                if not np.isnan(d):
                    dices.append(d)

    model.train()
    return float(np.mean(dices)) if dices else 0.0


def evaluate_full(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    output_dir: str | None = None,
) -> dict[str, float]:
    """Run full evaluation with all metrics.

    Returns dict of metric name → mean value across dataset.
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image = batch["image"].to(device)
            label = batch.get("label")
            if label is None:
                continue
            label = label.to(device)

            out = model(image)
            pred = (torch.sigmoid(out["mask_logits"]) > 0.5).float()

            for b in range(pred.shape[0]):
                pred_np = pred[b, 0].cpu().numpy().astype(bool)
                gt_np = label[b, 0].cpu().numpy().astype(bool)

                m = {
                    "dice": dice_score(pred_np, gt_np),
                    "iou": iou_score(pred_np, gt_np),
                    "hd95": hd95(pred_np, gt_np),
                    "asd": asd(pred_np, gt_np),
                    "boundary_iou": boundary_iou(pred_np, gt_np, width=5),
                    "boundary_f1": boundary_f1(pred_np, gt_np, width=5),
                    "cl_dice": cl_dice(pred_np, gt_np),
                }
                all_metrics.append(m)

    model.train()

    if len(all_metrics) == 0:
        return {}

    # Average
    result = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics if not np.isnan(m[key]) and np.isfinite(m[key])]
        result[key] = float(np.mean(values)) if values else 0.0

    return result


def run_ablation(
    checkpoint_paths: dict[str, str],
    val_dir: str,
    img_size: tuple[int, int] = (512, 512),
    output_csv: str = "results/ablation.csv",
):
    """Run ablation: load each checkpoint, evaluate, save CSV.

    Args:
        checkpoint_paths: dict of config_name → checkpoint_path.
        val_dir: path to validation data.
        img_size: (H, W) for resize.
        output_csv: path to save results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataset = CrackDataset(
        image_dir=val_dir,
        mask_dir=val_dir,
        transform=get_val_transform(img_size),
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for config_name, ckpt_path in checkpoint_paths.items():
        print(f"\n--- Evaluating: {config_name} ---")
        model = Specialist(pretrained=True).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

        metrics = evaluate_full(model, valloader, device)
        metrics["config"] = config_name
        results.append(metrics)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save CSV
    if results:
        fieldnames = ["config"] + [k for k in results[0] if k != "config"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemiSAM-Crack Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--val_dir", type=str, default="data/val/", help="Validation data dir")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--ablate", action="store_true", help="Run ablation mode")
    parser.add_argument("--output", type=str, default="results/metrics.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = tuple(args.img_size)

    if args.ablate:
        # Expect checkpoint paths as colon-separated pairs: "box_upfm=path1,box_noupfm=path2,..."
        # ponytail: simple format for now
        raise NotImplementedError(
            "Ablation mode: use run_ablation() directly with a dict of checkpoints"
        )
    else:
        val_dataset = CrackDataset(
            image_dir=args.val_dir,
            mask_dir=args.val_dir,
            transform=get_val_transform(img_size),
        )
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        model = Specialist(pretrained=True).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

        metrics = evaluate_full(model, valloader, device)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
