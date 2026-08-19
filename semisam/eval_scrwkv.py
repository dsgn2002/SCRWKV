"""Evaluate the SCRWKV specialist on the TUT test split."""

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
from model.consultation import ConsultationPredictor
from model.factory import build_specialist
from model.structure_risk import StructureRiskHeads
from utils.metrics import cl_dice, dice_score, iou_score
from utils.topology_metrics import (
    fragmentation_count,
    skeleton_precision,
    skeleton_recall,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="snapshots/tut_5pct_scrwkv/best_model.pth",
    )
    parser.add_argument(
        "--image-dir", default="/srv/shared/data/crack_detection/TUT/test_img"
    )
    parser.add_argument(
        "--mask-dir", default="/srv/shared/data/crack_detection/TUT/test_lab"
    )
    parser.add_argument("--scrwkv-root", default="/home/guest/dinu/SCRWKV/SCRWKV")
    parser.add_argument("--output-dir", default="snapshots/tut_5pct_scrwkv")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("SCRWKV evaluation requires CUDA")

    model = build_specialist(
        "scrwkv", pretrained=False, scrwkv_root=args.scrwkv_root,
        scrwkv_risk_scale=0.25,
    ).to(device)
    model.add_structure_risks(StructureRiskHeads(model._final_ch).to(device))
    model.add_consultation_predictor(
        ConsultationPredictor(in_features=8, hidden=32).to(device)
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    dataset = CrackDataset(
        args.image_dir,
        args.mask_dir,
        transform=get_val_transform((640, 640)),
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, ncols=70)):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            prediction = (
                torch.sigmoid(model(image)["mask_logits"]) > 0.5
            ).float()
            pred = prediction[0, 0].cpu().numpy().astype(bool)
            gt = label[0, 0].cpu().numpy().astype(bool)
            if gt.sum() == 0:
                continue
            rows.append({
                "image_idx": index,
                "dice": dice_score(pred, gt),
                "iou": iou_score(pred, gt),
                "cldice": cl_dice(pred, gt),
                "skel_prec": skeleton_precision(pred, gt),
                "skel_rec": skeleton_recall(pred, gt),
                "frags": fragmentation_count(pred),
            })

    if not rows:
        raise RuntimeError("No non-empty TUT test masks were evaluated")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_per_image_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    means = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("dice", "iou", "cldice", "skel_prec", "skel_rec", "frags")
    }
    summary = (
        f"SCRWKV TUT test: Dice={means['dice']:.4f}  "
        f"IoU={means['iou']:.4f}  clDice={means['cldice']:.4f}  "
        f"skel_prec={means['skel_prec']:.3f}  "
        f"skel_rec={means['skel_rec']:.3f}  "
        f"frags={means['frags']:.1f}  n={len(rows)}"
    )
    (output_dir / "test_eval.log").write_text(summary + "\n")
    print("\n" + summary)
    print(f"Per-image metrics: {csv_path}")


if __name__ == "__main__":
    main()
