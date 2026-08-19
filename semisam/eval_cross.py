"""Unified cross-dataset evaluation. Usage: python eval_cross.py"""
import sys; sys.path.insert(0, '.')
import csv, os, torch, numpy as np, cv2
from pathlib import Path
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from model.structure_risk import StructureRiskHeads
from torch.utils.data import DataLoader
from utils.metrics import dice_score

device = 'cuda'
IMG_SZ = (640, 640)

# (name, ckpt_path, has_risk_heads, has_upfm, has_fusion)
CKPTS = [
    ("T1_baseline",   "snapshots/tut_baseline_5pct/ssl_iter_10000_dice_0.7468.pth", False, False, False),
    ("T2_upfm",       "snapshots/semi_sam_upfm/ssl_iter_9400_dice_0.7532.pth",      False, True,  False),
    ("T3_scale_upfm", "snapshots/t3_scale_upfm/ssl_iter_6200_dice_0.7463.pth",       False, True,  False),
    ("E5TUT_full",    "snapshots/e5_tut_5pct/ssl_iter_4800_dice_0.7344.pth",        False, True,  False),
    ("T4_refactor",   "snapshots/t4_refactor/best_model.pth",                       False, True,  True),
    ("P4_topo2k",     "snapshots/topo_2k/best_model.pth",                           False, False, False),
    ("P7_toposam_v3", "snapshots/toposam_2k_v3/best_model.pth",                     True,  False, False),
    ("P9_agree",      "snapshots/toposam_agree/best_model.pth",                     True,  False, False),
]

DATASETS = [
    ("TUT_test",  "/srv/shared/data/crack_detection/TUT/test_img", "/srv/shared/data/crack_detection/TUT/test_lab"),
    ("TUT_val",   "/srv/shared/data/crack_detection/TUT/val_img",  "/srv/shared/data/crack_detection/TUT/val_lab"),
    ("ustCrack",     "/srv/shared/data/crack_detection/ustCrack/images", "/srv/shared/data/crack_detection/ustCrack/masks"),
    ("asphalt3k",    "/srv/shared/data/crack_detection/asphalt3k/images", "/srv/shared/data/crack_detection/asphalt3k/labels"),
    ("concrete3k",   "/srv/shared/data/crack_detection/concrete3k/images", "/srv/shared/data/crack_detection/concrete3k/labels"),
    ("omnicrack30k", "/srv/shared/data/crack_detection/omnicrack30k/images/test", "/srv/shared/data/crack_detection/omnicrack30k/annotations/test"),
]

mean_arr = np.array([0.485, 0.456, 0.406])
std_arr  = np.array([0.229, 0.224, 0.225])

# Cache datasets to avoid reloading
dataset_cache = {}
all_rows = []

for ds_name, img_dir, mask_dir in DATASETS:
    print(f"\n{'='*60}\nDataset: {ds_name}")
    ds = CrackDataset(img_dir, mask_dir, transform=get_val_transform(IMG_SZ))
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"  {len(ds)} images")

    for exp_name, ckpt_path, has_risk, has_upfm, has_fusion in CKPTS:
        model = Specialist(pretrained=True).to(device)
        if has_risk:
            model.add_structure_risks(StructureRiskHeads(32).to(device))
        # ponytail: UPFM/variance/fusion removed — old ckpts load via strict=False (dead keys ignored)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
        model.eval()

        dices = []
        viz_entries = []

        for i, batch in enumerate(dl):
            img_t = batch['image'].to(device)
            label = batch['label']
            img_np = img_t[0].permute(1,2,0).cpu().numpy()
            img_np = np.clip(img_np * std_arr + mean_arr, 0, 1)
            img = (img_np * 255).astype(np.uint8)

            with torch.no_grad():
                out = model(img_t)
                pred = (torch.sigmoid(out['mask_logits']) > 0.5).float()
            pred_np = pred[0,0].cpu().numpy().astype(bool)
            gt_np = label[0,0].cpu().numpy().astype(bool)

            d = dice_score(pred_np, gt_np)
            dices.append(d)

            overlay = img.copy()
            overlay[pred_np] = (overlay[pred_np]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
            overlay[gt_np]   = (overlay[gt_np]*0.5   + np.array([255,0,0])*0.5).astype(np.uint8)
            both = pred_np & gt_np
            overlay[both] = (overlay[both]*0.5 + np.array([255,255,0])*0.5).astype(np.uint8)
            cv2.putText(overlay, f'{exp_name} D={d:.3f}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            viz_entries.append((i, d, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)))

        mean_d = np.mean(dices)
        std_d  = np.std(dices)
        best_d = max(dices)
        worst_d = min(dices)
        n_zero = sum(1 for d in dices if d < 1e-6)
        print(f"  {exp_name:<16} mean={mean_d:.4f}  std={std_d:.4f}  best={best_d:.4f}  worst={worst_d:.4f}  zero={n_zero}")
        all_rows.append([ds_name, exp_name, f"{mean_d:.4f}", f"{std_d:.4f}", f"{best_d:.4f}", f"{worst_d:.4f}", str(n_zero)])

        # P4: dump per-image Dice CSV
        per_img_dir = Path(f'viz_results/{ds_name}/{exp_name}')
        per_img_dir.mkdir(parents=True, exist_ok=True)
        with open(per_img_dir / 'per_image_dice.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["image_idx", "dice"])
            for idx, d in enumerate(dices):
                w.writerow([idx, f"{d:.6f}"])

        # Select 10 representative: percentile-sampled
        viz_entries.sort(key=lambda x: x[1])
        n = len(viz_entries)
        selected = [viz_entries[0], viz_entries[-1]]  # worst + best
        for p in [5, 15, 30, 50, 70, 85, 95, 99]:
            selected.append(viz_entries[min(int(n * p / 100), n-1)])
        seen = set(); unique = []
        for v in selected:
            if v[0] not in seen: unique.append(v); seen.add(v[0])

        out_dir = Path(f'viz_results/{ds_name}/{exp_name}')
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, d_val, ov in unique:
            cv2.imwrite(str(out_dir / f'img_{idx:04d}_dice{d_val:.3f}.png'), ov)

        del model

# Write master CSV
csv_path = 'cross_dataset_eval.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["Dataset", "Experiment", "Mean Dice", "Std Dice", "Best Dice", "Worst Dice", "Zero-Dice Images"])
    w.writerows(all_rows)
print(f"\nSaved {csv_path}")
print(f"Viz structure: viz_results/{{dataset}}/{{experiment}}/")
