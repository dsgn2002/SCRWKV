"""Cross-dataset evaluation on ustCrack (246 images)."""
import sys; sys.path.insert(0, '.')
import csv, os, torch, numpy as np, cv2
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from torch.utils.data import DataLoader
from utils.metrics import dice_score

device = 'cuda'
DST = '/srv/shared/data/crack_detection/ustCrack'

val_ds = CrackDataset(f'{DST}/images', f'{DST}/masks',
                       transform=get_val_transform((640, 640)))
val_ldr = DataLoader(val_ds, batch_size=1, shuffle=False)
print(f"ustCrack: {len(val_ds)} images")

experiments = [
    ("T1_baseline",   "snapshots/tut_baseline_5pct/ssl_iter_10000_dice_0.7468.pth", False),
    ("T2_upfm",       "snapshots/semi_sam_upfm/ssl_iter_9400_dice_0.7532.pth",      True),
    ("T3_scale_upfm", "snapshots/t3_scale_upfm/ssl_iter_6200_dice_0.7463.pth",       True),
    ("E5TUT_full",    "snapshots/e5_tut_5pct/ssl_iter_4800_dice_0.7344.pth",        True),
]

mean_arr = np.array([0.485, 0.456, 0.406])
std_arr  = np.array([0.229, 0.224, 0.225])

results = []

for exp_name, ckpt_path, has_upfm in experiments:
    print(f"\n{'='*50}\n{exp_name}: loading {ckpt_path}")
    model = Specialist(pretrained=True).to(device)
    # ponytail: UPFM/variance removed — old ckpts load via strict=False (dead keys ignored)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    dices = []
    viz_entries = []  # (idx, dice, img, overlay) for picking 10 later

    for i, batch in enumerate(val_ldr):
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

        # Overlay
        overlay = img.copy()
        overlay[pred_np] = (overlay[pred_np]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
        overlay[gt_np]   = (overlay[gt_np]*0.5   + np.array([255,0,0])*0.5).astype(np.uint8)
        both = pred_np & gt_np
        overlay[both] = (overlay[both]*0.5 + np.array([255,255,0])*0.5).astype(np.uint8)
        cv2.putText(overlay, f'{exp_name} Dice={d:.3f}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        viz_entries.append((i, d, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)))

    mean_d = np.mean(dices)
    std_d = np.std(dices)
    best_d = max(dices)
    worst_d = min(dices)
    print(f"  mean={mean_d:.4f}  std={std_d:.4f}  best={best_d:.4f}  worst={worst_d:.4f}")
    results.append([exp_name, f"{mean_d:.4f}", f"{std_d:.4f}", f"{best_d:.4f}", f"{worst_d:.4f}"])

    # Select 10 representative images: best, worst, 8 sampled by percentile
    viz_entries.sort(key=lambda x: x[1])
    n = len(viz_entries)
    selected = [viz_entries[0], viz_entries[-1]]  # worst + best
    percentiles = [5, 15, 30, 50, 70, 85, 95, 99]
    for p in percentiles:
        idx = min(int(n * p / 100), n-1)
        selected.append(viz_entries[idx])
    # Deduplicate
    seen = set()
    unique = []
    for v in selected:
        if v[0] not in seen:
            unique.append(v)
            seen.add(v[0])

    out_dir = f'viz_ustcrack_{exp_name}'
    os.makedirs(out_dir, exist_ok=True)
    for idx, d, ov in unique:
        cv2.imwrite(f'{out_dir}/img_{idx:04d}_dice{d:.3f}.png', ov)
    print(f"  Saved {len(unique)} viz to {out_dir}/")

    del model

# Save CSV
with open('ustcrack_cross_eval.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["Experiment", "Mean Dice", "Std Dice", "Best Dice", "Worst Dice"])
    w.writerows(results)
print(f"\nSaved ustcrack_cross_eval.csv")
for r in results:
    print(f"  {r[0]:<20}  mean={r[1]:>6}  std={r[2]:>6}  best={r[3]:>6}  worst={r[4]:>6}")
