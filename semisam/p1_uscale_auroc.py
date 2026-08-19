"""P1: Test whether U_scale discriminates cracks from distractors (false positives).

Loads T1 baseline (no UPFM), runs forward_multiscale on TUT val,
computes U_scale per pixel, builds crack-vs-distractor labels,
computes AUROC, saves heatmap visualizations.

Decision: AUROC < 0.6  → U_scale is NOT a crack discriminator
          AUROC >= 0.6 → implement P2b (positive evidence weighting)
"""
import sys; sys.path.insert(0, '.')
import csv, os, torch, numpy as np, cv2
from pathlib import Path
from sklearn.metrics import roc_auc_score
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from uncertainty.scale import compute_u_scale
from torch.utils.data import DataLoader

device = 'cuda'
OUT = Path("p1_output"); OUT.mkdir(exist_ok=True)

# --- Load model ---
print("Loading T1 baseline...")
model = Specialist(pretrained=True).to(device)
ckpt = torch.load("snapshots/tut_baseline_5pct/ssl_iter_10000_dice_0.7468.pth",
                   map_location=device, weights_only=True)
model.load_state_dict(ckpt, strict=False)
model.eval()

# --- Load TUT val ---
ds = CrackDataset("data/tut_5pct/val/images", "data/tut_5pct/val/masks",
                   transform=get_val_transform((640, 640)))
dl = DataLoader(ds, batch_size=1, shuffle=False)
print(f"TUT val: {len(ds)} images")

mean_arr = np.array([0.485, 0.456, 0.406])
std_arr  = np.array([0.229, 0.224, 0.225])

all_u_crack = []       # U_scale values at GT crack pixels
all_u_distractor = []  # U_scale values at FP (pred=1, GT=0) pixels
per_img_stats = []
heatmap_entries = []   # (img_idx, u_scale_map, gt_map, pred_map, overlay)

for idx, batch in enumerate(dl):
    img_t = batch['image'].to(device)
    label = batch['label']

    # forward_multiscale expects BCHW input
    with torch.no_grad():
        out = model.forward_multiscale(img_t, scales=[0.5, 1.0, 2.0])
        pred_logits = out['mask_logits']
        pred_1x = torch.sigmoid(pred_logits)
        aligned_preds = out['aligned_preds']  # list of (B,1,H,W) already sigmoid
        u_scale = compute_u_scale(aligned_preds)  # [B, 1, H, W]

    pred_bin = (pred_1x > 0.5).float()

    u_np = u_scale[0, 0].cpu().numpy()
    pred_np = pred_bin[0, 0].cpu().numpy().astype(bool)
    gt_np = label[0, 0].cpu().numpy().astype(bool)

    # Crack pixels: GT = 1
    crack_mask = gt_np
    # Distractor pixels: model predicts crack but GT says no = false positives
    distractor_mask = pred_np & (~gt_np)

    u_crack = u_np[crack_mask]
    u_dist = u_np[distractor_mask]

    if len(u_crack) > 0:
        all_u_crack.extend(u_crack.tolist())
    if len(u_dist) > 0:
        all_u_distractor.extend(u_dist.tolist())

    per_img_stats.append({
        'idx': idx,
        'n_crack_px': int(crack_mask.sum()),
        'n_distractor_px': int(distractor_mask.sum()),
        'u_crack_mean': float(u_crack.mean()) if len(u_crack) > 0 else 0.0,
        'u_distractor_mean': float(u_dist.mean()) if len(u_dist) > 0 else 0.0,
    })

    # Store for heatmap viz (first 15 images)
    if idx < 15:
        img_np = img_t[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * std_arr + mean_arr, 0, 1)
        img_rgb = (img_np * 255).astype(np.uint8)
        heatmap_entries.append((idx, u_np, gt_np, pred_np, img_rgb))

    if idx % 20 == 0:
        print(f"  [{idx}/{len(ds)}] crack_px={crack_mask.sum()} distractor_px={distractor_mask.sum()}")

# --- Compute AUROC ---
n_crack = len(all_u_crack)
n_dist = len(all_u_distractor)
print(f"\nTotal crack pixels: {n_crack:,}")
print(f"Total distractor pixels: {n_dist:,}")

# Balance classes for AUROC
n_sample = min(n_crack, n_dist)
if n_sample < 100:
    print("ERROR: too few pixels for meaningful AUROC")
    sys.exit(1)

rng = np.random.RandomState(42)
crack_sample = rng.choice(all_u_crack, n_sample, replace=False)
dist_sample = rng.choice(all_u_distractor, n_sample, replace=False)

X = np.concatenate([crack_sample, dist_sample])
y = np.concatenate([np.ones(n_sample), np.zeros(n_sample)])

auroc = roc_auc_score(y, X)
crack_mean_u = np.mean(all_u_crack)
dist_mean_u = np.mean(all_u_distractor)

print(f"\n{'='*50}")
print(f"AUROC (U_scale → crack vs distractor): {auroc:.4f}")
print(f"Mean U_scale at crack pixels:         {crack_mean_u:.4f}")
print(f"Mean U_scale at distractor pixels:    {dist_mean_u:.4f}")
print(f"Separation (crack − distractor):      {crack_mean_u - dist_mean_u:.4f}")

if auroc < 0.6:
    print(f"\nDECISION: AUROC={auroc:.4f} < 0.6 → U_scale is NOT a crack-vs-distractor discriminator.")
    print("  → Drop the discriminator claim. Skip P2b (positive evidence weighting).")
    print("  → Keep U_scale as a general uncertainty gate (P2a: remove from gate).")
else:
    print(f"\nDECISION: AUROC={auroc:.4f} >= 0.6 → U_scale IS a crack-vs-distractor discriminator.")
    print("  → Implement P2b: positive evidence weighting (1 + alpha * U_scale * p_sam).")

# --- Save CSV ---
with open(OUT / "p1_auroc_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["auroc", f"{auroc:.4f}"])
    w.writerow(["n_crack_pixels", n_crack])
    w.writerow(["n_distractor_pixels", n_dist])
    w.writerow(["u_crack_mean", f"{crack_mean_u:.4f}"])
    w.writerow(["u_distractor_mean", f"{dist_mean_u:.4f}"])

with open(OUT / "p1_per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=per_img_stats[0].keys())
    w.writeheader()
    w.writerows(per_img_stats)

# --- Save heatmap visuals ---
heatmap_dir = OUT / "heatmaps"
heatmap_dir.mkdir(exist_ok=True)

for idx, u_map, gt, pred, img in heatmap_entries:
    # U_scale heatmap (jet colormap)
    u_vis = (np.clip(u_map, 0, 1) * 255).astype(np.uint8)
    u_color = cv2.applyColorMap(u_vis, cv2.COLORMAP_JET)

    # Composite: left=original, center=GT (red), right=U_scale heatmap
    gt_overlay = img.copy()
    gt_overlay[gt] = (gt_overlay[gt] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    # Mark FPs in blue
    fp = pred & (~gt)
    gt_overlay[fp] = (gt_overlay[fp] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    row = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                     gt_overlay,
                     u_color])
    cv2.putText(row, f"img {idx} | U_scale mean crack={per_img_stats[idx]['u_crack_mean']:.3f} dist={per_img_stats[idx]['u_distractor_mean']:.3f}",
                (10, row.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(heatmap_dir / f"img_{idx:03d}.png"), row)

print(f"\nSaved: {OUT}/p1_auroc_summary.csv, p1_per_image.csv, heatmaps/")
