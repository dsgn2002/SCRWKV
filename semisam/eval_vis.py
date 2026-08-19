"""Save validation predictions overlaid on original images."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, cv2, os
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from torch.utils.data import DataLoader

device = 'cuda'
model = Specialist(pretrained=True).to(device)
model.load_state_dict(torch.load(
    'snapshots/semisam_crack_box/best_model.pth',
    map_location=device, weights_only=True), strict=False)
model.eval()

val_ds = CrackDataset('data/val/images', 'data/val/masks',
                       transform=get_val_transform((640, 640)))
val_ldr = DataLoader(val_ds, batch_size=1, shuffle=False)

out_dir = 'viz_eval'
os.makedirs(out_dir, exist_ok=True)

# Denormalize for display
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, batch in enumerate(val_ldr):
    if i >= 20: break
    img_t = batch['image']
    label = batch['label']

    # Denormalize image
    img_np = img_t[0].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * std + mean, 0, 1)
    img = (img_np * 255).astype(np.uint8)

    with torch.no_grad():
        pred = (torch.sigmoid(model(img_t.to(device))['mask_logits']) > 0.5).float()
    pred_np = pred[0, 0].cpu().numpy().astype(bool)
    gt_np = label[0, 0].cpu().numpy().astype(bool)

    # Overlay: green=pred, red=GT, yellow=both
    overlay = img.copy()
    overlay[pred_np] = (overlay[pred_np] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[gt_np] = (overlay[gt_np] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    both = pred_np & gt_np
    overlay[both] = (overlay[both] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)

    from utils.metrics import dice_score
    d = dice_score(pred_np, gt_np)
    cv2.putText(overlay, f'Dice={d:.3f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(f'{out_dir}/val_{i:03d}_dice{d:.3f}.png',
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f'Saved {min(20, len(val_ds))} images to {out_dir}/')
print('Legend: green=pred, red=GT, yellow=both')
