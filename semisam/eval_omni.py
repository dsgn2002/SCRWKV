"""Evaluate topo_2K on full omnicrack30k test split."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, os
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from utils.metrics import dice_score, iou_score, hd95, boundary_iou, boundary_f1, cl_dice
from torch.utils.data import DataLoader

src_img = '/srv/shared/data/crack_detection/omnicrack30k/images/test'
src_ann = '/srv/shared/data/crack_detection/omnicrack30k/annotations/test'

device = 'cuda'
model = Specialist(pretrained=True).to(device)
model.load_state_dict(torch.load('snapshots/topo_2k/best_model.pth', map_location=device, weights_only=True), strict=False)
model.eval()

ds = CrackDataset(src_img, src_ann, transform=get_val_transform((640, 640)))
print(f'omnicrack30k test: {len(ds)} images')

ldr = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

results = {'dice':[], 'iou':[], 'hd95':[], 'biou':[], 'bf1':[], 'cldice':[]}
n_processed = 0
with torch.no_grad():
    for batch in ldr:
        img = batch['image'].to(device)
        label = batch['label'].to(device)
        pred = (torch.sigmoid(model(img)['mask_logits']) > 0.5).float()
        for b in range(pred.shape[0]):
            p = pred[b,0].cpu().numpy().astype(bool)
            g = label[b,0].cpu().numpy().astype(bool)
            if g.sum() == 0: continue
            results['dice'].append(dice_score(p,g))
            results['iou'].append(iou_score(p,g))
            results['hd95'].append(hd95(p,g))
            results['biou'].append(boundary_iou(p,g,5))
            results['bf1'].append(boundary_f1(p,g,5))
            results['cldice'].append(cl_dice(p,g))
        n_processed += 1
        if n_processed % 500 == 0:
            recent = results['dice'][-500:] if results['dice'] else []
            d = np.mean(recent) if recent else 0.0
            print(f'{n_processed}/{len(ds)} | recent_dice={d:.4f} | valid={len(results["dice"])}')

print()
print(f'=== omnicrack30k full test (n={len(results["dice"])}) ===')
for k, v in results.items():
    print(f'{k:>12}: {np.mean(v):.4f}  +/-{np.std(v):.4f}')
