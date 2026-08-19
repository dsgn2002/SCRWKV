"""P13 TUT test visualizations: per-image 1x2 grid — GT overlay | prediction overlay."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, cv2, os
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from model.structure_risk import StructureRiskHeads
from model.consultation import ConsultationPredictor
from utils.metrics import dice_score
from torch.utils.data import DataLoader

device = 'cuda'
model = Specialist(pretrained=True).to(device)
model.add_structure_risks(StructureRiskHeads(32).to(device))
model.add_consultation_predictor(ConsultationPredictor(8, 32).to(device))
model.load_state_dict(torch.load(
    'snapshots/tut_5pct_consult_gt/best_model.pth',
    map_location=device, weights_only=True), strict=False)
model.eval()

ds = CrackDataset('/srv/shared/data/crack_detection/TUT/test_img',
                  '/srv/shared/data/crack_detection/TUT/test_lab',
                  transform=get_val_transform((640, 640)))
ldr = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

out_dir = 'viz_p13_tut_test'
os.makedirs(out_dir, exist_ok=True)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
RED, GREEN, WHITE = (255, 0, 0), (0, 255, 0), (255, 255, 255)

dices = []
for i, batch in enumerate(ldr):
    img_t = batch['image']; label = batch['label']
    img_np = img_t[0].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * std + mean, 0, 1)
    img = (img_np * 255).astype(np.uint8)

    with torch.no_grad():
        pred = (torch.sigmoid(model(img_t.to(device))['mask_logits']) > 0.5).float()
    pred_np = pred[0, 0].cpu().numpy().astype(bool)
    gt_np = label[0, 0].cpu().numpy().astype(bool)

    def panel(mask, color, title):
        ov = img.copy()
        ov[mask] = (ov[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        ov = cv2.putText(ov, title, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        ov = cv2.putText(ov, f'Dice={d:.3f}', (10, 65),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        return ov

    d = dice_score(pred_np, gt_np)
    dices.append(d)
    grid = np.hstack([panel(gt_np, RED, 'GT'), panel(pred_np, GREEN, 'Predicted')])
    cv2.imwrite(f'{out_dir}/test_{i:03d}_dice{d:.3f}.png',
                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

print(f'Saved {len(dices)} grids to {out_dir}/')
print(f'mean dice={np.mean(dices):.4f}')
print('Grid: left=GT overlay (red), right=prediction overlay (green)')
