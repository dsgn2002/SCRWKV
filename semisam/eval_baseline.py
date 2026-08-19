"""Fully-supervised UNet baseline — TUT test eval (no risk/consult heads)."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist import Specialist
from utils.metrics import dice_score, iou_score, cl_dice
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda'
model = Specialist('resnet34', (256, 128, 64, 32), 3, pretrained=False).to(device)
model.load_state_dict(
    torch.load('snapshots/tut_fullysup_unet/best_pretrain.pth',
               map_location=device, weights_only=True),
    strict=False,
)
model.eval()

ds = CrackDataset('/srv/shared/data/crack_detection/TUT/test_img',
                  '/srv/shared/data/crack_detection/TUT/test_lab',
                  transform=get_val_transform((640, 640)))
ldr = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

dices, ious, cldices = [], [], []
with torch.no_grad():
    for batch in tqdm(ldr, ncols=70):
        img = batch['image'].to(device); label = batch['label'].to(device)
        pred = (torch.sigmoid(model(img)['mask_logits']) > 0.5).float()
        for b in range(pred.shape[0]):
            p = pred[b, 0].cpu().numpy().astype(bool)
            g = label[b, 0].cpu().numpy().astype(bool)
            if g.sum() == 0:
                continue
            dices.append(dice_score(p, g))
            ious.append(iou_score(p, g))
            cldices.append(cl_dice(p, g))

print(f'\nFully-sup UNet TUT test: dice={np.mean(dices):.4f}  iou={np.mean(ious):.4f}  '
      f'cldice={np.mean(cldices):.4f}  n={len(dices)}')
