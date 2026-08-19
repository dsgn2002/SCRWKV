"""P14 test eval — SegFormer-B1 specialist + measured-target consultation gate, 5%."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist_segformer import SpecialistSegFormer
from model.structure_risk import StructureRiskHeads
from model.consultation import ConsultationPredictor
from utils.metrics import dice_score, iou_score, cl_dice
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda'
model = SpecialistSegFormer('mit_b1', pretrained=False).to(device)  # weights come from ckpt
model.add_structure_risks(StructureRiskHeads(model._final_ch).to(device))
model.add_consultation_predictor(ConsultationPredictor(in_features=8, hidden=32).to(device))
model.load_state_dict(
    torch.load('snapshots/tut_5pct_segformer_b1/best_model.pth',
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

print(f'\nP14 TUT test: dice={np.mean(dices):.4f}  iou={np.mean(ious):.4f}  '
      f'cldice={np.mean(cldices):.4f}  n={len(dices)}')
