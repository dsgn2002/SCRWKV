#!/bin/bash
# Train + evaluate consultation experiment.
#
# Usage:
#   bash train.sh                    # MLP gate (ablation baseline)
#   XATTN=1 bash train.sh            # cross-attention fusion
#
# Runs in tmux session "consult" on GPU 0. Evaluation dumps TUT-test
# metrics to <snapshot>/eval_results.log.

set -e
SNAP=${SNAP:-snapshots/consult_$( [ "${XATTN:-0}" = "1" ] && echo xattn || echo mlp )}
GPU=${GPU:-0}
SESSION=${SESSION:-consult}

mkdir -p "$SNAP"

tmux new-session -d -s "$SESSION" 2>/dev/null || true
tmux send-keys -t "$SESSION" C-c 2>/dev/null || true

# --- training ---
tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --config config.yaml --stage ssl --snapshot $SNAP 2>&1 \
  | tee $SNAP/console.log" Enter

echo "Training launched in tmux session '$SESSION' → $SNAP"
echo "Monitor:  tmux attach -t $SESSION"
echo ""
echo "When training finishes, run the eval block below (also from this script):"
echo "  EVAL_SNAP=$SNAP bash train.sh eval"

# --- evaluation (run after training completes) ---
if [ "${1:-}" = "eval" ]; then
  BEST=$(ls $SNAP/ssl_iter_*dice_*.pth 2>/dev/null | sed 's/.*dice_//;s/.pth//' | sort -rn | head -1)
  CKPT=$(ls $SNAP/*dice_${BEST}.pth | head -1)
  echo "Evaluating best checkpoint: $CKPT (val $BEST)"

  tmux send-keys -t "$SESSION" "python3 -c \"
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from data.dataset import CrackDataset
from data.transforms import get_val_transform
from model.specialist_segformer import SpecialistSegFormer
from torch.utils.data import DataLoader
from utils.metrics import dice_score, cl_dice
from utils.topology_metrics import skeleton_precision, skeleton_recall, fragmentation_count
device = 'cuda'
ds = CrackDataset('/srv/shared/data/crack_detection/TUT/test_img',
                   '/srv/shared/data/crack_detection/TUT/test_lab',
                   transform=get_val_transform((640, 640)))
dl = DataLoader(ds, batch_size=1, shuffle=False)
model = SpecialistSegFormer('mit_b1', pretrained=False).to(device)
model.load_state_dict(torch.load('$CKPT', map_location=device, weights_only=True), strict=False)
model.eval()
dices, cldices, sp, sr, frags = [], [], [], [], []
with torch.no_grad():
    for batch in dl:
        img = batch['image'].to(device); label = batch['label']
        pred = (torch.sigmoid(model(img)['mask_logits']) > 0.5).float()
        pn = pred[0,0].cpu().numpy().astype(bool)
        gn = label[0,0].cpu().numpy().astype(bool)
        dices.append(dice_score(pn, gn)); cldices.append(cl_dice(pn, gn))
        sp.append(skeleton_precision(pn, gn)); sr.append(skeleton_recall(pn, gn))
        frags.append(fragmentation_count(pn))
print(f'Dice={np.mean(dices):.4f}  clDice={np.mean(cldices):.4f}  '
      f'skel_prec={np.mean(sp):.3f}  skel_rec={np.mean(sr):.3f}  frags={np.mean(frags):.1f}')
\" 2>&1 | tee $SNAP/eval_results.log" Enter
  echo "Eval launched in '$SESSION' → $SNAP/eval_results.log"
fi
