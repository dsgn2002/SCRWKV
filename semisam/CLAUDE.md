# CLAUDE.md — SemiSAM-Crack: Semi-Supervised Infrastructure Defect Segmentation

## 1. Project Context

Vision-based crack segmentation (concrete, pavement, masonry) with very few labels.
- **Input**: 2D RGB 640×640 images (UAV/robot/handheld)
- **Target**: binary crack mask
- **Dataset**: TUT crack detection (987 train / 139 val / 282 test) at `/srv/shared/data/crack_detection/TUT/`
- **Training**: 5% labels (49 labeled, 938 unlabeled) via `data/tut_5pct/`
- **Reference**: Zhang et al., "SemiSAM+", Medical Image Analysis, 2025
- **Adaptation**: Lu et al., "UnCoL", IEEE TMI, 2026

### Novelty boundary vs SemiSAM+ (paper positioning)
SemiSAM+ quantifies segmentation uncertainty as the **variance across masks produced
under different prompts** (Eq. 4: `U_x = D[F(x,p1),…,F(x,pn)]`), then uses it to
down-weight SAM supervision (Eq. 5). This prompt-disagreement variance is informative
for 3D blob-like organs but **near-degenerate for 2D thin cracks** — a crack mask
barely changes under box jitter, so mask-variance ≈ 0 and carries little signal.

SemiSAM-Crack's central contribution is therefore **structure-aware uncertainty
quantification based on topology and morphology**: `R_topo` (endpoint density /
tortuosity via DeformConv following crack curvature) and `R_morph` (boundary
geometry via dilated conv) capture the structural uncertainty that mask-variance
cannot in 2D.

- **Borrowed from SemiSAM+ (cited, NOT claimed as novel):** U_geo
  (prompt-disagreement mask variance) and the U_geo acceptance gate.
- **Genuine deltas:** (1) structure-aware risk heads (R_topo, R_morph), (2)
  per-connected-component selective SAM querying, (3) learned two-teacher blend
  `α·SAM + (1−α)·EMA` with error-type routing, (4) measured-target gate
  supervision (P13: `dice_sam − dice_ema` on labeled components).

## 2. Method: Structure-Aware Selective Consultation

### Architecture
```
Specialist (ResNet34-UNet or SegFormer-B1)
  ├── mask head → crack probability
  └── StructureRiskHeads → R_topo, R_morph        (TWO heads, no app head)
        ├── Topo:  DeformConv2d (snake-like, follows crack curvature)
        └── Morph: dilated conv  (wider boundary context)

Generalist (frozen SAM-1, crack fine-tuned)
  └── box-prompt → pseudo-labels

EMA Teacher → consistency + pseudo-labels
```

### SCRWKV specialist integration (18 August 2026)

SCRWKV is available as a third specialist option through
`model/factory.py`; the existing SegFormer-B1 configuration remains the
default. Set `model.specialist_encoder: scrwkv` or use
`config_scrwkv.yaml`. The source is pinned to the `semisam-integration`
branch of `dsgn2002/SCRWKV` at `/home/guest/dinu/SCRWKV/SCRWKV`.

- `model/specialist_scrwkv.py` converts SemiSAM's ImageNet-normalized images
  to SCRWKV's `[-1,1]` normalization internally.
- SCRWKV returns the existing `mask_logits` and `features` interface plus its
  four-scale SFE pyramid, CSHF harmonic feature and scale-attention map.
- Existing topology/morphology heads run at `scrwkv_risk_scale=0.25` and are
  upsampled to the input resolution. This keeps the deformable risk head from
  operating on the full 640×640 grid.
- Training uses AdamW, matching the architecture's original optimization
  family. The standardized initial experiment remains 2,000 steps (~83
  epochs with 24 steps/epoch).
- Verified in `sam3_crack`: rectangular forward/backward, strict state loading,
  full 640×640 forward, and batch-4 student+EMA backward. The last test peaked
  at approximately 15.0 GiB allocated GPU memory on an RTX 4090.

This integration changes only the specialist representation. It does not yet
change SAM pseudo-label generation, uncertainty calibration or the SSL loss.

### Component-Level Selective SAM Query
```
EMA prediction
 → extract_components() → {C_1, ..., C_k}
 → per-component box → SAM query (predict + predict_unc)
 → U_geo acceptance gate (< 0.3)
 → pool_risk_maps(learned R_topo, R_morph) over component
 → error-type routing (topo→SAM, app→EMA, morph→blend)
```

### Losses (current)
```
L_sup = BCE + Dice + 0.1 × soft_clDice + 0.05 × L_risk  (every iter, after warmup)
L_risk = 2.0 × MSE(R_topo, endpoint_density) + 1.0 × MSE(R_morph, boundary_distance)
L_total = L_sup + consistency_weight × L_consistency + 0.1 × L_sam
```

### Decision Pipeline

| Stage | Type | What |
|-------|------|------|
| Specialist encoder+decoder | Learned | ResNet34-UNet or SegFormer-B1 (13.9M) |
| mask head | Learned | Conv2d(→1) |
| StructureRiskHeads | **Learned — CORE NOVELTY** | **TWO heads**: DeformConv2d (topo) + dilated conv (morph); quantifies topology/morphology uncertainty that mask-variance (SemiSAM+ U_geo) cannot capture in 2D |
| Risk pooling per component | Learned | Mean-pool R_maps over CC mask |
| Component extraction (CC, skeleton) | Heuristic | cv2 + skimage |
| SAM box prompts per component | Heuristic | CC bboxes + margin |
| Acceptance gate (U_geo < 0.3) | Heuristic (**from SemiSAM+**) | Mask-variance; near-degenerate for 2D cracks — motivates the structure-risk heads |
| **Consultation gate (P13)** | **Learned** | **α=σ(2·Δ̂); MSE(Δ̂, dice_sam−dice_ema) on labeled comps; 8 feats → 32→1→tanh** |
| **Cross-attention consultation (P15/P16)** | **Learned — FALSIFIED** | `CrossAttentionConsultation` in `model/consultation.py`: morph tokens query topo tokens (1-head, d=64, dropout 0.3). P15: 4+4 scalars (0.775). P16: pooled decoder feats topo 512d + morph 768d (0.776). Both below baseline 0.781 |
| **Learned quality selector (P17)** | **Learned — FALSIFIED** | `QualitySelector` in `model/quality_selector.py`: SAM↔EMA cross-attention on prob maps + decoder feats → per-pixel w(p), dense BCE on labeled pixels, replaces (1−U) in L_sam (WSEL=1). 0.774 — below baseline |

### Selector falsification (2k iters, SegFormer-B1, TUT test)
| Model | Dice | clDice | skel_prec | skel_rec | frags | Val |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (hand-crafted 1−U)** | **0.781** | **0.143** | **0.737** | 0.749 | **5.4** | 0.739 |
| P14 MLP gate (8 scalars) | 0.783 | — | — | — | — | 0.742 |
| P15 XAttn (4+4 scalars) | 0.775 | 0.131 | 0.697 | 0.752 | 6.6 | 0.729 |
| P16 XAttn (pooled 512+768) | 0.776 | 0.135 | 0.715 | 0.751 | 6.1 | 0.738 |
| P17 learned w (dense BCE) | 0.774 | 0.134 | 0.712 | 0.752 | 6.5 | 0.736 |

**Four-way falsification of selective SAM consultation at 5% labels.** All learned
selectors land 0.774–0.783 vs baseline 0.781. Structural conclusions:
1. Pseudo-label filtering has nothing to filter — TUT pseudo-labels are good where
   used, ambiguous where no quality signal exists in the input.
2. Any selector trained on 49 images overfits (−0.005..−0.007 at test).
3. β_sam's global fade-out ramp is already the correct quality schedule.

**What works:** three-head structure-aware supervision (+0.010 test at 10k:
0.789 vs 0.779), topology-aware evaluation, SegFormer-B1 backbone (P14).
**Next levers:** label budget (TUT 10%), risk-head supervision — not more selectors.

## 3. Key Experiments

| # | Experiment | Val Dice | Test Dice | Key config |
|---|-----------|----------|-----------|------------|
| T2 | UPFM only | 0.753 | — | UPFM+MT+SAM, 10K iters |
| T4 | Refactor | 0.749 | 0.796 | Detach+invert+fuse, 10K iters |
| P4 | topo_2k | 0.711 | 0.756 | aux centerline/boundary heads, 2K iters |
| P5 | toposam_2k | 0.684 | 0.726 | Simplified: no aux heads, broken rampup, 2K iters |
| P6 | toposam_1k | 0.639 | 0.679 | Risk heads + fixed rampup, 1K iters |
| P7 | toposam_2k_v3 | 0.711 | 0.752 | Risk heads + WTA routing, 2K iters |
| P8 | toposam_2h_2k | 0.667 | 0.723 | Risk heads WTA (2-head, no topo), 2K iters |
| **P9** | **toposam_agree** | **0.712** | **0.755** | **Agreement-gated blend: α=clip(R_topo×R_morph,0.2,0.8), 2K iters** |
| P10 | tut_10pct_agree | 0.684 | — | P9 config + 10% labels (99 labeled), clip-agreement degraded |
| P11 | tut_10pct_consult | 0.723 | 0.758 | 10% + consultation MLP gate — fixed degradation, beats P9 |
| P12 | tut_5pct_consult | 0.678 | 0.723 | 5% + MLP gate (topo/morph proxy target) — loses to P9 |
| **P13** | **tut_5pct_consult_gt** | **0.715** | **0.757** | **5% + MLP gate w/ measured dice_sam−dice_ema target on labeled comps — fixes P12 regression, beats P9** |
| **P14** | **tut_5pct_segformer_b1** | **0.742** | **0.783** | **SegFormer-B1 (13.9M) + full method — architecture-agnostic, beats UNet P13 by +0.026 test** |
| FS-UNet | tut_fullysup_unet | 0.785 | **0.818** | Fully-supervised upper bound (987 labels, no SAM/SSL) — the ceiling; P13 reaches 92.5% of it at 5% labels |
| FS-SegFormer | tut_fullysup_segformer | 0.786 | **0.826** | Fully-supervised SegFormer-B1 ceiling (AdamW); P14 reaches 94.8% of it at 5% labels |

### P9 Cross-Dataset (agreement-gated, 2K iters)
| Dataset | P4 (aux heads) | P7 (WTA) | P9 (agree) | Δ (P9−P7) |
|---------|---------------|----------|------------|------------|
| TUT test | 0.756 | 0.752 | **0.755** | **+0.003** |
| ustCrack | 0.234 | 0.221 | 0.231 | +0.010 |
| asphalt3k | 0.427 | **0.467** | 0.389 | −0.078 |
| omnicrack30k | 0.277 | 0.286 | — | — |

P9 beats P7 on TUT and ustCrack, loses on asphalt3k — agreement gate favors in-domain.

## 4. File Structure

```
SemiSAM/
├── train.py                    # Two-stage: pretrain() + ssl_train()
├── config.yaml                 # All experiment config
├── evaluate.py                 # Validation + full eval loop
├── eval_vis.py                 # Per-image predicted-mask overlay
├── eval_cross.py               # Cross-dataset evaluation
├── data/
│   ├── dataset.py              # CrackDataset + TwoStreamBatchSampler
│   └── transforms.py           # Albumentations: train/val/strong/weak
├── model/
│   ├── specialist.py           # ResNet-UNet: mask head + StructureRiskHeads/consultation slots
│   ├── specialist_segformer.py # SegFormer-B1/B2 specialist (drop-in, same interface)
│   ├── structure_risk.py       # StructureRiskHeads: DeformConv2d(topo) + dilated(morph) — TWO heads
│   ├── consultation.py         # ConsultationPredictor (MLP) + CrossAttentionConsultation
│   ├── quality_selector.py     # Per-pixel pseudo-label quality (SAM↔EMA cross-attn)
│   └── generalist_wrapper.py   # SAM-1/SAM-3 wrapper
├── losses/
│   ├── supervised.py           # BCE+Dice, heteroscedastic_loss, FocalLoss
│   ├── topology.py             # soft_cl_dice_loss()
│   ├── consistency.py          # MSE consistency (Mean Teacher)
│   ├── confidence_aware.py     # Boundary-split confidence-aware L_sam
│   └── boundary.py             # cv2 boundary band mask
├── training/
│   ├── dual_teacher.py         # Entropy, pseudo-label fusion (BCP fns now unused)
│   └── risk_targets.py         # Per-component GT risk scores
├── prompting/
│   ├── boxes.py                # boxes_from_mask(), jitter_boxes()
│   ├── components.py           # CrackComponent + extract_components()
│   └── skeleton.py             # Skeleton extraction + point sampling
├── uncertainty/
│   └── geo.py                  # U_geo: box-perturbation disagreement (U_app/U_scale/fusion removed)
└── utils/
    ├── config.py               # Config dataclasses
    ├── metrics.py              # Dice, IoU, HD95, ASD, clDice
    ├── topology_metrics.py     # skeleton precision/recall, endpoint error
    └── ramps.py                # Sigmoid rampup
```

## 5. Usage

```bash
# Train (SSL stage, from scratch)
python train.py --config config.yaml --stage ssl --snapshot snapshots/my_experiment

# Evaluate on TUT test set
python evaluate.py --checkpoint snapshots/my_experiment/best_model.pth

# Cross-dataset eval
python eval_cross.py  # edit CKPTS list first

# Visualize predictions
python eval_vis.py    # edit checkpoint path in script
```
