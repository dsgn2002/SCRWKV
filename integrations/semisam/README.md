# SemiSAM integration

These files record the integration used by the SemiSAM workspace on the
`semisam-integration` branch. SCRWKV exposes the specialist interface in
`models/semisam_adapter.py`; the files in this directory are installed on the
SemiSAM side.

## Where SCRWKV is integrated

SCRWKV replaces the **specialist segmentation network**, not SAM. In the
original configuration this specialist is SegFormer-B1 (or ResNet34-UNet in
older experiments). In the SCRWKV configuration, both the trainable student
and its EMA teacher are complete SCRWKV networks:

```text
                                      labeled image + ground truth
                                                 |
                                                 v
RGB image --> SCRWKV student --> mask logits --> BCE + Dice + clDice
                |       |
                |       +--> 8-channel decoder features --> topology and
                |                                      morphology risk heads
                |
                +-- EMA update --> SCRWKV teacher --> unlabeled EMA mask
                                                     |          |
                                                     |          +--> prompts
                                                     |                frozen SAM
                                                     v                  |
                                             consistency loss      SAM mask + U_geo
                                                                        |
                                                                        v
                                                    confidence-weighted pseudo-label loss
```

The integration is therefore at **SemiSAM Stage 2 (semi-supervised
specialist training)**. SCRWKV processes every labeled and unlabeled image and
produces the final segmentation at inference. The frozen SAM model is queried
only during training. SAM is not required to run the trained SCRWKV model on a
test image.

The adapter can also be constructed by the Stage-1 code, but the reported
2,000-step experiment was launched directly with `--stage ssl`. Because no
`best_pretrain.pth` existed, that run initialized SCRWKV from scratch rather
than from a separate supervised pretraining stage.

## What is taken from SAM

The current implementation does **not** take feature maps, image embeddings,
prompt embeddings, attention maps, or decoder tokens from SAM. SAM remains a
frozen mask-producing teacher. The information crossing the SAM/SCRWKV
boundary is limited to:

| SAM-side quantity | How it is obtained | How SemiSAM uses it |
|---|---|---|
| Component mask `sam_comp` | A box-prompted SAM prediction | Blended with the EMA mask to form the unlabeled pseudo-label |
| Geometric uncertainty `U_geo` | Variation across jittered versions of the component box | Rejects unstable SAM corrections and weights the pseudo-label loss |
| Pseudo-label entropy | Computed from the corrected SAM/EMA probability mask | Available to the optional dual-teacher fusion path, which is disabled in the reference config |

The boxes are derived from connected components in the **SCRWKV EMA
prediction**, with at most five SAM queries per image. A correction is accepted
when mean component `U_geo < 0.3`. The consultation MLP then blends SAM and EMA
using topology risk, morphology risk, confidence, and component geometry. Thus
SAM supplies mask-level supervision; it does not supply representation-level
features to SCRWKV.

## SCRWKV multi-scale features

For a 640x640 input, the Structure-Field Encoder (SFE) returns four feature
levels:

| SFE level | Tensor shape | Main role |
|---|---:|---|
| `c4` | `B x 128 x 80 x 80` | Coarse context and long-range crack structure |
| `c3` | `B x 64 x 160 x 160` | Intermediate structural context |
| `c2` | `B x 32 x 320 x 320` | Local crack geometry |
| `c1` | `B x 16 x 640 x 640` | Fine boundaries and thin-crack detail |

CSHF projects every level to eight channels. Learned DySample operators enlarge
the lower-resolution levels, and a final interpolation handles arbitrary or
non-square input dimensions. The adapter exposes the original pyramid as
`multiscale_features` and the full-resolution projected tuple as
`aligned_multiscale_features`.

### Scale-aware attention

Scale-aware attention is the spatial scale selector inside the Cross-Scale
Harmonic Fusion (CSHF) decoder. It operates as follows:

1. A learned scale embedding is added to each aligned feature so the decoder
   can distinguish coarse and fine levels after they share the same resolution.
2. The four eight-channel tensors are concatenated into a 32-channel tensor.
3. A learned 1x1 convolution produces four logits at every pixel.
4. Softmax across the four scales produces
   `scale_attention` with shape `B x 4 x H x W`; its four weights sum to one at
   every spatial location.
5. The aligned features are combined using those pixel-specific weights. A
   projection expands this fused representation and modulates the concatenated
   multi-scale tensor by element-wise multiplication, producing the
   `harmonic_features` passed to the final decoder.

Conceptually, the network can favor coarse-scale context around fragmented or
ambiguous crack regions while assigning more weight to the fine scale along
thin boundaries. These weights are learned indirectly through the segmentation
loss; there is no hand-designed target saying which scale a pixel should use.

In the **current SemiSAM experiment**, scale attention is used internally to
form SCRWKV's mask features and is exported by the adapter for visualization or
future losses. It is not yet passed to SAM, the consultation MLP, or the
topology/morphology heads as a separate input. Those risk heads receive the
final eight-channel `decoder_features` after CSHF.

`scrwkv_risk_scale: 0.25` is unrelated to scale-aware attention. It only
downsamples the final decoder features before the computationally heavier risk
heads, then upsamples `R_topo` and `R_morph` to the input resolution.

## Layout

- `specialist_scrwkv.py` -> `SemiSAM/model/specialist_scrwkv.py`
- `factory.py` -> `SemiSAM/model/factory.py`
- `smoke_test_scrwkv.py` -> `SemiSAM/smoke_test_scrwkv.py`
- `eval_scrwkv.py` -> `SemiSAM/eval_scrwkv.py`
- `config_scrwkv.yaml` -> `SemiSAM/config_scrwkv.yaml`

The SemiSAM config parser also needs these `ModelConfig` fields:

```python
scrwkv_root: str = "/home/guest/dinu/SCRWKV/SCRWKV"
scrwkv_risk_scale: float = 0.25
```

Training and evaluation should construct specialists through
`build_specialist(...)`. SCRWKV and SegFormer use AdamW; the original
ResNet/UNet path keeps its existing optimizer.

On the reference server:

```bash
source /home/guest/dinu/ENTER/etc/profile.d/conda.sh
conda activate sam3_crack
cd /home/guest/dinu/SemiSAM
export TORCH_CUDA_ARCH_LIST=8.9 CUDA_VISIBLE_DEVICES=0
python smoke_test_scrwkv.py --config config_scrwkv.yaml
python train.py --config config_scrwkv.yaml --stage ssl \
  --snapshot snapshots/tut_5pct_scrwkv
python eval_scrwkv.py \
  --checkpoint snapshots/tut_5pct_scrwkv/best_model.pth
```

The existing SemiSAM default remains SegFormer-B1. The separate SCRWKV config
keeps the standardized 2,000-step budget and changes only the specialist.

The evaluator follows the existing TUT-test protocol: 640x640 inputs, a 0.5
probability threshold, non-empty ground-truth cases, and the project's Dice,
IoU, clDice, skeleton precision/recall, and fragmentation implementations. It
writes `test_eval.log` and `test_per_image_metrics.csv` into the output folder.


## Feature-level consistency (2026-08-19)

`topology.feature_consistency` (default 0 = off) adds a consistency term on
unlabeled data between the student and the EMA teacher, riding the same
sigmoid ramp as mask consistency (`consistency_rampup` epochs):

- MSE on `harmonic_features` (representation stability).
- KL divergence on `scale_attention` routing distributions
  (KL(teacher || student), both post-softmax).

Implemented in SemiSAM's `train.py` (`ssl_train`, guarded on
`"scale_attention" in student_out`), so the term is a no-op on ResNet-UNet
and SegFormer specialists. Early KL is near zero (untrained attention is
~uniform) and peaks mid-training once routing sharpens; the ramp withholds
the gradient while the EMA teacher still lags the student.

Reference run: `snapshots/tut_5pct_scrwkv_featcons` (2,000 steps, same seed
and budget as the SCRWKV baseline, val 0.7155 / test 0.7544).
