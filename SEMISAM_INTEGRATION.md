# SemiSAM integration

This branch adapts the authors' SCRWKV implementation for use as the specialist
inside SemiSAM while retaining the original tensor-only forward interface.

## Model interfaces

The original supervised interface is unchanged:

```python
logits = model(images)
```

The SemiSAM adapter exposes the final decoder representation, four-scale SFE
pyramid, aligned CSHF inputs, harmonic feature, and scale-attention map:

```python
from models import SemiSAMSCRWKV

model = SemiSAMSCRWKV().cuda()
output = model(images)

logits = output["mask_logits"]
features = output["features"]
pyramid = output["multiscale_features"]
scale_attention = output["scale_attention"]
```

For a 640 x 640 input the feature shapes are:

```text
mask_logits:      B x 1   x 640 x 640
decoder features: B x 8   x 640 x 640
SFE pyramid:      B x 128 x  80 x  80
                  B x 64  x 160 x 160
                  B x 32  x 320 x 320
                  B x 16  x 640 x 640
scale attention:  B x 4   x 640 x 640
```

`add_structure_risks()` and `add_consultation_predictor()` match the extension
hooks used by the current SemiSAM specialists. SAM masks remain training-time
targets and are not required as model inputs at inference.

## Compatibility changes

- SFE pyramid sizes are derived from the input dimensions rather than fixed at
  the paper's 512 x 512 resolution.
- Rectangular token maps use the supplied patch height and width rather than
  reconstructing a square with `sqrt(token_count)`.
- The model path uses standalone PyTorch modules and does not require MMCV.
- The CUDA extension is compiled for the visible GPU architecture instead of
  hard-coding compute capability 8.6.
- Dy-WKV preserves the input CUDA device rather than forcing tensors to GPU 0.

## Smoke tests

From the repository root in a CUDA-enabled environment:

```bash
python -m scripts.smoke_test_semisam --height 128 --width 160
python -m scripts.smoke_test_semisam --height 640 --width 640 --no-backward
```

The first command verifies a rectangular forward and backward pass. The second
verifies the production SemiSAM resolution without retaining backward graphs.

