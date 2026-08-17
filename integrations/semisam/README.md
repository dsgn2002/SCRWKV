# SemiSAM integration

These files record the integration used by the SemiSAM workspace on the
`semisam-integration` branch. SCRWKV itself exposes the architecture-neutral
interface in `models/semisam_adapter.py`; the files here are installed on the
SemiSAM side.

## Layout

- `specialist_scrwkv.py` -> `SemiSAM/model/specialist_scrwkv.py`
- `factory.py` -> `SemiSAM/model/factory.py`
- `smoke_test_scrwkv.py` -> `SemiSAM/smoke_test_scrwkv.py`
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
```

The existing SemiSAM default remains SegFormer-B1. The separate SCRWKV config
keeps the standardized 2,000-step budget and changes only the specialist.

