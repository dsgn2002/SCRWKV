from .decoder import build
from .semisam_adapter import SemiSAMSCRWKV, build_semisam_scrwkv

def build_model(args):
    return build(args)


__all__ = [
    "build_model",
    "SemiSAMSCRWKV",
    "build_semisam_scrwkv",
]
