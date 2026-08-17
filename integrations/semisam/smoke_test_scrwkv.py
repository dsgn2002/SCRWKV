"""Repeatable SemiSAM/SCRWKV integration smoke test."""

import argparse

import torch

from losses.supervised import bce_dice_loss
from losses.topology import soft_cl_dice_loss
from model.factory import build_specialist
from model.structure_risk import StructureRiskHeads
from utils.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_scrwkv.yaml")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SCRWKV requires CUDA")
    cfg = Config.from_yaml(args.config)
    model = build_specialist(
        cfg.model.specialist_encoder,
        tuple(cfg.model.specialist_decoder_channels),
        3,
        True,
        cfg.model.scrwkv_root,
        cfg.model.scrwkv_risk_scale,
    ).cuda()
    model.add_structure_risks(StructureRiskHeads(model._final_ch).cuda())

    image = torch.randn(
        args.batch_size, 3, args.height, args.width, device="cuda"
    )
    target = (
        torch.rand(args.batch_size, 1, args.height, args.width, device="cuda")
        > 0.97
    ).float()
    output = model(image)
    expected = (args.batch_size, 1, args.height, args.width)
    assert output["mask_logits"].shape == expected
    assert output["R_topo"].shape == expected
    assert output["R_morph"].shape == expected

    bce, dice = bce_dice_loss(output["mask_logits"], target)
    cldice = soft_cl_dice_loss(torch.sigmoid(output["mask_logits"]), target)
    loss = bce + dice + cfg.topology.cl_dice_weight * cldice
    if not args.no_backward:
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    print("mask_logits", tuple(output["mask_logits"].shape))
    print("features", tuple(output["features"].shape))
    print("risk_scale", cfg.model.scrwkv_risk_scale)
    print("loss", float(loss.detach()))
    print("backward", not args.no_backward)


if __name__ == "__main__":
    main()

