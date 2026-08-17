"""GPU smoke test for the SemiSAM-compatible SCRWKV interface."""

import argparse

import torch

from models import SemiSAMSCRWKV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SCRWKV smoke test requires CUDA")

    device = torch.device("cuda")
    model = SemiSAMSCRWKV().to(device)
    image = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    output = model(image)

    expected_size = (args.height, args.width)
    assert output["mask_logits"].shape == (
        args.batch_size, 1, *expected_size
    )
    assert output["features"].shape[-2:] == expected_size
    assert len(output["multiscale_features"]) == 4
    assert output["scale_attention"].shape == (
        args.batch_size, 4, *expected_size
    )

    if not args.no_backward:
        output["mask_logits"].mean().backward()
        assert any(
            parameter.grad is not None for parameter in model.parameters()
            if parameter.requires_grad
        )

    print("mask_logits", tuple(output["mask_logits"].shape))
    print("decoder_features", tuple(output["features"].shape))
    print(
        "pyramid",
        [tuple(feature.shape) for feature in output["multiscale_features"]],
    )
    print("scale_attention", tuple(output["scale_attention"].shape))
    print("backward", not args.no_backward)


if __name__ == "__main__":
    main()

