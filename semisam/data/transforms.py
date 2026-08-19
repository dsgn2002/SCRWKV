"""2D image augmentations via albumentations + multi-scale utilities."""

from __future__ import annotations

import random

import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size: tuple[int, int] = (512, 512)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(img_size: tuple[int, int] = (512, 512)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)  # omnicrack30k has 331 mismatched image/mask dims


def get_weak_transform(img_size: tuple[int, int] = (512, 512)):
    """Weak augmentations for teacher branch."""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_strong_transform(img_size: tuple[int, int] = (512, 512)):
    """Strong augmentations for unlabeled student branch."""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# multi-scale utilities for scale-equivariant U_scale (Scale_equivariant.md)
# ---------------------------------------------------------------------------

def scale_crop_zoom(
    x: torch.Tensor, scale: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Random crop at 1/scale size, resize back to original → zoomed-in view.

    ponytail: random crop (not center crop) — cracks appear anywhere in UAV
    imagery; center-cropping loses edge cracks. Monte Carlo coverage over
    training iterations.

    Args:
        x: (B, C, H, W) normalized tensor.
        scale: zoom factor (>1 = zoom in).

    Returns:
        (zoomed_tensor, crop_coords_dict).
    """
    B, C, H, W = x.shape
    crop_h = int(H / scale)
    crop_w = int(W / scale)
    crop_h = max(1, crop_h)
    crop_w = max(1, crop_w)

    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    cropped = x[:, :, top:top + crop_h, left:left + crop_w]
    zoomed = F.interpolate(
        cropped, size=(H, W), mode="bilinear", align_corners=True,
    )
    return zoomed, {"top": top, "left": left, "crop_h": crop_h, "crop_w": crop_w}


def scale_pad_zoom(
    x: torch.Tensor, scale: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Pad to larger size then resize down → zoomed-out view.

    For 0.5×: pad 640×640 to 1280×1280, resize to 640×640.
    Uses constant-value padding (ImageNet mean ≈ 0 for normalized tensors).

    Args:
        x: (B, C, H, W) normalized tensor.
        scale: zoom factor (<1 = zoom out).

    Returns:
        (zoomed_tensor, orig_dims_dict).
    """
    B, C, H, W = x.shape
    pad_h = int(H * (1.0 / scale - 1.0))
    pad_w = int(W * (1.0 / scale - 1.0))

    # Pad right and bottom with constant 0 (≈ImageNet mean for normalized input)
    padded = F.pad(x, [0, pad_w, 0, pad_h], mode="constant", value=0.0)
    zoomed = F.interpolate(
        padded, size=(H, W), mode="bilinear", align_corners=True,
    )
    return zoomed, {"orig_h": H, "orig_w": W, "pad_h": pad_h, "pad_w": pad_w}


def align_prediction(
    pred: torch.Tensor,
    coords: dict,
    scale: float,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """Align a scale-variant prediction back to 1× coordinate space.

    Args:
        pred: (B, C, H, W) prediction at this scale.
        coords: metadata from scale_crop_zoom / scale_pad_zoom.
        scale: the scale factor used.
        target_shape: (H, W) of the 1× reference.

    Returns:
        (B, C, target_H, target_W) aligned prediction.
    """
    if scale == 1.0:
        if pred.shape[2:] != target_shape:
            pred = F.interpolate(
                pred, size=target_shape, mode="bilinear", align_corners=True,
            )
        return pred

    B, C = pred.shape[:2]
    th, tw = target_shape

    if scale > 1.0:
        # Zoomed-in: shrink back to crop size, paste into crop region
        top, left = coords["top"], coords["left"]
        ch, cw = coords["crop_h"], coords["crop_w"]
        shrunk = F.interpolate(
            pred, size=(ch, cw), mode="bilinear", align_corners=True,
        )
        aligned = torch.zeros(B, C, th, tw, device=pred.device, dtype=pred.dtype)
        aligned[:, :, top:top + ch, left:left + cw] = shrunk
        return aligned

    # scale < 1.0: Zoomed-out — upscale then center-crop original region
    oh, ow = coords["orig_h"], coords["orig_w"]
    ph, pw = coords["pad_h"], coords["pad_w"]
    up_h, up_w = oh + ph, ow + pw
    upscaled = F.interpolate(
        pred, size=(up_h, up_w), mode="bilinear", align_corners=True,
    )
    start_h = (up_h - th) // 2
    start_w = (up_w - tw) // 2
    return upscaled[:, :, start_h:start_h + th, start_w:start_w + tw]
