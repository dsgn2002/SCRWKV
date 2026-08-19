"""2D crack dataset + TwoStreamBatchSampler.

Combined dataset: labeled images first (with masks), then unlabeled (dummy masks).
Handles TUT format: JPG images, PNG masks, matched by stem.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CrackDataset(Dataset):
    """Combined labeled + unlabeled crack image dataset.

    Each item: {'image': (C, H, W), 'label': (1, H, W)}.
    Unlabeled images get a zero mask.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str | None = None,
        transform: Callable | None = None,
        unlabeled_dir: str | None = None,
    ):
        self.transform = transform
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None

        self.labeled_images = sorted(
            list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
        )

        self.unlabeled_images: list[Path] = []
        if unlabeled_dir:
            ud = Path(unlabeled_dir)
            self.unlabeled_images = sorted(
                list(ud.glob("*.png")) + list(ud.glob("*.jpg")) + list(ud.glob("*.jpeg"))
            )

        self.all_images = self.labeled_images + self.unlabeled_images
        self.num_labeled = len(self.labeled_images)

        if len(self.all_images) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.all_images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.all_images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        has_label = idx < self.num_labeled and self.mask_dir is not None
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if has_label:
            # ponytail: TUT has .jpg images but .png masks — match by stem, try both ext
            stem = img_path.stem
            for ext in (".png", ".jpg", ".jpeg"):
                p = self.mask_dir / f"{stem}{ext}"
                if p.exists():
                    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 127).astype(np.uint8)
                    break

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image_t = transformed["image"]
            mask_t = transformed["mask"].unsqueeze(0).float()
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return {"image": image_t, "label": mask_t}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices (labeled + unlabeled).

    Ported verbatim from SemiSAM+ dataloaders/dataset.py.
    """

    def __init__(
        self, primary_indices, secondary_indices, batch_size, secondary_batch_size,
    ):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = _iterate_once(self.primary_indices)
        secondary_iter = _iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                _grouper(primary_iter, self.primary_batch_size),
                _grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def _iterate_once(iterable):
    return np.random.permutation(iterable)


def _iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def _grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
