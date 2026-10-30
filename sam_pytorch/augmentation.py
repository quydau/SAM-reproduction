"""CIFAR-10 image and batch augmentations implemented with torchvision."""

from collections.abc import Callable

import torch
from torch import Tensor
from torchvision import transforms


class CIFAR10Augmentation:
    """Builds image transforms and batch Cutout for a CIFAR-10 run.

    Training pipelines (in execution order):
      - basic: horizontal flip -> padded random crop -> tensor -> normalize
      - cutout: horizontal flip -> padded random crop -> tensor -> normalize
        -> batch Cutout
      - autoaugment: AutoAugment -> horizontal flip -> padded random crop
        -> tensor -> normalize

    AutoAugment before the basic transforms matches the reference JAX
    pipeline. Cutout is a batch transform, so ``apply_cutout`` is called by
    the DataLoader collator after each image has been normalized.
    """

    MEAN = (0.49139968, 0.48215841, 0.44653091)
    STD = (0.24703223, 0.24348513, 0.26158784)
    OPTIONS = ("none", "basic", "cutout", "autoaugment")

    def __init__(self, name: str, crop_padding: int = 4,
                 cutout_length: int = 16,
                 horizontal_flip_probability: float = 0.5):
        if name not in self.OPTIONS:
            raise ValueError(f"Unknown CIFAR-10 augmentation: {name}")
        if (crop_padding < 0 or cutout_length < 1
                or not 0.0 <= horizontal_flip_probability <= 1.0):
            raise ValueError("crop_padding must be >= 0 and cutout_length >= 1")

        self.name = name
        self.cutout_length = cutout_length
        ops: list[Callable] = []

        # AutoAugment acts on the PIL image before the basic image transforms.
        if name == "autoaugment":
            ops.append(transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.CIFAR10))

        # Cutout uses the same per-image transforms as basic augmentation.
        if name in ("basic", "cutout", "autoaugment"):
            ops.extend((
                transforms.RandomHorizontalFlip(horizontal_flip_probability),
                transforms.RandomCrop(32, padding=crop_padding,
                                      padding_mode="reflect"),
            ))

        # Convert and normalize only after all per-image transforms.
        ops.extend((transforms.ToTensor(),
                    transforms.Normalize(self.MEAN, self.STD)))
        self.image_transform = transforms.Compose(ops)

    def apply_cutout(self, images: Tensor) -> Tensor:
        """After normalization, zero one shared random square in the batch."""
        if self.name != "cutout":
            return images
        if images.ndim != 4:
            raise ValueError("Cutout expects images in NCHW batch format")

        _, _, height, width = images.shape
        center_y = torch.randint(height, ()).item()
        center_x = torch.randint(width, ()).item()
        half = self.cutout_length // 2
        y0, y1 = max(0, center_y - half), min(height, center_y + half)
        x0, x1 = max(0, center_x - half), min(width, center_x + half)
        images = images.clone()
        images[:, :, y0:y1, x0:x1] = 0
        return images
