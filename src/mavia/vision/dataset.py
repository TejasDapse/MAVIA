"""MVTec AD dataset access.

Canonical layout produced by ``scripts/download_mvtec.py``::

    data/mvtec_ad/<category>/train/good/000.png
    data/mvtec_ad/<category>/test/<defect>/000.png
    data/mvtec_ad/<category>/ground_truth/<defect>/000_mask.png

Note that ``test/good`` has no corresponding mask directory: good samples carry
an all-zero mask, which this module synthesises so every test item has one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)  # fmt: skip

TEXTURE_CATEGORIES: frozenset[str] = frozenset({"carpet", "grid", "leather", "tile", "wood"})

# ImageNet statistics - the backbone is pretrained, so inputs must match its domain.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    """One MVTec AD image with its label and optional mask path."""

    image_path: Path
    defect_type: str  # "good" for defect-free
    mask_path: Path | None

    @property
    def is_anomalous(self) -> bool:
        return self.defect_type != "good"


def build_transform(image_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Resize/centre-crop/normalise, matching the standard PatchCore protocol."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mask_transform(image_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Masks use NEAREST so labels stay binary, and are never normalised."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def list_samples(root: Path, category: str, split: str) -> list[Sample]:
    """Enumerate samples for one category/split, pairing masks where they exist."""
    split_dir = root / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"No '{split}' split for category '{category}' at {split_dir}. "
            "Run `uv run python scripts/download_mvtec.py`."
        )

    samples: list[Sample] = []
    for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        for image_path in sorted(defect_dir.glob("*.png")):
            mask_path: Path | None = None
            if defect_type != "good":
                candidate = (
                    root / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                )
                mask_path = candidate if candidate.exists() else None
            samples.append(
                Sample(image_path=image_path, defect_type=defect_type, mask_path=mask_path)
            )
    return samples


class MVTecDataset(Dataset[dict[str, object]]):
    """Torch dataset over one MVTec AD category.

    Each item is a dict with ``image`` (CHW float tensor), ``mask`` (1HW float
    tensor, all zeros for good samples), ``label`` (0/1), ``defect_type`` and
    ``path``.
    """

    def __init__(
        self,
        root: Path,
        category: str,
        split: str = "train",
        image_size: int = 256,
        crop_size: int = 224,
        samples: Sequence[Sample] | None = None,
    ) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Expected one of {CATEGORIES}.")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.crop_size = crop_size
        self.samples = (
            list(samples) if samples is not None else list_samples(self.root, category, split)
        )
        self.transform = build_transform(image_size, crop_size)
        self.mask_transform = build_mask_transform(image_size, crop_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        tensor = self.transform(image)

        if sample.mask_path is not None:
            mask = self.mask_transform(Image.open(sample.mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.crop_size, self.crop_size)

        return {
            "image": tensor,
            "mask": mask,
            "label": int(sample.is_anomalous),
            "defect_type": sample.defect_type,
            "path": str(sample.image_path),
        }

    def labels(self) -> np.ndarray:
        return np.array([int(s.is_anomalous) for s in self.samples], dtype=np.int64)


def available_categories(root: Path) -> list[str]:
    """Categories actually present on disk, in canonical order."""
    return [c for c in CATEGORIES if (Path(root) / c / "train" / "good").is_dir()]
