"""Patch-level feature extraction for PatchCore.

PatchCore deliberately uses *mid-level* features (WideResNet-50 ``layer2`` and
``layer3``) rather than the deepest ones. The reasoning matters and is worth
stating, because it is the first thing an interviewer will probe:

* The final layers of an ImageNet backbone are heavily biased toward the
  classification task and discard the local texture detail that distinguishes a
  scratch from a highlight.
* The earliest layers are too generic and too high-resolution to be
  discriminative or affordable.

Each spatial position is additionally aggregated over its 3x3 neighbourhood
(average pooling, stride 1) so that a patch descriptor carries local context
rather than a single pixel's activation. This gives tolerance to small spatial
misalignment between the query and the memory bank without blurring the map.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

SUPPORTED_LAYERS = ("layer1", "layer2", "layer3", "layer4")


class PatchFeatureExtractor(nn.Module):
    """Frozen backbone that returns concatenated, neighbourhood-aggregated patches.

    Output shape is ``(B, H, W, C)`` where ``H, W`` is the spatial grid of the
    *first* requested layer and ``C`` is the summed channel count of all
    requested layers.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("layer2", "layer3"),
        patch_size: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        unknown = set(layers) - set(SUPPORTED_LAYERS)
        if unknown:
            raise ValueError(f"Unsupported layers {sorted(unknown)}; expected {SUPPORTED_LAYERS}.")

        self.layers = tuple(layers)
        self.patch_size = patch_size
        self.device = torch.device(device)

        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        self.backbone = backbone.to(self.device)

        self._captured: dict[str, Tensor] = {}
        for name in self.layers:
            getattr(self.backbone, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, _inputs: object, output: Tensor) -> None:
            self._captured[name] = output

        return hook

    @property
    def embedding_dim(self) -> int:
        """Channel count of the concatenated descriptor."""
        channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        return sum(channels[name] for name in self.layers)

    def _aggregate(self, feature: Tensor) -> Tensor:
        """Average each position over its local neighbourhood, preserving resolution."""
        padding = self.patch_size // 2
        return F.avg_pool2d(feature, kernel_size=self.patch_size, stride=1, padding=padding)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        self._captured.clear()
        self.backbone(images.to(self.device))

        features = [self._aggregate(self._captured[name]) for name in self.layers]

        # Deeper layers are lower-resolution; bring everything to the first layer's grid.
        target_size = features[0].shape[-2:]
        features = [
            f
            if f.shape[-2:] == target_size
            else F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in features
        ]

        embedding = torch.cat(features, dim=1)  # (B, C, H, W)
        return embedding.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

    @torch.no_grad()
    def extract_flat(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Return ``(B*H*W, C)`` patch embeddings plus the ``(H, W)`` grid shape."""
        embedding = self.forward(images)
        _, height, width, channels = embedding.shape
        return embedding.reshape(-1, channels), (height, width)
