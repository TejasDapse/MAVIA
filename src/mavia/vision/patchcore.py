"""PatchCore: memory-bank anomaly detection.

Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022).

The method in one paragraph: embed every patch of every *defect-free* training
image with a frozen ImageNet backbone, keep a coverage-maximising subset of those
embeddings as a memory bank, and score a test patch by its distance to the
nearest bank entry. Nothing is trained. That is the property that makes it the
right fit for manufacturing — a new product line needs only good samples, and
there is no gradient step to tune, no defect labels to collect, and no risk of
the model quietly overfitting the handful of defects you happened to have.

Implemented directly rather than via anomalib so that every step - neighbourhood
aggregation, coreset selection, the neighbour reweighting in the image score - is
explicit and testable, and so the project carries no heavyweight training
framework it does not otherwise use.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from mavia.vision.coreset import greedy_coreset_indices
from mavia.vision.features import PatchFeatureExtractor


@dataclass(frozen=True)
class PatchCoreConfig:
    layers: tuple[str, ...] = ("layer2", "layer3")
    patch_size: int = 3
    sampling_ratio: float = 0.01
    projection_dim: int | None = 128
    num_neighbours: int = 9
    image_size: int = 256
    crop_size: int = 224
    blur_sigma: float = 4.0
    seed: int = 0


@dataclass(frozen=True)
class PatchCoreOutput:
    """Per-image scores and anomaly maps for a batch."""

    image_scores: np.ndarray  # (B,)
    anomaly_maps: np.ndarray  # (B, H, W) at crop resolution


def _gaussian_kernel1d(sigma: float, device: torch.device) -> Tensor:
    radius = max(1, round(4.0 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(maps: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur over ``(B, 1, H, W)``.

    PatchCore smooths the upsampled distance field because a single anomalous
    patch produces a hard-edged square at the feature grid's resolution; blurring
    turns that into the contiguous region a human would draw.
    """
    if sigma <= 0:
        return maps
    kernel = _gaussian_kernel1d(sigma, maps.device)
    radius = (kernel.numel() - 1) // 2
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    blurred = F.conv2d(F.pad(maps, (radius, radius, 0, 0), mode="reflect"), horizontal)
    return F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="reflect"), vertical)


class PatchCore:
    """Memory-bank anomaly detector for a single product category."""

    def __init__(
        self,
        config: PatchCoreConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or PatchCoreConfig()
        self.device = torch.device(device or _default_device())
        self.extractor = PatchFeatureExtractor(
            layers=self.config.layers, patch_size=self.config.patch_size, device=self.device
        )
        self.memory_bank: Tensor | None = None
        self.grid_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------ fitting

    @torch.no_grad()
    def fit(self, loader: DataLoader, verbose: bool = True) -> PatchCore:
        """Build the memory bank from defect-free images."""
        chunks: list[Tensor] = []
        grid: tuple[int, int] | None = None

        for batch_index, batch in enumerate(loader):
            images = _batch_images(batch)
            flat, grid = self.extractor.extract_flat(images)
            chunks.append(flat.cpu())
            if verbose and batch_index % 10 == 0:
                print(f"  embedded batch {batch_index + 1}/{len(loader)}", flush=True)

        if not chunks:
            raise ValueError("Cannot fit PatchCore on an empty loader.")

        embeddings = torch.cat(chunks, dim=0)
        if verbose:
            print(f"  {embeddings.shape[0]:,} patch embeddings of dim {embeddings.shape[1]}")

        indices = greedy_coreset_indices(
            embeddings,
            sampling_ratio=self.config.sampling_ratio,
            projection_dim=self.config.projection_dim,
            seed=self.config.seed,
        )
        self.memory_bank = embeddings[indices].to(self.device)
        self.grid_shape = grid
        if verbose:
            print(
                f"  memory bank: {self.memory_bank.shape[0]:,} entries "
                f"({100 * self.config.sampling_ratio:.1f}% coreset)"
            )
        return self

    # ----------------------------------------------------------------- scoring

    @torch.no_grad()
    def _patch_distances(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Nearest-neighbour distance and index for each patch embedding."""
        if self.memory_bank is None:
            raise RuntimeError("PatchCore is not fitted; call fit() or load() first.")
        distances = torch.cdist(embeddings, self.memory_bank)
        nn_distance, nn_index = distances.min(dim=1)
        return nn_distance, nn_index

    @torch.no_grad()
    def _image_score(
        self, patch_distances: Tensor, patch_indices: Tensor, patches: Tensor
    ) -> Tensor:
        """PatchCore's neighbour-reweighted image score.

        The naive score is simply the largest patch distance. PatchCore refines
        it: if the memory-bank entry closest to the most anomalous patch is
        itself isolated (far from its own neighbours in the bank), that entry
        represents a rare-but-normal appearance, and a large distance to it is
        less alarming. The weight discounts the score in exactly that case,
        which is what stops rare legitimate features from firing as defects.
        """
        if self.memory_bank is None:
            raise RuntimeError("PatchCore is not fitted.")

        max_position = int(torch.argmax(patch_distances).item())
        max_distance = patch_distances[max_position]

        num_neighbours = min(self.config.num_neighbours, self.memory_bank.shape[0])
        if num_neighbours <= 1:
            return max_distance

        nearest_entry = self.memory_bank[patch_indices[max_position]].unsqueeze(0)
        bank_distances = torch.cdist(nearest_entry, self.memory_bank).squeeze(0)
        neighbour_idx = torch.topk(bank_distances, k=num_neighbours, largest=False).indices

        query = patches[max_position].unsqueeze(0)
        query_to_neighbours = torch.cdist(query, self.memory_bank[neighbour_idx]).squeeze(0)

        # Softmax-style weight; subtracting the max keeps the exponentials stable.
        shifted = query_to_neighbours - query_to_neighbours.max()
        weight = 1.0 - torch.exp(shifted[0]) / torch.exp(shifted).sum()
        return weight * max_distance

    @torch.no_grad()
    def predict(self, images: Tensor) -> PatchCoreOutput:
        """Score a batch of preprocessed images."""
        if self.memory_bank is None:
            raise RuntimeError("PatchCore is not fitted; call fit() or load() first.")

        images = images.to(self.device)
        batch_size = images.shape[0]
        embedding = self.extractor.forward(images)  # (B, H, W, C)
        height, width, channels = embedding.shape[1:]

        scores: list[Tensor] = []
        maps: list[Tensor] = []
        for i in range(batch_size):
            patches = embedding[i].reshape(-1, channels)
            distances, indices = self._patch_distances(patches)
            scores.append(self._image_score(distances, indices, patches))
            maps.append(distances.reshape(height, width))

        anomaly_maps = torch.stack(maps).unsqueeze(1)  # (B, 1, H, W)
        anomaly_maps = F.interpolate(
            anomaly_maps,
            size=(self.config.crop_size, self.config.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        anomaly_maps = gaussian_blur(anomaly_maps, self.config.blur_sigma).squeeze(1)

        return PatchCoreOutput(
            image_scores=torch.stack(scores).cpu().numpy(),
            anomaly_maps=anomaly_maps.cpu().numpy(),
        )

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader, verbose: bool = False) -> PatchCoreOutput:
        """Score every image in a loader, preserving order."""
        all_scores: list[np.ndarray] = []
        all_maps: list[np.ndarray] = []
        for batch_index, batch in enumerate(loader):
            output = self.predict(_batch_images(batch))
            all_scores.append(output.image_scores)
            all_maps.append(output.anomaly_maps)
            if verbose and batch_index % 10 == 0:
                print(f"  scored batch {batch_index + 1}/{len(loader)}", flush=True)
        return PatchCoreOutput(
            image_scores=np.concatenate(all_scores),
            anomaly_maps=np.concatenate(all_maps),
        )

    # -------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        """Persist the memory bank and config. The backbone is not saved - it is
        pretrained and reconstructed on load, so artefacts stay small."""
        if self.memory_bank is None:
            raise RuntimeError("Nothing to save; PatchCore is not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "config": asdict(self.config),
                "grid_shape": self.grid_shape,
            },
            path,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "memory_bank_size": int(self.memory_bank.shape[0]),
                    "embedding_dim": int(self.memory_bank.shape[1]),
                    "grid_shape": list(self.grid_shape) if self.grid_shape else None,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | str | None = None) -> PatchCore:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = PatchCoreConfig(
            **{**payload["config"], "layers": tuple(payload["config"]["layers"])}
        )
        model = cls(config=config, device=device)
        model.memory_bank = payload["memory_bank"].to(model.device)
        grid = payload.get("grid_shape")
        model.grid_shape = tuple(grid) if grid else None
        return model


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batch_images(batch: object) -> Tensor:
    """Accept either a dict batch from MVTecDataset or a bare tensor."""
    if isinstance(batch, dict):
        return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, Tensor):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")
