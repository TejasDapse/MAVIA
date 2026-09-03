"""CLIP zero-shot fallback for categories with no memory bank.

PatchCore needs defect-free examples of the exact product. On the day a new SKU
reaches the line there are none, and an inspection system that answers "unknown
category, cannot help" during precisely the ramp-up when defects are most likely
is not much of a system.

CLIP fills that gap by comparing the image against natural-language descriptions
of a good and a flawed part. It is materially weaker than a fitted memory bank -
this is a stopgap that keeps the line covered until enough good samples exist,
and every result it produces is flagged so the analyst agent discounts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

# Ensembling several phrasings is standard practice for zero-shot CLIP: any one
# prompt is a lottery ticket on wording, while the mean embedding is stable.
NORMAL_TEMPLATES = (
    "a photo of a flawless {category}",
    "a photo of a normal {category} without defects",
    "a close-up photo of an undamaged {category}",
    "a product photo of a perfect {category}",
)

ANOMALOUS_TEMPLATES = (
    "a photo of a damaged {category}",
    "a photo of a {category} with a defect",
    "a close-up photo of a {category} with a visible flaw",
    "a product photo of a broken or contaminated {category}",
)

DEFECT_VOCABULARY = (
    "scratch", "crack", "contamination", "hole", "dent",
    "discoloration", "misalignment", "missing part", "deformation",
)  # fmt: skip


@dataclass(frozen=True)
class ZeroShotResult:
    is_anomalous: bool
    anomaly_probability: float
    defect_label: str | None
    defect_confidence: float | None
    model_name: str


class ClipZeroShot:
    """Language-prompted anomaly screening for unseen product categories."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: torch.device | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or _default_device())
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def _encode_text(self, prompts: tuple[str, ...]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    @lru_cache(maxsize=64)  # noqa: B019 - bounded by category count, lives with the model
    def _category_embeddings(self, category: str) -> tuple[torch.Tensor, torch.Tensor]:
        normal = self._encode_text(tuple(t.format(category=category) for t in NORMAL_TEMPLATES))
        anomalous = self._encode_text(
            tuple(t.format(category=category) for t in ANOMALOUS_TEMPLATES)
        )
        return normal.mean(dim=0, keepdim=True), anomalous.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def classify(self, image: Image.Image, category: str) -> ZeroShotResult:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        normal, anomalous = self._category_embeddings(category)
        prototypes = torch.cat([normal, anomalous], dim=0)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ prototypes.T
        probabilities = logits.softmax(dim=-1).squeeze(0)
        anomaly_probability = float(probabilities[1])

        defect_label: str | None = None
        defect_confidence: float | None = None
        if anomaly_probability > 0.5:
            defect_label, defect_confidence = self._name_defect(image_features, category)

        return ZeroShotResult(
            is_anomalous=anomaly_probability > 0.5,
            anomaly_probability=anomaly_probability,
            defect_label=defect_label,
            defect_confidence=defect_confidence,
            model_name=f"CLIP/{self.model_name}",
        )

    @torch.no_grad()
    def _name_defect(self, image_features: torch.Tensor, category: str) -> tuple[str, float]:
        """Attach the most plausible defect word, to seed the retrieval query."""
        prompts = tuple(f"a photo of a {category} with a {d}" for d in DEFECT_VOCABULARY)
        text_features = self._encode_text(prompts)
        probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0)
        best = int(torch.argmax(probabilities).item())
        return DEFECT_VOCABULARY[best], float(probabilities[best])

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image], category: str) -> np.ndarray:
        """Anomaly probabilities for a batch, for evaluating the fallback itself."""
        return np.array([self.classify(image, category).anomaly_probability for image in images])


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
