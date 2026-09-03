"""Agent 1 - Vision Inspector.

Wraps the detection stack behind one call that returns a typed ``VisionResult``.
Everything downstream - retrieval, LLM reasoning, the approval gate, the report -
consumes that contract and never touches a tensor.

Routing logic, in order:

1. A fitted PatchCore memory bank for the category, if one exists. This is the
   accurate path and the one used in steady state.
2. CLIP zero-shot, when the category has never been fitted. Flagged in the result
   so downstream agents know to discount its confidence.

Model loading is lazy and cached: a dashboard inspecting one bottle should not
pay to load fifteen memory banks, and a batch of bottles should load one once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mavia.config import Settings, get_settings
from mavia.logging_setup import get_logger
from mavia.schemas import Verdict, VisionResult
from mavia.vision.dataset import build_transform
from mavia.vision.patchcore import PatchCore
from mavia.vision.postprocess import extract_regions, normalise_map, render_overlay

logger = get_logger(__name__)

# After normalisation the calibrated threshold sits at 0.5 by construction.
NORMALISED_THRESHOLD = 0.5


class VisionInspector:
    """Detect and localise defects in a single product image."""

    def __init__(
        self,
        settings: Settings | None = None,
        models_dir: Path | None = None,
        enable_zero_shot: bool = True,
        save_artifacts: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.models_dir = Path(models_dir or self.settings.models_dir) / "patchcore"
        self.enable_zero_shot = enable_zero_shot
        self.save_artifacts = save_artifacts

        self._models: dict[str, PatchCore] = {}
        self._zero_shot: object | None = None
        self._thresholds = self._load_thresholds()
        self._transform = build_transform()

    # ------------------------------------------------------------------ loading

    def _load_thresholds(self) -> dict[str, float]:
        """Per-category decision thresholds written by scripts/train_vision.py."""
        path = self.models_dir / "thresholds.json"
        if path.exists():
            return {k: float(v) for k, v in json.loads(path.read_text()).items()}
        logger.warning("no_thresholds_file", path=str(path))
        return {}

    def available_categories(self) -> list[str]:
        if not self.models_dir.is_dir():
            return []
        return sorted(p.stem for p in self.models_dir.glob("*.pt"))

    def _model_for(self, category: str) -> PatchCore | None:
        if category in self._models:
            return self._models[category]
        path = self.models_dir / f"{category}.pt"
        if not path.exists():
            return None
        logger.info("loading_memory_bank", category=category)
        model = PatchCore.load(path)
        self._models[category] = model
        return model

    def _zero_shot_model(self) -> object | None:
        if not self.enable_zero_shot:
            return None
        if self._zero_shot is None:
            from mavia.vision.zeroshot import ClipZeroShot

            logger.info("loading_clip_fallback")
            self._zero_shot = ClipZeroShot(
                model_name=self.settings.clip_model, pretrained=self.settings.clip_pretrained
            )
        return self._zero_shot

    # --------------------------------------------------------------- inspection

    def inspect(self, image_path: str | Path, category: str | None = None) -> VisionResult:
        """Inspect one image, returning the typed contract for the pipeline."""
        started = time.perf_counter()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"No image at {path}")

        category = category or self._infer_category(path)
        image = Image.open(path).convert("RGB")

        model = self._model_for(category)
        if model is not None:
            return self._inspect_with_patchcore(model, image, path, category, started)
        return self._inspect_with_zero_shot(image, path, category, started)

    def _inspect_with_patchcore(
        self, model: PatchCore, image: Image.Image, path: Path, category: str, started: float
    ) -> VisionResult:
        tensor = self._transform(image).unsqueeze(0)
        output = model.predict(tensor)

        raw_score = float(output.image_scores[0])
        threshold = self._thresholds.get(category)
        if threshold is None:
            logger.warning("uncalibrated_category", category=category)
            threshold = raw_score  # forces a 0.5 normalised score: explicitly undecided

        normalised_score = (
            float(np.clip(0.5 * raw_score / threshold, 0.0, 1.0)) if threshold > 0 else 0.0
        )
        anomaly_map = normalise_map(output.anomaly_maps[0], threshold)
        regions = (
            extract_regions(anomaly_map, NORMALISED_THRESHOLD)
            if normalised_score >= NORMALISED_THRESHOLD
            else []
        )

        map_path, overlay_path = self._persist(path, category, anomaly_map, image, regions)

        return VisionResult(
            category=category,
            verdict=Verdict.FAIL if normalised_score >= NORMALISED_THRESHOLD else Verdict.PASS,
            anomaly_score=normalised_score,
            decision_threshold=NORMALISED_THRESHOLD,
            regions=regions,
            anomaly_map_path=map_path,
            overlay_path=overlay_path,
            model_name=f"PatchCore/WideResNet50-2/{category}",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _inspect_with_zero_shot(
        self, image: Image.Image, path: Path, category: str, started: float
    ) -> VisionResult:
        model = self._zero_shot_model()
        if model is None:
            raise RuntimeError(
                f"No memory bank for '{category}' and the CLIP fallback is disabled. "
                f"Fit one with: uv run python scripts/train_vision.py -c {category}"
            )

        result = model.classify(image, category)  # type: ignore[attr-defined]
        logger.info("zero_shot_fallback", category=category, score=result.anomaly_probability)

        return VisionResult(
            category=category,
            verdict=Verdict.FAIL if result.is_anomalous else Verdict.PASS,
            anomaly_score=result.anomaly_probability,
            decision_threshold=NORMALISED_THRESHOLD,
            regions=[],  # CLIP scores the whole image; it cannot localise
            zero_shot_label=result.defect_label,
            zero_shot_confidence=result.defect_confidence,
            model_name=result.model_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ----------------------------------------------------------------- helpers

    def _infer_category(self, path: Path) -> str:
        """Recover the category from an MVTec-style path, else fall back to a generic noun."""
        from mavia.vision.dataset import CATEGORIES

        for part in path.parts[::-1]:
            if part in CATEGORIES:
                return part
        known = set(self.available_categories())
        for part in path.parts[::-1]:
            if part in known:
                return part
        logger.warning("category_not_inferred", path=str(path))
        return "product"

    def _persist(
        self,
        image_path: Path,
        category: str,
        anomaly_map: np.ndarray,
        image: Image.Image,
        regions: list,
    ) -> tuple[str | None, str | None]:
        """Write the heatmap and overlay a reviewer will look at before approving."""
        if not self.save_artifacts:
            return None, None

        out_dir = Path(self.settings.artifacts_dir) / "inspections" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        map_path = out_dir / f"{stem}_map.png"
        Image.fromarray((anomaly_map * 255).astype(np.uint8)).save(map_path)

        overlay_path = out_dir / f"{stem}_overlay.png"
        render_overlay(image, anomaly_map, regions=regions).save(overlay_path)

        return str(map_path), str(overlay_path)


@torch.no_grad()
def inspect_image(image_path: str | Path, category: str | None = None) -> VisionResult:
    """One-shot convenience wrapper; prefer reusing a VisionInspector for batches."""
    return VisionInspector().inspect(image_path, category)
