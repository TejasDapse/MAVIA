"""Agent 2 - History Retriever.

Turns a ``VisionResult`` into a natural-language observation, searches the
defect-history corpus for comparable past cases, and returns them as a typed
``RetrievalResult``.

The query is built by the *same* ``describe_observation`` function that produced
the stored records, from the geometry the vision agent actually measured. Nothing
the vision agent does not know is smuggled into the query.

Degradation is deliberate: an empty or unreachable store yields an empty result
with a recorded latency, never an exception. A missing history should cost the
analyst its supporting context, not cost the line its inspection.
"""

from __future__ import annotations

import time
from datetime import datetime

from mavia.config import Settings, get_settings
from mavia.logging_setup import get_logger
from mavia.memory.corpus import describe_observation
from mavia.memory.store import DefectMemory
from mavia.schemas import HistoricalCase, RetrievalResult, VisionResult

logger = get_logger(__name__)


class HistoryRetriever:
    """Retrieve comparable historical defect cases for a detection."""

    def __init__(
        self,
        memory: DefectMemory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.memory = memory or DefectMemory(settings=self.settings)

    # ------------------------------------------------------------ query text

    @staticmethod
    def extract_geometry(vision: VisionResult, crop_size: int = 224) -> tuple[float, int, float]:
        """Area fraction, region count and elongation, as the corpus records them."""
        total_area = float(crop_size * crop_size)
        defect_area = sum(box.area_px for box in vision.regions)
        area_fraction = defect_area / total_area if total_area else 0.0

        elongation = 1.0
        if vision.regions:
            box = vision.regions[0]
            long_side = max(box.width, box.height)
            short_side = max(1, min(box.width, box.height))
            elongation = long_side / short_side

        return area_fraction, len(vision.regions), elongation

    @classmethod
    def build_query(cls, vision: VisionResult, crop_size: int = 224) -> str:
        """Render the detection as the same kind of sentence the corpus stores."""
        area_fraction, region_count, elongation = cls.extract_geometry(vision, crop_size)
        return describe_observation(
            category=vision.category,
            area_fraction=area_fraction,
            region_count=region_count,
            elongation=elongation if vision.regions else None,
        )

    # -------------------------------------------------------------- retrieve

    def retrieve(
        self,
        vision: VisionResult,
        top_k: int | None = None,
        restrict_to_category: bool = True,
        hybrid: bool = True,
        alpha: float = 0.7,
    ) -> RetrievalResult:
        started = time.perf_counter()
        top_k = top_k or self.settings.retrieval_top_k
        query_text = self.build_query(vision)
        category = vision.category if restrict_to_category else None

        cases: list[HistoricalCase] = []
        try:
            if hybrid:
                hits = self.memory.search_hybrid(
                    query_text,
                    query_geometry=self.extract_geometry(vision),
                    top_k=top_k,
                    category=category,
                    alpha=alpha,
                )
            else:
                hits = self.memory.search(query_text, top_k=top_k, category=category)
            cases = [self._to_case(payload, score) for payload, score in hits]
        except Exception as error:
            logger.warning("retrieval_failed", error=str(error), category=vision.category)

        return RetrievalResult(
            query_text=query_text,
            cases=cases,
            top_k=top_k,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            embedding_model=self.memory.embedding_model_name,
        )

    @staticmethod
    def _to_case(payload: dict[str, object], score: float) -> HistoricalCase:
        occurred = payload.get("occurred_at")
        if isinstance(occurred, str):
            occurred_at = datetime.fromisoformat(occurred)
        elif isinstance(occurred, datetime):
            occurred_at = occurred
        else:
            occurred_at = datetime.fromtimestamp(0)

        return HistoricalCase(
            case_id=str(payload.get("case_id", "unknown")),
            category=str(payload.get("category", "unknown")),
            defect_type=str(payload.get("defect_type", "unknown")),
            root_cause=str(payload.get("root_cause", "")),
            action_taken=str(payload.get("action_taken", "")),
            outcome=str(payload.get("outcome")) if payload.get("outcome") else None,
            occurred_at=occurred_at,
            # Cosine similarity is in [-1, 1]; clamp so the contract's [0, 1] holds.
            similarity=max(0.0, min(1.0, score)),
        )
