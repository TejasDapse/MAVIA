"""Tests for the defect-history corpus and the retrieval agent.

The corpus tests guard the two properties the retrieval evaluation depends on:
query/document text symmetry, and the leakage-free mask split. Both were bugs
found during Phase 3, and both silently inflate retrieval scores when broken -
exactly the kind of failure a test must catch, because the metric still looks
fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from mavia.memory.corpus import (
    DefectCase,
    MorphologyStats,
    _mask_statistics,
    describe_observation,
    generate_cases_for_defect,
    split_morphology,
)
from mavia.memory.knowledge import KNOWLEDGE_BASE, get_knowledge, known_categories
from mavia.schemas import BoundingBox, Verdict, VisionResult
from mavia.vision.dataset import CATEGORIES

# ------------------------------------------------------------------ knowledge


def test_knowledge_base_covers_all_mvtec_categories() -> None:
    assert set(known_categories()) == set(CATEGORIES)


def test_every_entry_is_complete_and_ordered() -> None:
    for (category, defect), knowledge in KNOWLEDGE_BASE.items():
        assert knowledge.root_causes, f"{category}/{defect} has no root causes"
        assert knowledge.actions, f"{category}/{defect} has no actions"
        assert len(knowledge.actions) == len(knowledge.root_causes), (
            f"{category}/{defect}: each cause needs a matching action"
        )
        assert knowledge.process_step
        assert knowledge.description


def test_lookup_returns_none_for_unknown_mode() -> None:
    assert get_knowledge("bottle", "not_a_real_defect") is None
    assert get_knowledge("bottle", "contamination") is not None


# ---------------------------------------------------------------- observation


def test_observation_omits_defect_type_and_root_cause() -> None:
    """Retrieval must be scored on what the vision agent can actually see."""
    text = describe_observation("bottle", area_fraction=0.11, region_count=1, elongation=1.3)
    lowered = text.lower()
    assert "bottle" in lowered
    for leaked in ("broken", "contamination", "crack", "mould", "thermal"):
        assert leaked not in lowered, f"'{leaked}' leaked into the embedded observation"


def test_observation_reflects_extent_and_shape() -> None:
    small = describe_observation("tile", 0.001, 1, 1.0)
    large = describe_observation("tile", 0.20, 1, 1.0)
    assert "minimal" in small and "extensive" in large

    linear = describe_observation("carpet", 0.01, 1, 4.0)
    compact = describe_observation("carpet", 0.01, 1, 1.1)
    assert "elongated" in linear and "compact" in compact


def test_region_count_is_described() -> None:
    assert "a single localised region" in describe_observation("pill", 0.01, 1)
    assert "two separate regions" in describe_observation("pill", 0.01, 2)
    assert "scattered" in describe_observation("pill", 0.01, 7)


# ----------------------------------------------------------- mask statistics


def test_mask_statistics_measure_geometry() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:20] = 255  # 200 px, 2:1 elongation
    area, count, elongation = _mask_statistics(mask)
    assert area == pytest.approx(0.02)
    assert count == 1
    assert elongation == pytest.approx(2.0)


def test_mask_statistics_count_separate_regions() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[60:70, 60:70] = 255
    _, count, _ = _mask_statistics(mask)
    assert count == 2


def test_empty_mask_has_no_regions() -> None:
    area, count, elongation = _mask_statistics(np.zeros((32, 32), dtype=np.uint8))
    assert area == 0.0
    assert count == 0
    assert elongation == 1.0


# -------------------------------------------------------------------- split


def _stats(n: int = 10) -> MorphologyStats:
    samples = tuple((0.01 * (i + 1), 1, 1.5) for i in range(n))
    return MorphologyStats("bottle", "crack", n, 0.05, 0.02, 1.0, 1.5, samples)


def test_split_produces_disjoint_sample_pools() -> None:
    """Overlapping pools let a query match a duplicate of itself."""
    index_stats, query_stats = split_morphology(_stats(10), holdout_fraction=0.3)
    assert set(index_stats.samples).isdisjoint(set(query_stats.samples))
    assert len(index_stats.samples) + len(query_stats.samples) == 10


def test_split_always_yields_a_query_sample() -> None:
    index_stats, query_stats = split_morphology(_stats(2), holdout_fraction=0.3)
    assert len(query_stats.samples) >= 1
    assert len(index_stats.samples) >= 1


# ------------------------------------------------------------ case generation


def test_generated_cases_bootstrap_real_measurements() -> None:
    """Cases must derive from measured masks, not from a fitted distribution."""
    import random

    stats = _stats(6)
    knowledge = KNOWLEDGE_BASE[("bottle", "contamination")]
    cases = generate_cases_for_defect(
        "bottle", "contamination", knowledge, stats, n_cases=40, rng=random.Random(0)
    )

    real_areas = [s[0] for s in stats.samples]
    for case in cases:
        # Jitter is +-8%, so every case must sit close to some real measurement.
        assert any(abs(case.area_fraction - area) / area < 0.12 for area in real_areas)


def test_generated_cases_are_complete_and_consistent() -> None:
    import random

    knowledge = KNOWLEDGE_BASE[("pill", "crack")]
    cases = generate_cases_for_defect(
        "pill", "crack", knowledge, _stats(5), n_cases=10, rng=random.Random(1)
    )
    assert len(cases) == 10
    for case in cases:
        assert isinstance(case, DefectCase)
        assert case.category == "pill"
        assert case.defect_type == "crack"
        assert case.root_cause in knowledge.root_causes
        assert case.action_taken in knowledge.actions
        assert case.area_fraction > 0
        assert case.region_count >= 1


def test_cause_sampling_is_pareto_shaped() -> None:
    """Real failure data is not uniform; the most likely cause must dominate."""
    import random

    knowledge = KNOWLEDGE_BASE[("bottle", "contamination")]
    cases = generate_cases_for_defect(
        "bottle", "contamination", knowledge, _stats(8), n_cases=400, rng=random.Random(3)
    )
    counts = [sum(c.root_cause == cause for c in cases) for cause in knowledge.root_causes]
    assert counts[0] > counts[1] > 0, f"expected a decreasing cause distribution, got {counts}"


# --------------------------------------------------------------- query build


def _vision(category: str, boxes: list[BoundingBox]) -> VisionResult:
    return VisionResult(
        category=category,
        verdict=Verdict.FAIL,
        anomaly_score=0.8,
        decision_threshold=0.5,
        regions=boxes,
        model_name="test",
        latency_ms=1.0,
    )


def test_query_geometry_matches_the_regions() -> None:
    from mavia.memory.retrieval import HistoryRetriever

    box = BoundingBox(x=0, y=0, width=40, height=20, area_px=800, peak_score=0.9)
    area, count, elongation = HistoryRetriever.extract_geometry(_vision("bottle", [box]))

    assert area == pytest.approx(800 / (224 * 224))
    assert count == 1
    assert elongation == pytest.approx(2.0)


def test_query_text_uses_the_same_renderer_as_the_corpus() -> None:
    """If query and document text diverge, retrieval degrades for no real reason."""
    from mavia.memory.retrieval import HistoryRetriever

    box = BoundingBox(x=0, y=0, width=40, height=20, area_px=800, peak_score=0.9)
    vision = _vision("carpet", [box])
    area, count, elongation = HistoryRetriever.extract_geometry(vision)

    assert HistoryRetriever.build_query(vision) == describe_observation(
        "carpet", area, count, elongation
    )


def test_clean_part_yields_a_zero_area_query() -> None:
    from mavia.memory.retrieval import HistoryRetriever

    area, count, _ = HistoryRetriever.extract_geometry(_vision("tile", []))
    assert area == 0.0
    assert count == 0


# ------------------------------------------------------- geometry similarity


def test_geometry_similarity_is_highest_for_identical_geometry() -> None:
    from mavia.memory.store import DefectMemory

    identical = DefectMemory._geometry_similarity((0.05, 2, 1.5), (0.05, 2, 1.5))
    different = DefectMemory._geometry_similarity((0.05, 2, 1.5), (0.001, 5, 4.0))
    assert identical == pytest.approx(1.0)
    assert different < identical


def test_geometry_similarity_separates_magnitudes_embeddings_conflate() -> None:
    """A speck and a shattered part must not look alike, whatever the text says."""
    from mavia.memory.store import DefectMemory

    speck_to_speck = DefectMemory._geometry_similarity((0.003, 1, 1.2), (0.004, 1, 1.3))
    speck_to_large = DefectMemory._geometry_similarity((0.003, 1, 1.2), (0.117, 1, 1.3))
    assert speck_to_speck > speck_to_large * 2
