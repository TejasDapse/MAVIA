"""End-to-end tests for the Vision Inspector agent.

Marked ``integration`` because they need both the MVTec dataset and a fitted
memory bank on disk; CI runs without them. They are the tests that prove the
agent actually distinguishes a good part from a defective one, rather than
merely returning a well-formed object.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mavia.config import get_settings
from mavia.schemas import Verdict, VisionResult
from mavia.vision.inspector import VisionInspector

pytestmark = pytest.mark.integration

CATEGORY = "toothbrush"


def _require_assets() -> tuple[Path, Path]:
    settings = get_settings()
    model_path = settings.models_dir / "patchcore" / f"{CATEGORY}.pt"
    category_dir = settings.mvtec_dir / CATEGORY
    if not model_path.exists() or not category_dir.is_dir():
        pytest.skip(f"needs {model_path} and {category_dir}; run train_vision.py first")
    return model_path, category_dir


@pytest.fixture(scope="module")
def inspector() -> VisionInspector:
    _require_assets()
    return VisionInspector(save_artifacts=False)


def _first_image(split_dir: Path) -> Path:
    images = sorted(split_dir.glob("*.png"))
    if not images:
        pytest.skip(f"no images in {split_dir}")
    return images[0]


def test_good_part_passes(inspector: VisionInspector) -> None:
    _, category_dir = _require_assets()
    result = inspector.inspect(_first_image(category_dir / "test" / "good"), CATEGORY)

    assert isinstance(result, VisionResult)
    assert result.verdict == Verdict.PASS
    assert result.anomaly_score < 0.5
    assert result.regions == []


def test_defective_part_fails_and_is_localised(inspector: VisionInspector) -> None:
    _, category_dir = _require_assets()
    defect_dirs = [
        d for d in sorted((category_dir / "test").iterdir()) if d.is_dir() and d.name != "good"
    ]
    if not defect_dirs:
        pytest.skip("no defect classes present")

    result = inspector.inspect(_first_image(defect_dirs[0]), CATEGORY)

    assert result.verdict == Verdict.FAIL
    assert result.anomaly_score >= 0.5
    assert result.regions, "a failing part should have at least one localised region"

    box = result.regions[0]
    assert box.area_px > 0
    assert 0.0 <= box.peak_score <= 1.0
    assert box.x >= 0 and box.y >= 0


def test_category_is_inferred_from_an_mvtec_path(inspector: VisionInspector) -> None:
    _, category_dir = _require_assets()
    result = inspector.inspect(_first_image(category_dir / "test" / "good"))
    assert result.category == CATEGORY


def test_result_carries_provenance(inspector: VisionInspector) -> None:
    """Every field the audit trail and report depend on must be populated."""
    _, category_dir = _require_assets()
    result = inspector.inspect(_first_image(category_dir / "test" / "good"), CATEGORY)

    assert "PatchCore" in result.model_name
    assert result.latency_ms > 0
    assert result.decision_threshold == 0.5
    assert 0.0 <= result.anomaly_score <= 1.0


def test_artifacts_are_written_when_enabled(tmp_path: Path) -> None:
    settings = get_settings()
    _, category_dir = _require_assets()

    inspector = VisionInspector(save_artifacts=True)
    result = inspector.inspect(_first_image(category_dir / "test" / "good"), CATEGORY)

    assert result.anomaly_map_path is not None
    assert result.overlay_path is not None
    assert Path(result.anomaly_map_path).exists()
    assert Path(result.overlay_path).exists()
    assert str(settings.artifacts_dir) in result.anomaly_map_path


def test_separates_good_from_defective_across_the_split(inspector: VisionInspector) -> None:
    """The property that matters: scores must actually separate the two classes."""
    _, category_dir = _require_assets()

    good = [inspector.inspect(p, CATEGORY).anomaly_score
            for p in sorted((category_dir / "test" / "good").glob("*.png"))[:5]]  # fmt: skip

    defect_dirs = [
        d for d in sorted((category_dir / "test").iterdir()) if d.is_dir() and d.name != "good"
    ]
    bad = [inspector.inspect(p, CATEGORY).anomaly_score
           for p in sorted(defect_dirs[0].glob("*.png"))[:5]]  # fmt: skip

    assert max(good) < min(bad), f"scores overlap: good={good} defective={bad}"
