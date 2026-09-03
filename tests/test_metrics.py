"""Tests for the detection metrics.

The PRO tests are the important ones. PRO exists because pixel AUROC is
misleading on this problem: defects cover a tiny fraction of pixels, so a model
that finds one large defect and misses several small ones still scores well
pixel-wise. PRO weights each connected defect *region* equally, which is what
actually matters on a line - missing a small flaw is a missed flaw. The tests
below assert that distinction holds rather than just checking a number comes out.
"""

from __future__ import annotations

import math

import numpy as np

from mavia.vision.metrics import (
    evaluate,
    image_auroc,
    optimal_threshold,
    pixel_auroc,
    pro_score,
)


def _masks_with_two_regions(size: int = 64) -> np.ndarray:
    """One image with a large defect region and a small distant one."""
    mask = np.zeros((1, size, size), dtype=np.float32)
    mask[0, 5:25, 5:25] = 1.0  # large: 400 px
    mask[0, 50:54, 50:54] = 1.0  # small: 16 px
    return mask


def _background_gradient(size: int = 64, high: float = 0.4, seed: int = 0) -> np.ndarray:
    """Low-level background variation so the FPR sweep has somewhere to go."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, high, size=(1, size, size)).astype(np.float32)


# ------------------------------------------------------------------ image AUROC


def test_image_auroc_is_one_for_perfect_separation() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert image_auroc(labels, scores) == 1.0


def test_image_auroc_is_half_for_uninformative_scores() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert image_auroc(labels, scores) == 0.5


def test_image_auroc_is_nan_when_only_one_class_present() -> None:
    labels = np.zeros(5, dtype=int)
    assert math.isnan(image_auroc(labels, np.random.rand(5)))


# ------------------------------------------------------------------ pixel AUROC


def test_pixel_auroc_rewards_correct_localisation() -> None:
    masks = _masks_with_two_regions()
    perfect = masks.copy()
    assert pixel_auroc(masks, perfect) == 1.0


def test_pixel_auroc_is_nan_on_defect_free_masks() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    assert math.isnan(pixel_auroc(masks, np.random.rand(2, 8, 8)))


# ------------------------------------------------------------------------- PRO


def test_pro_is_high_when_both_regions_are_found() -> None:
    masks = _masks_with_two_regions()
    maps = masks + _background_gradient()
    assert pro_score(masks, maps) > 0.9


def test_pro_is_low_for_uninformative_maps() -> None:
    masks = _masks_with_two_regions()
    rng = np.random.default_rng(1)
    maps = rng.uniform(0, 1, size=masks.shape).astype(np.float32)
    assert pro_score(masks, maps) < 0.5


def test_pro_penalises_missing_a_small_region_far_more_than_pixel_auroc() -> None:
    """The headline property: PRO weights regions equally, pixel AUROC does not.

    A model that nails the 400px defect but completely misses the 16px one is
    barely penalised pixel-wise, yet has missed half the defects on the part.
    """
    masks = _masks_with_two_regions()
    background = _background_gradient()

    both_found = masks + background

    large_only = background.copy()
    large_only[0, 5:25, 5:25] += 1.0  # small region left at background level

    pro_both = pro_score(masks, both_found)
    pro_large = pro_score(masks, large_only)
    px_both = pixel_auroc(masks, both_found)
    px_large = pixel_auroc(masks, large_only)

    pro_drop = pro_both - pro_large
    pixel_drop = px_both - px_large

    assert pro_drop > 0.3, f"PRO barely moved ({pro_drop:.3f}) when a region was missed"
    assert pro_drop > pixel_drop * 2, (
        f"PRO drop {pro_drop:.3f} should dwarf pixel AUROC drop {pixel_drop:.3f}"
    )


def test_pro_is_nan_without_any_defect() -> None:
    masks = np.zeros((1, 16, 16), dtype=np.float32)
    assert math.isnan(pro_score(masks, np.random.rand(1, 16, 16)))


# ------------------------------------------------------------ threshold picking


def test_optimal_threshold_separates_a_clean_split() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.15, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95])
    threshold, f1, precision, recall = optimal_threshold(labels, scores)
    assert 0.25 < threshold <= 0.8
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_optimal_threshold_handles_single_class_gracefully() -> None:
    labels = np.zeros(4, dtype=int)
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold, f1, _, _ = optimal_threshold(labels, scores)
    assert math.isnan(f1)
    assert not math.isnan(threshold), "a usable fallback threshold is still expected"


# -------------------------------------------------------------------- evaluate


def test_evaluate_assembles_the_full_metric_set() -> None:
    masks = np.concatenate([_masks_with_two_regions(), np.zeros((1, 64, 64), dtype=np.float32)])
    maps = masks + _background_gradient(seed=3).repeat(2, axis=0)
    labels = np.array([1, 0])
    scores = np.array([0.9, 0.1])

    metrics = evaluate(labels, scores, masks=masks, anomaly_maps=maps)

    assert metrics.image_auroc == 1.0
    assert metrics.pixel_auroc is not None and metrics.pixel_auroc > 0.9
    assert metrics.pro is not None and metrics.pro > 0.9
    assert set(metrics.as_dict()) == {
        "image_auroc", "pixel_auroc", "pro", "optimal_threshold", "f1", "precision", "recall",
    }  # fmt: skip


def test_evaluate_without_masks_skips_pixel_metrics() -> None:
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert metrics.pixel_auroc is None
    assert metrics.pro is None
    assert metrics.image_auroc == 1.0
