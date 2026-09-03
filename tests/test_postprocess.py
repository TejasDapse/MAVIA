"""Tests for anomaly-map post-processing.

These cover the reduction from a float map to the discrete regions the QA report
cites as evidence, plus the normalisation that makes scores comparable across
product categories.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from mavia.vision.postprocess import extract_regions, normalise_map, render_overlay


def _map_with_regions(size: int = 64) -> np.ndarray:
    anomaly_map = np.zeros((size, size), dtype=np.float32)
    anomaly_map[5:25, 5:25] = 0.9  # large region, 400 px
    anomaly_map[50:58, 50:58] = 0.7  # smaller region, 64 px
    anomaly_map[40, 10] = 0.95  # single-pixel speck: noise, must be dropped
    return anomaly_map


# ----------------------------------------------------------------- normalisation


def test_threshold_maps_to_one_half() -> None:
    """The whole point: the calibrated threshold becomes 0.5 for every category."""
    raw = np.array([[4.0]], dtype=np.float32)
    assert normalise_map(raw, threshold=4.0)[0, 0] == 0.5


def test_normalisation_saturates_at_twice_the_threshold() -> None:
    raw = np.array([[0.0, 4.0, 8.0, 20.0]], dtype=np.float32)
    normalised = normalise_map(raw, threshold=4.0)
    assert list(normalised[0]) == [0.0, 0.5, 1.0, 1.0]


def test_categories_with_different_scales_become_comparable() -> None:
    """Two categories, wildly different raw distance scales, same normalised score."""
    coarse = normalise_map(np.array([[12.0]], dtype=np.float32), threshold=8.0)
    fine = normalise_map(np.array([[0.15]], dtype=np.float32), threshold=0.1)
    assert np.isclose(coarse[0, 0], fine[0, 0])


def test_non_positive_threshold_is_handled() -> None:
    raw = np.array([[0.3, 2.0]], dtype=np.float32)
    assert list(normalise_map(raw, threshold=0.0)[0]) == [0.3, 1.0]


# -------------------------------------------------------------- region extraction


def test_finds_both_real_regions_and_drops_the_speck() -> None:
    regions = extract_regions(_map_with_regions(), threshold=0.5, min_area=16)
    assert len(regions) == 2
    areas = sorted(box.area_px for box in regions)
    assert areas == [64, 400]


def test_regions_are_ordered_by_severity() -> None:
    regions = extract_regions(_map_with_regions(), threshold=0.5, min_area=16)
    peaks = [box.peak_score for box in regions]
    assert peaks == sorted(peaks, reverse=True)


def test_bounding_box_bounds_the_region_exactly() -> None:
    anomaly_map = np.zeros((32, 32), dtype=np.float32)
    anomaly_map[10:20, 4:12] = 0.8
    box = extract_regions(anomaly_map, threshold=0.5, min_area=1)[0]
    assert (box.x, box.y, box.width, box.height) == (4, 10, 8, 10)
    assert box.area_px == 80


def test_clean_map_yields_no_regions() -> None:
    assert extract_regions(np.full((32, 32), 0.1, dtype=np.float32), threshold=0.5) == []


def test_min_area_filters_small_regions() -> None:
    anomaly_map = np.zeros((32, 32), dtype=np.float32)
    anomaly_map[5:8, 5:8] = 0.9  # 9 px
    assert extract_regions(anomaly_map, threshold=0.5, min_area=16) == []
    assert len(extract_regions(anomaly_map, threshold=0.5, min_area=4)) == 1


def test_max_regions_keeps_only_the_worst() -> None:
    anomaly_map = np.zeros((64, 64), dtype=np.float32)
    for i, peak in enumerate([0.6, 0.7, 0.8, 0.9]):
        anomaly_map[i * 12 : i * 12 + 6, 0:6] = peak
    regions = extract_regions(anomaly_map, threshold=0.5, min_area=4, max_regions=2)
    assert len(regions) == 2
    assert regions[0].peak_score > regions[1].peak_score
    assert np.isclose(regions[0].peak_score, 0.9, atol=1e-6)


def test_diagonally_touching_pixels_form_one_region() -> None:
    """8-connectivity: a diagonal crack is one flaw, not a string of separate ones."""
    anomaly_map = np.zeros((16, 16), dtype=np.float32)
    for i in range(6):
        anomaly_map[i + 2, i + 2] = 0.9
    regions = extract_regions(anomaly_map, threshold=0.5, min_area=1)
    assert len(regions) == 1
    assert regions[0].area_px == 6


# ------------------------------------------------------------------- overlay


def test_overlay_matches_the_map_dimensions() -> None:
    image = Image.new("RGB", (512, 512), color=(120, 120, 120))
    anomaly_map = _map_with_regions()
    overlay = render_overlay(image, anomaly_map)
    assert overlay.size == (anomaly_map.shape[1], anomaly_map.shape[0])
    assert overlay.mode == "RGB"


def test_overlay_draws_region_boxes() -> None:
    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    anomaly_map = _map_with_regions()
    regions = extract_regions(anomaly_map, threshold=0.5, min_area=16)

    plain = np.asarray(render_overlay(image, anomaly_map))
    boxed = np.asarray(render_overlay(image, anomaly_map, regions=regions))

    assert not np.array_equal(plain, boxed), "region boxes should be visible in the overlay"
