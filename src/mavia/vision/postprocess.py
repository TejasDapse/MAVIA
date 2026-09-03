"""Turn a continuous anomaly map into the discrete evidence the report cites.

The downstream agents cannot reason about a 224x224 float array. They need
"two regions, the larger 1,840 px at (31, 88), peak score 0.91". This module
performs that reduction, and also renders the heatmap overlay a human reviewer
sees in the dashboard before approving a line stop.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from mavia.schemas import BoundingBox


def normalise_map(anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    """Rescale raw distances to [0, 1] with the decision threshold pinned at 0.5.

    Raw PatchCore distances have no absolute meaning - their scale depends on the
    category's own memory bank - so they cannot be shown to a human or compared
    across products. Anchoring the calibrated threshold at 0.5 makes "above 0.5"
    mean "defective" for every category, which is what lets one dashboard and one
    risk policy serve all of them. Values at or above twice the threshold
    saturate at 1.0.
    """
    if threshold <= 0:
        return np.clip(anomaly_map, 0.0, 1.0)
    return np.clip(0.5 * anomaly_map / threshold, 0.0, 1.0)


def extract_regions(
    anomaly_map: np.ndarray,
    threshold: float,
    min_area: int = 16,
    max_regions: int = 5,
) -> list[BoundingBox]:
    """Threshold the map and return connected components as bounding boxes.

    Args:
        anomaly_map: 2-D map, expected already normalised to [0, 1].
        threshold: values at or above this are considered anomalous.
        min_area: drop specks below this pixel count - they are almost always
            sensor noise rather than a real flaw, and a report full of 3-pixel
            "defects" is worse than no report.
        max_regions: keep only the most severe regions, largest peak first.
    """
    binary = anomaly_map >= threshold
    if not binary.any():
        return []

    labelled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    regions: list[BoundingBox] = []

    for region_id in range(1, count + 1):
        region = labelled == region_id
        area = int(region.sum())
        if area < min_area:
            continue
        rows, cols = np.where(region)
        y, x = int(rows.min()), int(cols.min())
        regions.append(
            BoundingBox(
                x=x,
                y=y,
                width=int(cols.max() - x + 1),
                height=int(rows.max() - y + 1),
                area_px=area,
                peak_score=float(np.clip(anomaly_map[region].max(), 0.0, 1.0)),
            )
        )

    regions.sort(key=lambda box: box.peak_score, reverse=True)
    return regions[:max_regions]


def _colourise(anomaly_map: np.ndarray) -> np.ndarray:
    """Blue -> red heatmap without pulling in matplotlib at inference time."""
    values = np.clip(anomaly_map, 0.0, 1.0)
    red = np.clip(1.5 * values - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.5 * values - 1.25), 0.0, 1.0)
    blue = np.clip(1.0 - 1.8 * values, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def render_overlay(
    image: Image.Image,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
    regions: list[BoundingBox] | None = None,
) -> Image.Image:
    """Blend the heatmap over the product image, optionally boxing the regions."""
    base = image.convert("RGB").resize(
        (anomaly_map.shape[1], anomaly_map.shape[0]), Image.Resampling.BILINEAR
    )
    heat = Image.fromarray(_colourise(anomaly_map))
    blended = Image.blend(base, heat, alpha)

    if regions:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(blended)
        for box in regions:
            draw.rectangle(
                [box.x, box.y, box.x + box.width, box.y + box.height],
                outline=(255, 255, 255),
                width=2,
            )
    return blended
