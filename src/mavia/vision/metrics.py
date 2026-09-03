"""Evaluation metrics for anomaly detection.

These are the metrics MVTec AD is actually benchmarked with, so results here are
directly comparable to published numbers:

* **Image AUROC** - can the system tell a defective unit from a good one at all.
* **Pixel AUROC** - is the defect localised to the right place. Dominated by the
  background, because defects occupy a tiny fraction of pixels, so a high value
  is necessary but not sufficient.
* **PRO** - per-region overlap. Averages coverage over each *connected defect
  region* rather than over pixels, so one large defect cannot mask the model's
  failure to find several small ones. This is the metric that best reflects
  whether an operator would actually be shown the flaw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from sklearn.metrics import auc, roc_auc_score, roc_curve


@dataclass(frozen=True)
class DetectionMetrics:
    image_auroc: float
    pixel_auroc: float | None
    pro: float | None
    optimal_threshold: float
    f1_at_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "image_auroc": self.image_auroc,
            "pixel_auroc": self.pixel_auroc,
            "pro": self.pro,
            "optimal_threshold": self.optimal_threshold,
            "f1": self.f1_at_threshold,
            "precision": self.precision_at_threshold,
            "recall": self.recall_at_threshold,
        }


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC over image-level scores. Undefined when only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """AUROC over every pixel of every image."""
    flat_masks = masks.reshape(-1).astype(np.int8)
    if len(np.unique(flat_masks)) < 2:
        return float("nan")
    return float(roc_auc_score(flat_masks, anomaly_maps.reshape(-1)))


def pro_score(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """Per-Region Overlap, integrated up to ``max_fpr`` and normalised.

    For each threshold: label the connected components of the ground truth, take
    the fraction of each component covered by the prediction, and average those
    fractions *per region*. Plotting that against the false-positive rate on
    defect-free pixels and integrating to 0.3 FPR gives the standard PRO-AUC.
    """
    if masks.size == 0 or masks.max() == 0:
        return float("nan")

    structure = np.ones((3, 3), dtype=int)
    # Pre-label each image's regions once rather than inside the threshold loop.
    labelled: list[tuple[np.ndarray, int]] = []
    for mask in masks:
        components, count = ndimage.label(mask, structure=structure)
        labelled.append((components, count))

    lo = float(anomaly_maps.min())
    hi = float(anomaly_maps.max())
    if hi <= lo:
        return float("nan")
    thresholds = np.linspace(lo, hi, num_thresholds)

    inverse_masks = 1 - masks
    total_normal = float(inverse_masks.sum())
    if total_normal == 0:
        return float("nan")

    pros: list[float] = []
    fprs: list[float] = []

    for threshold in thresholds:
        predictions = anomaly_maps >= threshold

        overlaps: list[float] = []
        for (components, count), prediction in zip(labelled, predictions, strict=True):
            for region_id in range(1, count + 1):
                region = components == region_id
                area = float(region.sum())
                if area > 0:
                    overlaps.append(float((prediction & region).sum()) / area)

        if not overlaps:
            continue

        pros.append(float(np.mean(overlaps)))
        fprs.append(float((predictions & (inverse_masks == 1)).sum()) / total_normal)

    if len(pros) < 2:
        return float("nan")

    fpr_array = np.asarray(fprs)
    pro_array = np.asarray(pros)
    order = np.argsort(fpr_array)
    fpr_array, pro_array = fpr_array[order], pro_array[order]

    keep = fpr_array <= max_fpr
    if keep.sum() < 2:
        return float("nan")

    # Normalise the x-axis to [0, 1] so the value is comparable to published PRO.
    return float(auc(fpr_array[keep] / max_fpr, pro_array[keep]))


def optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Pick the score threshold maximising F1, returning (threshold, f1, precision, recall).

    A fixed global threshold would be wrong here: each category's score
    distribution has a different scale, because the distances depend on the
    category's own memory bank. Calibrating per category is what makes the
    PASS/FAIL decision meaningful rather than arbitrary.
    """
    if len(np.unique(labels)) < 2:
        return float(np.median(scores)), float("nan"), float("nan"), float("nan")

    false_pos, true_pos, thresholds = roc_curve(labels, scores)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)

    tp = true_pos * n_pos
    fp = false_pos * n_neg
    fn = n_pos - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best]), float(precision[best]), float(recall[best])


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray | None = None,
    anomaly_maps: np.ndarray | None = None,
    compute_pro: bool = True,
) -> DetectionMetrics:
    """Compute the full metric set for one category."""
    threshold, f1, precision, recall = optimal_threshold(labels, scores)
    px_auroc = (
        pixel_auroc(masks, anomaly_maps) if masks is not None and anomaly_maps is not None else None
    )
    pro = (
        pro_score(masks, anomaly_maps)
        if compute_pro and masks is not None and anomaly_maps is not None
        else None
    )
    return DetectionMetrics(
        image_auroc=image_auroc(labels, scores),
        pixel_auroc=px_auroc,
        pro=pro,
        optimal_threshold=threshold,
        f1_at_threshold=f1,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
    )
