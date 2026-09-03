#!/usr/bin/env python3
"""Establish the information-theoretic ceiling for morphology-based retrieval.

Before tuning a retriever it is worth knowing what "good" can possibly mean. The
retriever sees only what the vision agent can measure: product category and
defect geometry. If two defect types of the same product produce
indistinguishable geometry - and MVTec's masks show that several do - then no
embedding model, re-ranker, or prompt can tell them apart.

This script measures that ceiling directly. Using the *real* per-mask features
(area fraction, region count, elongation) it runs leave-one-out k-NN within each
category and reports the precision@k an oracle operating on those exact features
would achieve. Any retriever's score should be read against this number, not
against 1.0.

Usage:
    uv run python scripts/retrieval_ceiling.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from mavia.config import get_settings
from mavia.memory.corpus import _mask_statistics
from mavia.memory.knowledge import KNOWLEDGE_BASE


def collect_features(root: Path) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Per-category feature matrix and defect labels, from the ground-truth masks."""
    per_category: dict[str, list[tuple[list[float], str]]] = defaultdict(list)

    for category, defect_type in sorted(KNOWLEDGE_BASE):
        mask_dir = root / category / "ground_truth" / defect_type
        if not mask_dir.is_dir():
            continue
        for path in sorted(mask_dir.glob("*.png")):
            array = np.asarray(Image.open(path).convert("L"))
            area, count, elongation = _mask_statistics(array)
            # log area: defect sizes span orders of magnitude, so a linear scale
            # would let one large-defect class dominate every distance.
            per_category[category].append(
                ([np.log10(max(area, 1e-6)), float(count), elongation], defect_type)
            )

    return {
        category: (np.array([f for f, _ in rows]), [label for _, label in rows])
        for category, rows in per_category.items()
    }


def leave_one_out_precision(
    features: np.ndarray, labels: list[str], top_k: int
) -> tuple[float, float]:
    """Precision@k and hit-rate@k for an oracle k-NN on these exact features."""
    standardised = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    distances = np.linalg.norm(standardised[:, None, :] - standardised[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)

    precisions, hits = [], []
    for index in range(len(labels)):
        neighbours = np.argsort(distances[index])[:top_k]
        matched = [labels[n] == labels[index] for n in neighbours]
        precisions.append(sum(matched) / len(matched))
        hits.append(1.0 if any(matched) else 0.0)
    return float(np.mean(precisions)), float(np.mean(hits))


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=settings.mvtec_dir)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=settings.artifacts_dir / "retrieval_ceiling.json"
    )
    args = parser.parse_args()

    per_category = collect_features(args.root)

    print(f"{'category':<13}{'masks':>7}{'modes':>7}{'chance':>9}{'ceiling':>9}{'hit@k':>8}")
    print("-" * 53)

    rows, weights = [], []
    for category, (features, labels) in sorted(per_category.items()):
        n_modes = len(set(labels))
        precision, hit_rate = leave_one_out_precision(features, labels, args.top_k)
        chance = 1.0 / n_modes
        rows.append(
            {
                "category": category,
                "n_masks": len(labels),
                "n_modes": n_modes,
                "chance": chance,
                "ceiling_precision": precision,
                "ceiling_hit_rate": hit_rate,
            }
        )
        weights.append(len(labels))
        print(
            f"{category:<13}{len(labels):>7}{n_modes:>7}{chance:>9.3f}{precision:>9.3f}{hit_rate:>8.3f}"
        )

    weights_array = np.array(weights, dtype=float)
    mean_ceiling = float(np.average([r["ceiling_precision"] for r in rows], weights=weights_array))
    mean_chance = float(np.average([r["chance"] for r in rows], weights=weights_array))
    mean_hit = float(np.average([r["ceiling_hit_rate"] for r in rows], weights=weights_array))

    print("-" * 53)
    print(
        f"{'WEIGHTED':<13}{int(weights_array.sum()):>7}{'':>7}{mean_chance:>9.3f}{mean_ceiling:>9.3f}{mean_hit:>8.3f}"
    )
    print(
        f"\nCeiling for morphology-only retrieval at k={args.top_k}: "
        f"precision {mean_ceiling:.3f} (chance {mean_chance:.3f})."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "weighted_ceiling_precision": mean_ceiling,
                "weighted_chance": mean_chance,
                "weighted_ceiling_hit_rate": mean_hit,
                "per_category": rows,
            },
            indent=2,
        )
    )
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
