#!/usr/bin/env python3
"""Fit a PatchCore memory bank per MVTec AD category and evaluate it.

"Training" is a misnomer kept for familiarity: no gradients are computed. The
script embeds the defect-free training images, selects a coreset, then scores the
test split and calibrates a per-category decision threshold.

Usage:
    uv run python scripts/train_vision.py -c toothbrush
    uv run python scripts/train_vision.py                    # all available
    uv run python scripts/train_vision.py --sampling-ratio 0.05 --no-pro
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from mavia.config import get_settings
from mavia.vision.dataset import MVTecDataset, available_categories
from mavia.vision.metrics import evaluate
from mavia.vision.patchcore import PatchCore, PatchCoreConfig


def _loader(dataset: MVTecDataset, batch_size: int) -> DataLoader:
    # num_workers=0: the bottleneck is the backbone forward pass, and worker
    # processes each re-import torch, which costs more than it saves here.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def run_category(
    category: str,
    root: Path,
    models_dir: Path,
    config: PatchCoreConfig,
    batch_size: int,
    compute_pro: bool,
    max_train_images: int | None,
) -> dict[str, object]:
    print(f"\n{'=' * 70}\n{category}\n{'=' * 70}")

    train_set = MVTecDataset(root, category, "train", config.image_size, config.crop_size)
    if max_train_images is not None and len(train_set) > max_train_images:
        train_set.samples = train_set.samples[:max_train_images]
    test_set = MVTecDataset(root, category, "test", config.image_size, config.crop_size)

    print(f"train {len(train_set)} defect-free | test {len(test_set)}")

    model = PatchCore(config=config)
    print(f"device: {model.device}")

    fit_start = time.perf_counter()
    model.fit(_loader(train_set, batch_size))
    fit_seconds = time.perf_counter() - fit_start
    print(f"  fit: {fit_seconds:.1f}s")

    score_start = time.perf_counter()
    output = model.predict_loader(_loader(test_set, batch_size))
    score_seconds = time.perf_counter() - score_start
    latency_ms = 1000.0 * score_seconds / max(1, len(test_set))
    print(f"  score: {score_seconds:.1f}s ({latency_ms:.1f} ms/image)")

    labels = test_set.labels()
    masks = np.stack([test_set[i]["mask"].squeeze(0).numpy() for i in range(len(test_set))])

    metrics = evaluate(
        labels, output.image_scores, masks=masks,
        anomaly_maps=output.anomaly_maps, compute_pro=compute_pro,
    )  # fmt: skip

    model_path = models_dir / "patchcore" / f"{category}.pt"
    model.save(model_path)

    print(
        f"  image AUROC {metrics.image_auroc:.4f} | "
        f"pixel AUROC {metrics.pixel_auroc:.4f} | "
        f"PRO {metrics.pro if metrics.pro is None else f'{metrics.pro:.4f}'} | "
        f"F1 {metrics.f1_at_threshold:.4f}"
    )

    return {
        "category": category,
        **metrics.as_dict(),
        "n_train": len(train_set),
        "n_test": len(test_set),
        "memory_bank_size": int(model.memory_bank.shape[0]) if model.memory_bank is not None else 0,
        "fit_seconds": round(fit_seconds, 2),
        "latency_ms_per_image": round(latency_ms, 2),
        "model_path": str(model_path),
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--category", action="append", dest="categories")
    parser.add_argument("--root", type=Path, default=settings.mvtec_dir)
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampling-ratio", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--no-pro", action="store_true", help="Skip PRO (it is the slow metric).")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=settings.artifacts_dir / "detection_metrics.json"
    )
    args = parser.parse_args()

    root: Path = args.root
    categories = args.categories or available_categories(root)
    if not categories:
        print(f"No MVTec categories found at {root}. Run scripts/download_mvtec.py first.")
        return 1

    config = PatchCoreConfig(
        sampling_ratio=args.sampling_ratio,
        image_size=args.image_size,
        crop_size=args.crop_size,
    )

    results: list[dict[str, object]] = []
    for category in categories:
        results.append(
            run_category(
                category,
                root,
                args.models_dir,
                config,
                args.batch_size,
                not args.no_pro,
                args.max_train_images,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"config": config.__dict__, "results": results}, indent=2, default=str)
    )

    # The inspector needs the calibrated thresholds at runtime; merge rather than
    # overwrite so training one category does not discard the others' calibration.
    thresholds_path = args.models_dir / "patchcore" / "thresholds.json"
    thresholds: dict[str, float] = {}
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text())
    for row in results:
        thresholds[str(row["category"])] = float(row["optimal_threshold"])  # type: ignore[arg-type]
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(dict(sorted(thresholds.items())), indent=2))
    print(f"Thresholds written to {thresholds_path}")

    print(f"\n{'=' * 70}\nSummary ({len(results)} categories)\n{'=' * 70}")
    header = f"{'category':<13}{'img AUROC':>11}{'px AUROC':>10}{'PRO':>8}{'ms/img':>9}"
    print(header)
    print("-" * len(header))
    for row in results:
        pro = row["pro"]
        pro_text = (
            "-" if pro is None or (isinstance(pro, float) and np.isnan(pro)) else f"{pro:.4f}"
        )
        print(
            f"{row['category']:<13}{row['image_auroc']:>11.4f}"
            f"{row['pixel_auroc']:>10.4f}{pro_text:>8}{row['latency_ms_per_image']:>9.1f}"
        )

    if len(results) > 1:
        print("-" * len(header))
        mean_img = float(np.mean([r["image_auroc"] for r in results]))
        mean_px = float(np.mean([r["pixel_auroc"] for r in results]))
        print(f"{'MEAN':<13}{mean_img:>11.4f}{mean_px:>10.4f}")

    print(f"\nMetrics written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
