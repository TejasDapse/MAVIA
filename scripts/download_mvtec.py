#!/usr/bin/env python3
"""Download MVTec AD and normalise it into the canonical benchmark layout.

MVTec's own distribution requires registration, and the ``mydrive.ch`` link that
anomalib historically used now returns 404. We therefore pull from an ungated
Hugging Face mirror that carries the complete dataset (5,354 images + 1,258
masks, matching the official counts) and reorganise it into the layout every
MVTec AD paper and codebase assumes:

    data/mvtec_ad/<category>/train/good/000.png
    data/mvtec_ad/<category>/test/<defect>/000.png
    data/mvtec_ad/<category>/ground_truth/<defect>/000_mask.png

MVTec AD is released under CC BY-NC-SA 4.0 (research / non-commercial use).
Source: https://www.mvtec.com/company/research/datasets/mvtec-ad

Usage:
    uv run python scripts/download_mvtec.py                    # all 15 categories
    uv run python scripts/download_mvtec.py -c bottle -c tile  # a subset
    uv run python scripts/download_mvtec.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "mvtec_ad"

HF_REPO = "TheoM55/mvtec_anomaly_detection"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip

# Official per-category image counts (train + test), used as an integrity check.
EXPECTED_COUNTS = {
    "bottle": (209, 83), "cable": (224, 150), "capsule": (219, 132),
    "carpet": (280, 117), "grid": (264, 78), "hazelnut": (391, 110),
    "leather": (245, 124), "metal_nut": (220, 115), "pill": (267, 167),
    "screw": (320, 160), "tile": (230, 117), "toothbrush": (60, 42),
    "transistor": (213, 100), "wood": (247, 79), "zipper": (240, 151),
}  # fmt: skip


def category_is_complete(dest: Path, category: str) -> bool:
    """True when a category already has roughly the expected number of images."""
    train_dir = dest / category / "train" / "good"
    test_dir = dest / category / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False
    n_train = len(list(train_dir.glob("*.png")))
    n_test = len(list(test_dir.rglob("*.png")))
    exp_train, exp_test = EXPECTED_COUNTS[category]
    return n_train == exp_train and n_test == exp_test


def reorganise(cache_dir: Path, dest: Path, categories: list[str]) -> None:
    """Move the mirror's split-first layout into the canonical category-first one."""
    for category in categories:
        # images/train/<cat>/good -> <cat>/train/good
        # images/test/<cat>/<defect> -> <cat>/test/<defect>
        for split in ("train", "test"):
            src = cache_dir / "images" / split / category
            if not src.is_dir():
                continue
            for defect_dir in sorted(src.iterdir()):
                if not defect_dir.is_dir():
                    continue
                target = dest / category / split / defect_dir.name
                target.mkdir(parents=True, exist_ok=True)
                for image in defect_dir.glob("*.png"):
                    shutil.copy2(image, target / image.name)

        # masks/test/<cat>/<defect> -> <cat>/ground_truth/<defect>
        mask_src = cache_dir / "masks" / "test" / category
        if mask_src.is_dir():
            for defect_dir in sorted(mask_src.iterdir()):
                if not defect_dir.is_dir():
                    continue
                target = dest / category / "ground_truth" / defect_dir.name
                target.mkdir(parents=True, exist_ok=True)
                for mask in defect_dir.glob("*.png"):
                    shutil.copy2(mask, target / mask.name)


def verify(dest: Path, categories: list[str]) -> bool:
    ok = True
    print("\nIntegrity check (train / test images per category):")
    for category in categories:
        n_train = len(list((dest / category / "train" / "good").glob("*.png")))
        n_test = len(list((dest / category / "test").rglob("*.png")))
        n_mask = len(list((dest / category / "ground_truth").rglob("*.png")))
        exp_train, exp_test = EXPECTED_COUNTS[category]
        good = n_train == exp_train and n_test == exp_test
        ok &= good
        flag = "ok " if good else "BAD"
        print(
            f"  {flag} {category:<12} train {n_train:>4}/{exp_train:<4} "
            f"test {n_test:>4}/{exp_test:<4} masks {n_mask:>4}"
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "-c", "--category", action="append", dest="categories",
        choices=CATEGORIES, help="Download only these categories (repeatable).",
    )  # fmt: skip
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-cache", action="store_true", help="Keep the raw HF snapshot after reorganising."
    )
    args = parser.parse_args()

    dest: Path = args.dest
    categories: list[str] = args.categories or CATEGORIES

    missing = [c for c in categories if not category_is_complete(dest, c)]
    print(f"Destination : {dest}")
    print(f"Source      : https://huggingface.co/datasets/{HF_REPO}")
    print("License     : CC BY-NC-SA 4.0 (non-commercial research use)")
    print(f"Categories  : {len(categories)} requested, {len(missing)} missing")

    if not missing:
        print("\nAll requested categories already present. Nothing to do.")
        return 0
    if args.dry_run:
        print(f"\n--dry-run set. Would download: {', '.join(missing)}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\nERROR: huggingface_hub is required. Run `uv sync`.", file=sys.stderr)
        return 1

    patterns = [f"images/*/{c}/**" for c in missing] + [f"masks/test/{c}/**" for c in missing]

    print(f"\nDownloading {len(missing)} categories (~5 GB for the full set)...")
    cache_dir = Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=patterns,
            cache_dir=str(dest.parent / ".hf_cache"),
        )
    )

    print(f"Reorganising into canonical MVTec layout at {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    reorganise(cache_dir, dest, missing)

    if not args.keep_cache:
        shutil.rmtree(dest.parent / ".hf_cache", ignore_errors=True)

    ok = verify(dest, categories)
    print("\nDone." if ok else "\nDone, but some counts do not match - see BAD rows above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
