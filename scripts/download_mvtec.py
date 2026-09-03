#!/usr/bin/env python3
"""Download and extract the MVTec AD benchmark into ``data/mvtec_ad``.

MVTec AD is released under CC BY-NC-SA 4.0 (research / non-commercial use).
Source: https://www.mvtec.com/company/research/datasets/mvtec-ad

Usage:
    uv run python scripts/download_mvtec.py                 # full dataset (~4.9 GB)
    uv run python scripts/download_mvtec.py --dry-run       # show what would happen
    uv run python scripts/download_mvtec.py --no-verify     # skip checksum check
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "mvtec_ad"

ARCHIVE_URL = (
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
)
ARCHIVE_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100 / total_size)
    bar = "=" * int(pct // 2)
    sys.stdout.write(
        f"\r  [{bar:<50}] {pct:5.1f}%  ({downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB)"
    )
    sys.stdout.flush()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def already_present(dest: Path) -> bool:
    return all((dest / category / "train" / "good").is_dir() for category in CATEGORIES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dest: Path = args.dest
    if already_present(dest):
        print(f"MVTec AD already present at {dest} ({len(CATEGORIES)} categories). Nothing to do.")
        return 0

    archive = dest.parent / "mvtec_anomaly_detection.tar.xz"
    print(f"Destination : {dest}")
    print(f"Archive     : {archive}")
    print(f"Source      : {ARCHIVE_URL}")
    print("License     : CC BY-NC-SA 4.0 (non-commercial research use)")
    if args.dry_run:
        print("\n--dry-run set, exiting without downloading.")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print("\nDownloading (~4.9 GB, this takes a while)...")
        urllib.request.urlretrieve(ARCHIVE_URL, archive, reporthook=_progress)
        print()
    else:
        print("\nArchive already downloaded, reusing it.")

    if not args.no_verify:
        print("Verifying checksum...")
        actual = sha256_of(archive)
        if actual != ARCHIVE_SHA256:
            print(
                f"  ERROR: checksum mismatch\n    expected {ARCHIVE_SHA256}\n    got      {actual}"
            )
            print(
                "  Delete the archive and retry, or rerun with --no-verify if you trust the source."
            )
            return 1
        print("  OK")

    print(f"Extracting to {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(dest, filter="data")

    # The archive unpacks category folders directly; normalise a nested root if present.
    nested = dest / "mvtec_anomaly_detection"
    if nested.is_dir():
        for child in nested.iterdir():
            child.rename(dest / child.name)
        nested.rmdir()

    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    found = [c for c in CATEGORIES if (dest / c).is_dir()]
    print(f"Done. {len(found)}/{len(CATEGORIES)} categories available at {dest}")
    return 0 if len(found) == len(CATEGORIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
