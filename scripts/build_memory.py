#!/usr/bin/env python3
"""Generate the defect-history corpus and index it into the vector store.

Usage:
    uv run python scripts/build_memory.py
    uv run python scripts/build_memory.py --cases-per-defect 20 --recreate
    uv run python scripts/build_memory.py --dump artifacts/defect_corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mavia.config import get_settings
from mavia.memory.corpus import build_corpus
from mavia.memory.store import DefectMemory


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=settings.mvtec_dir)
    parser.add_argument("--cases-per-defect", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection.")
    parser.add_argument("--dump", type=Path, default=settings.artifacts_dir / "defect_corpus.json")
    args = parser.parse_args()

    print(f"Dataset root : {args.root}")
    print(f"Cases/defect : {args.cases_per_defect}")

    print("\nMeasuring defect morphology from ground-truth masks...")
    corpus = build_corpus(args.root, cases_per_defect=args.cases_per_defect, seed=args.seed)
    categories = sorted({case.category for case in corpus})
    defect_modes = {(case.category, case.defect_type) for case in corpus}
    print(
        f"  {len(corpus)} cases | {len(defect_modes)} defect modes | {len(categories)} categories"
    )

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps([c.to_payload() for c in corpus], indent=2))
        print(f"  corpus written to {args.dump}")

    print("\nIndexing into the vector store...")
    memory = DefectMemory(settings=settings)
    memory.ensure_collection(recreate=args.recreate)
    indexed = memory.index(corpus)
    print(f"  indexed {indexed} cases; collection now holds {memory.count()}")

    print("\nSample retrieval:")
    sample = corpus[0]
    for payload, score in memory.search(sample.observation, top_k=3, category=sample.category):
        print(f"  {score:.3f}  {payload['case_id']:<32} {payload['defect_type']}")

    memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
