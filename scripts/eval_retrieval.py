#!/usr/bin/env python3
"""Evaluate the History Retriever against the defect-history corpus.

Protocol: hold out one case per defect mode as the query, index the rest, and ask
whether the retrieved cases share the query's defect type. Retrieval is scored on
information the vision agent actually has - category and geometry - because the
defect type is never embedded.

Usage:
    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --top-k 5 --no-category-filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mavia.config import get_settings
from mavia.memory.corpus import build_corpus
from mavia.memory.store import DefectMemory


def evaluate(
    memory: DefectMemory,
    queries: list,
    top_k: int,
    use_category_filter: bool,
    n_per_mode: int,
    alpha: float | None = None,
) -> dict[str, float]:
    precisions, recalls, reciprocal_ranks, hits = [], [], [], []

    for case in queries:
        category = case.category if use_category_filter else None
        if alpha is None:
            results = memory.search(case.observation, top_k=top_k, category=category)
        else:
            results = memory.search_hybrid(
                case.observation,
                query_geometry=(case.area_fraction, case.region_count, case.elongation),
                top_k=top_k,
                category=category,
                alpha=alpha,
            )
        retrieved = [payload.get("defect_type") for payload, _ in results]
        relevant = [defect == case.defect_type for defect in retrieved]

        precisions.append(sum(relevant) / len(relevant) if relevant else 0.0)
        recalls.append(sum(relevant) / float(n_per_mode))
        hits.append(1.0 if any(relevant) else 0.0)
        rank = next((i + 1 for i, ok in enumerate(relevant) if ok), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return {
        f"precision@{top_k}": float(np.mean(precisions)),
        f"recall@{top_k}": float(np.mean(recalls)),
        f"hit_rate@{top_k}": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "n_queries": float(len(queries)),
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=settings.mvtec_dir)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cases-per-defect", type=int, default=10)
    parser.add_argument("--queries-per-defect", type=int, default=5)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--no-category-filter", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=settings.artifacts_dir / "retrieval_metrics.json"
    )
    args = parser.parse_args()

    # Split at the *mask* level, not the case level. Cases bootstrap real mask
    # measurements, so a case-level split would still let a query and an indexed
    # record derive from the same mask - the retriever would then be scored on
    # finding a duplicate of itself.
    indexed = build_corpus(
        args.root, cases_per_defect=args.cases_per_defect, mask_subset="index", seed=42
    )
    queries = build_corpus(
        args.root, cases_per_defect=args.queries_per_defect, mask_subset="query", seed=7
    )

    print(
        f"indexed {len(indexed)} cases | queries {len(queries)} "
        f"(disjoint mask pools, {int(args.holdout * 100)}% held out)"
    )

    memory = DefectMemory(settings=settings, collection="retrieval_eval")
    memory.ensure_collection(recreate=True)
    memory.index(indexed)

    results: dict[str, dict[str, float]] = {}

    print(f"\n{'configuration':<34}{'precision@k':>13}{'hit@k':>9}{'MRR':>8}")
    print("-" * 64)

    def report(label: str, metrics: dict[str, float]) -> None:
        results[label] = metrics
        print(
            f"{label:<34}{metrics[f'precision@{args.top_k}']:>13.4f}"
            f"{metrics[f'hit_rate@{args.top_k}']:>9.4f}{metrics['mrr']:>8.4f}"
        )

    report(
        "dense only, no category filter",
        evaluate(memory, queries, args.top_k, False, args.cases_per_defect),
    )
    report(
        "dense only, category filter",
        evaluate(memory, queries, args.top_k, True, args.cases_per_defect),
    )
    for alpha in (0.7, 0.5, 0.3, 0.0):
        report(
            f"hybrid alpha={alpha:.1f} (dense/geometry)",
            evaluate(memory, queries, args.top_k, True, args.cases_per_defect, alpha=alpha),
        )

    # Random baseline: chance of drawing a same-mode case within the category.
    from mavia.memory.knowledge import KNOWLEDGE_BASE

    modes_per_category: dict[str, int] = {}
    for category, _defect in KNOWLEDGE_BASE:
        modes_per_category[category] = modes_per_category.get(category, 0) + 1
    chance = float(np.mean([1.0 / n for n in modes_per_category.values()]))
    print(f"\nrandom baseline precision (within category): {chance:.4f}")
    results["random_baseline_within_category"] = {"precision": chance}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {args.out}")

    memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
