#!/usr/bin/env python3
"""Evaluate the Root Cause Analyst over real inspections.

Runs the full Vision -> Retrieval -> Analyst chain on sampled defective images and
measures what can actually be checked automatically:

* **Citation grounding rate** - of the case_ids the analysis cites, how many exist
  in the set that was retrieved. Inventing a citation is the specific failure mode
  that makes "grounded in history" a lie, so it is measured rather than assumed.
* **Risk agreement** - does the assigned risk_level match the severity the
  knowledge base records for the defect actually present. The ground truth comes
  from the image's directory, which the pipeline never sees.
* **Action specificity** - does the recommended action name a concrete
  intervention rather than "investigate further".
* **Retrieval ablation** - the same images analysed with and without history, to
  show whether the memory layer changes the output or merely decorates it.

Runs with or without an API key; without one it measures the deterministic
fallback path, and says so.

Usage:
    uv run python scripts/eval_analysis.py --samples 20
    uv run python scripts/eval_analysis.py --samples 20 --no-ablation
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from mavia.agents.analyst import FALLBACK_MODEL_NAME, RootCauseAnalyst
from mavia.config import get_settings
from mavia.memory.knowledge import get_knowledge
from mavia.memory.retrieval import HistoryRetriever
from mavia.schemas import RetrievalResult, RiskLevel
from mavia.vision.inspector import VisionInspector

RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}

VAGUE_ACTIONS = (
    "investigate further",
    "look into",
    "review the process",
    "monitor the situation",
    "further analysis",
)


def sample_defective_images(root: Path, n: int, seed: int = 0) -> list[tuple[Path, str, str]]:
    """Return (image_path, category, true_defect_type) triples."""
    candidates: list[tuple[Path, str, str]] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_dir = category_dir / "test"
        if not test_dir.is_dir():
            continue
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            if defect_dir.name == "good":
                continue
            for image in sorted(defect_dir.glob("*.png"))[:3]:
                candidates.append((image, category_dir.name, defect_dir.name))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=settings.mvtec_dir)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=settings.artifacts_dir / "analysis_metrics.json"
    )
    args = parser.parse_args()

    using_llm = settings.anthropic_api_key is not None
    path_label = (
        f"LLM ({settings.llm_model})" if using_llm else "rule-based fallback (no ANTHROPIC_API_KEY)"
    )
    print(f"Analyst path : {path_label}")

    samples = sample_defective_images(args.root, args.samples, args.seed)
    if not samples:
        print(f"No defective images found under {args.root}.")
        return 1
    print(f"Samples      : {len(samples)} defective images\n")

    inspector = VisionInspector(save_artifacts=False)
    retriever = HistoryRetriever()
    analyst = RootCauseAnalyst(settings=settings)

    rows: list[dict[str, object]] = []
    for index, (image_path, category, true_defect) in enumerate(samples, start=1):
        vision = inspector.inspect(image_path, category)
        retrieval = retriever.retrieve(vision)

        started = time.perf_counter()
        analysis = analyst.analyse(vision, retrieval)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        retrieved_ids = {case.case_id for case in retrieval.cases}
        cited = analysis.cited_case_ids
        grounded = all(case_id in retrieved_ids for case_id in cited) if cited else None

        knowledge = get_knowledge(category, true_defect)
        expected_risk = knowledge.severity if knowledge else None
        risk_delta = (
            RISK_ORDER[analysis.risk_level] - RISK_ORDER[expected_risk]
            if expected_risk is not None
            else None
        )

        action = analysis.recommended_action.lower()
        specific = not any(phrase in action for phrase in VAGUE_ACTIONS)

        row = {
            "image": str(image_path),
            "category": category,
            "true_defect": true_defect,
            "detected": vision.verdict.value,
            "anomaly_score": round(vision.anomaly_score, 4),
            "n_retrieved": len(retrieval.cases),
            "top_retrieved_defect": retrieval.cases[0].defect_type if retrieval.cases else None,
            "retrieval_correct": bool(
                retrieval.cases and retrieval.cases[0].defect_type == true_defect
            ),
            "risk_level": analysis.risk_level.value,
            "expected_risk": expected_risk.value if expected_risk else None,
            "risk_delta": risk_delta,
            "confidence": round(analysis.confidence, 3),
            "n_cited": len(cited),
            "citations_grounded": grounded,
            "action_specific": specific,
            "latency_ms": round(elapsed_ms, 1),
            "fallback": FALLBACK_MODEL_NAME in analysis.model_name,
        }

        if not args.no_ablation:
            blind = RetrievalResult(
                query_text=retrieval.query_text,
                cases=[],
                top_k=retrieval.top_k,
                latency_ms=0.0,
                embedding_model=retrieval.embedding_model,
            )
            without = analyst.analyse(vision, blind)
            row["ablation_risk_without_history"] = without.risk_level.value
            row["ablation_confidence_without_history"] = round(without.confidence, 3)
            row["ablation_changed_risk"] = without.risk_level != analysis.risk_level

        rows.append(row)
        expected_label = expected_risk.value if expected_risk else "?"
        print(
            f"[{index:>3}/{len(samples)}] {category}/{true_defect:<22} "
            f"risk={analysis.risk_level.value:<8} (exp {expected_label:<8}) "
            f"conf={analysis.confidence:.2f} cited={len(cited)} {elapsed_ms:>7.0f}ms"
        )

    # ------------------------------------------------------------- aggregate
    with_citations = [r for r in rows if r["n_cited"]]
    grounding_rate = (
        float(np.mean([1.0 if r["citations_grounded"] else 0.0 for r in with_citations]))
        if with_citations
        else float("nan")
    )
    deltas = [r["risk_delta"] for r in rows if r["risk_delta"] is not None]
    summary = {
        "analyst_path": settings.llm_model if using_llm else FALLBACK_MODEL_NAME,
        "n_samples": len(rows),
        "citation_grounding_rate": grounding_rate,
        "share_with_citations": float(np.mean([1.0 if r["n_cited"] else 0.0 for r in rows])),
        "risk_exact_agreement": float(np.mean([1.0 if d == 0 else 0.0 for d in deltas]))
        if deltas
        else float("nan"),
        "risk_within_one_level": float(np.mean([1.0 if abs(d) <= 1 else 0.0 for d in deltas]))
        if deltas
        else float("nan"),
        "risk_under_called": float(np.mean([1.0 if d < 0 else 0.0 for d in deltas]))
        if deltas
        else float("nan"),
        "action_specificity": float(np.mean([1.0 if r["action_specific"] else 0.0 for r in rows])),
        "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
        "retrieval_top1_correct": float(
            np.mean([1.0 if r["retrieval_correct"] else 0.0 for r in rows])
        ),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
    }
    if not args.no_ablation:
        summary["ablation_risk_changed_share"] = float(
            np.mean([1.0 if r.get("ablation_changed_risk") else 0.0 for r in rows])
        )
        summary["ablation_mean_confidence_without_history"] = float(
            np.mean([r["ablation_confidence_without_history"] for r in rows])
        )

    print("\n" + "=" * 62)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<42}{value:>8.4f}")
        else:
            print(f"  {key:<42}{value:>8}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
