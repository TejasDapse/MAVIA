"""Tests for the Root Cause Analyst.

All of these run without an API key. The LLM path is exercised with a stub client
so the guardrails - citation verification and confidence capping - are tested
deterministically rather than against a live model that may or may not misbehave
on any given call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mavia.agents.analyst import FALLBACK_MODEL_NAME, AnalystOutput, RootCauseAnalyst
from mavia.config import Settings
from mavia.schemas import (
    BoundingBox,
    HistoricalCase,
    RetrievalResult,
    RiskLevel,
    Verdict,
    VisionResult,
)


def _vision(
    category: str = "bottle",
    verdict: Verdict = Verdict.FAIL,
    score: float = 0.81,
    regions: list[BoundingBox] | None = None,
    zero_shot: str | None = None,
) -> VisionResult:
    return VisionResult(
        category=category,
        verdict=verdict,
        anomaly_score=score,
        decision_threshold=0.5,
        regions=regions
        if regions is not None
        else [BoundingBox(x=72, y=53, width=146, height=171, area_px=13179, peak_score=0.94)],
        zero_shot_label=zero_shot,
        zero_shot_confidence=0.42 if zero_shot else None,
        model_name="PatchCore/WideResNet50-2/bottle",
        latency_ms=54.0,
    )


def _case(case_id: str, defect: str = "broken_large", similarity: float = 0.97) -> HistoricalCase:
    return HistoricalCase(
        case_id=case_id,
        category="bottle",
        defect_type=defect,
        root_cause="Thermal shock from an excessive cooling gradient in the annealing lehr",
        action_taken="Re-profile the annealing lehr temperature curve",
        outcome="Corrective action verified effective",
        occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
        similarity=similarity,
    )


def _retrieval(cases: list[HistoricalCase] | None = None) -> RetrievalResult:
    return RetrievalResult(
        query_text="Visual inspection of a bottle unit...",
        cases=cases if cases is not None else [_case("bottle-broken_large-000")],
        top_k=3,
        latency_ms=9.0,
        embedding_model="all-MiniLM-L6-v2",
    )


def _analyst_without_key() -> RootCauseAnalyst:
    return RootCauseAnalyst(settings=Settings(anthropic_api_key=None))


# ------------------------------------------------------------------- prompting


def test_prompt_carries_the_measured_evidence() -> None:
    prompt = RootCauseAnalyst.build_user_prompt(_vision(), _retrieval())
    assert "bottle" in prompt
    assert "0.810" in prompt
    assert "13179 px" in prompt
    assert "146x171" in prompt


def test_prompt_lists_retrieved_case_ids() -> None:
    retrieval = _retrieval([_case("bottle-a-001"), _case("bottle-b-002", "contamination")])
    prompt = RootCauseAnalyst.build_user_prompt(_vision(), retrieval)
    assert "bottle-a-001" in prompt
    assert "bottle-b-002" in prompt
    assert "contamination" in prompt


def test_prompt_flags_zero_shot_detections_as_weak() -> None:
    prompt = RootCauseAnalyst.build_user_prompt(_vision(zero_shot="crack"), _retrieval())
    assert "zero-shot" in prompt.lower()
    assert "weak evidence" in prompt.lower()


def test_prompt_states_when_no_history_was_found() -> None:
    prompt = RootCauseAnalyst.build_user_prompt(_vision(), _retrieval([]))
    assert "No comparable cases were retrieved" in prompt
    assert "reduce your confidence" in prompt


def test_prompt_handles_a_detection_with_no_regions() -> None:
    prompt = RootCauseAnalyst.build_user_prompt(_vision(regions=[]), _retrieval())
    assert "No discrete defect regions" in prompt


# ---------------------------------------------------------------- citations


def test_verify_citations_keeps_only_real_ids() -> None:
    retrieval = _retrieval([_case("real-001"), _case("real-002")])
    kept, dropped = RootCauseAnalyst.verify_citations(
        ["real-001", "invented-999", "real-002"], retrieval
    )
    assert kept == ["real-001", "real-002"]
    assert dropped == ["invented-999"]


def test_verify_citations_with_nothing_retrieved() -> None:
    kept, dropped = RootCauseAnalyst.verify_citations(["anything"], _retrieval([]))
    assert kept == []
    assert dropped == ["anything"]


# ------------------------------------------------------------------- fallback


def test_fallback_uses_the_nearest_precedent() -> None:
    analysis = _analyst_without_key().analyse(_vision(), _retrieval())

    assert FALLBACK_MODEL_NAME in analysis.model_name
    assert "bottle-broken_large-000" in analysis.cited_case_ids
    assert "annealing lehr" in analysis.recommended_action
    assert analysis.confidence <= 0.45


def test_fallback_risk_comes_from_the_knowledge_base() -> None:
    """broken_large is CRITICAL in the knowledge base; the fallback must reflect that."""
    analysis = _analyst_without_key().analyse(_vision(), _retrieval())
    assert analysis.risk_level == RiskLevel.CRITICAL


def test_fallback_confidence_scales_with_similarity() -> None:
    strong = _analyst_without_key().analyse(_vision(), _retrieval([_case("c1", similarity=0.99)]))
    weak = _analyst_without_key().analyse(_vision(), _retrieval([_case("c2", similarity=0.30)]))
    assert weak.confidence < strong.confidence


def test_fallback_without_any_history_escalates_to_a_human() -> None:
    analysis = _analyst_without_key().analyse(_vision(), _retrieval([]))

    assert analysis.cited_case_ids == []
    assert analysis.confidence <= 0.1
    assert analysis.risk_level == RiskLevel.HIGH
    assert "quality engineer" in analysis.recommended_action


def test_passing_part_is_not_marked_high_risk_by_the_fallback() -> None:
    analysis = _analyst_without_key().analyse(
        _vision(verdict=Verdict.PASS, score=0.2, regions=[]), _retrieval()
    )
    assert analysis.risk_level == RiskLevel.LOW


def test_fallback_records_latency() -> None:
    analysis = _analyst_without_key().analyse(_vision(), _retrieval())
    assert analysis.latency_ms >= 0.0


# ------------------------------------------------------------ stubbed LLM path


class _StubResponse:
    def __init__(self, parsed: AnalystOutput) -> None:
        self.parsed_output = parsed


class _StubMessages:
    def __init__(self, parsed: AnalystOutput | Exception) -> None:
        self._parsed = parsed
        self.last_kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> _StubResponse:
        self.last_kwargs = kwargs
        if isinstance(self._parsed, Exception):
            raise self._parsed
        return _StubResponse(self._parsed)


class _StubClient:
    def __init__(self, parsed: AnalystOutput | Exception) -> None:
        self.messages = _StubMessages(parsed)


def _output(**overrides: Any) -> AnalystOutput:
    base: dict[str, Any] = {
        "root_cause": "Thermal shock during annealing",
        "evidence": ["Anomaly score 0.81", "13179 px affected"],
        "recommended_action": "Re-profile the annealing lehr curve",
        "risk_level": RiskLevel.CRITICAL,
        "confidence": 0.88,
        "affected_process_step": "glass forming / annealing",
        "cited_case_ids": ["bottle-broken_large-000"],
    }
    base.update(overrides)
    return AnalystOutput(**base)


def test_llm_path_returns_the_parsed_analysis() -> None:
    client = _StubClient(_output())
    analyst = RootCauseAnalyst(settings=Settings(), client=client)

    analysis = analyst.analyse(_vision(), _retrieval())

    assert analysis.root_cause == "Thermal shock during annealing"
    assert analysis.risk_level == RiskLevel.CRITICAL
    assert analysis.confidence == 0.88
    assert analysis.cited_case_ids == ["bottle-broken_large-000"]
    assert FALLBACK_MODEL_NAME not in analysis.model_name


def test_hallucinated_citations_are_stripped_and_confidence_capped() -> None:
    """The guardrail that makes 'grounded in history' measurable rather than assumed."""
    client = _StubClient(
        _output(cited_case_ids=["bottle-broken_large-000", "totally-made-up"], confidence=0.95)
    )
    analyst = RootCauseAnalyst(settings=Settings(), client=client)

    analysis = analyst.analyse(_vision(), _retrieval())

    assert analysis.cited_case_ids == ["bottle-broken_large-000"]
    assert analysis.confidence <= 0.5, "inventing a citation must reduce trust in the rest"


def test_api_failure_degrades_to_the_fallback() -> None:
    client = _StubClient(RuntimeError("upstream 529 overloaded"))
    analyst = RootCauseAnalyst(settings=Settings(), client=client)

    analysis = analyst.analyse(_vision(), _retrieval())

    assert FALLBACK_MODEL_NAME in analysis.model_name
    assert "529" in analysis.model_name
    assert analysis.root_cause, "a degraded analysis must still be produced"


def test_request_uses_structured_output_and_a_cached_system_prompt() -> None:
    client = _StubClient(_output())
    analyst = RootCauseAnalyst(settings=Settings(), client=client)
    analyst.analyse(_vision(), _retrieval())

    kwargs = client.messages.last_kwargs
    assert kwargs["output_format"] is AnalystOutput
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["thinking"] == {"type": "adaptive"}
