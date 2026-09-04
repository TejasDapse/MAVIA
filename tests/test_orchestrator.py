"""Tests for the LangGraph inspection pipeline.

Stub agents throughout - no models are loaded, so these run in CI in under a
second. What is being tested is the *orchestration*: routing, suspension,
durability, fail-closed approval, and audit completeness. The agents themselves
are covered by their own test modules.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from mavia.audit import AuditLogger, verify_chain
from mavia.config import Settings
from mavia.orchestrator.graph import InspectionPipeline, _coerce_decision, _serializer
from mavia.schemas import (
    ApprovalStatus,
    BoundingBox,
    HistoricalCase,
    RetrievalResult,
    RiskLevel,
    RootCauseAnalysis,
    Verdict,
    VisionResult,
)

# ------------------------------------------------------------------- stubs


class StubInspector:
    def __init__(self, score: float = 0.9, verdict: Verdict = Verdict.FAIL) -> None:
        self.score = score
        self.verdict = verdict
        self.calls = 0

    def inspect(self, image_path: Any, category: str | None = None) -> VisionResult:
        self.calls += 1
        return VisionResult(
            category=category or "bottle",
            verdict=self.verdict,
            anomaly_score=self.score,
            decision_threshold=0.5,
            regions=[BoundingBox(x=1, y=2, width=30, height=40, area_px=1200, peak_score=0.9)],
            model_name="stub-detector",
            latency_ms=1.0,
        )


class StubRetriever:
    def __init__(self) -> None:
        self.memory = None

    def retrieve(self, vision: VisionResult) -> RetrievalResult:
        return RetrievalResult(
            query_text="stub query",
            cases=[
                HistoricalCase(
                    case_id="bottle-broken_large-000",
                    category="bottle",
                    defect_type="broken_large",
                    root_cause="Thermal shock",
                    action_taken="Re-profile the lehr",
                    outcome=None,
                    occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
                    similarity=0.95,
                )
            ],
            top_k=3,
            latency_ms=1.0,
            embedding_model="stub-embedder",
        )


class StubAnalyst:
    def __init__(self, risk: RiskLevel = RiskLevel.CRITICAL) -> None:
        self.risk = risk

    def analyse(self, vision: VisionResult, retrieval: RetrievalResult) -> RootCauseAnalysis:
        return RootCauseAnalysis(
            root_cause="Thermal shock during annealing",
            evidence=["score high", "large region"],
            recommended_action="Re-profile the annealing lehr curve",
            risk_level=self.risk,
            confidence=0.8,
            affected_process_step="annealing",
            cited_case_ids=["bottle-broken_large-000"],
            latency_ms=1.0,
            model_name="stub-analyst",
        )


class FailingInspector:
    def inspect(self, image_path: Any, category: str | None = None) -> VisionResult:
        raise RuntimeError("camera offline")


def _pipeline(
    tmp_path: Path,
    *,
    score: float = 0.9,
    risk: RiskLevel = RiskLevel.CRITICAL,
    verdict: Verdict = Verdict.FAIL,
    inspector: Any = None,
    connection: sqlite3.Connection | None = None,
) -> InspectionPipeline:
    settings = Settings(artifacts_dir=tmp_path, high_risk_threshold=0.75)
    connection = connection or sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False)
    saver = SqliteSaver(connection, serde=_serializer())
    saver.setup()
    return InspectionPipeline(
        settings=settings,
        inspector=inspector or StubInspector(score=score, verdict=verdict),
        retriever=StubRetriever(),
        analyst=StubAnalyst(risk=risk),
        audit=AuditLogger(tmp_path / "audit.jsonl"),
        checkpointer=saver,
    )


# ------------------------------------------------------------------ routing


def test_low_risk_completes_without_a_human(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, score=0.55, risk=RiskLevel.LOW)
    state = pipeline.run("part.png", "bottle")

    assert not pipeline.is_suspended(state.inspection_id)
    assert state.approval.status == ApprovalStatus.NOT_REQUIRED
    assert state.total_latency_ms is not None


def test_high_risk_analysis_escalates(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, score=0.55, risk=RiskLevel.HIGH)
    state = pipeline.run("part.png", "bottle")

    assert pipeline.is_suspended(state.inspection_id)
    assert "risk_level=HIGH" in pipeline.pending_approval(state.inspection_id)["reason"]


def test_high_anomaly_score_escalates_even_at_low_risk(tmp_path: Path) -> None:
    """The two signals are OR-ed: a disagreement escalates rather than resolves."""
    pipeline = _pipeline(tmp_path, score=0.92, risk=RiskLevel.LOW)
    state = pipeline.run("part.png", "bottle")

    reason = pipeline.pending_approval(state.inspection_id)["reason"]
    assert "anomaly_score" in reason
    assert "risk_level" not in reason


def test_both_signals_are_reported_when_both_fire(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, score=0.92, risk=RiskLevel.CRITICAL)
    state = pipeline.run("part.png", "bottle")

    reason = pipeline.pending_approval(state.inspection_id)["reason"]
    assert "risk_level=CRITICAL" in reason
    assert "anomaly_score" in reason


# --------------------------------------------------------------- interrupt


def test_interrupt_payload_carries_what_a_reviewer_needs(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    state = pipeline.run("part.png", "bottle")

    payload = pipeline.pending_approval(state.inspection_id)
    for key in ("inspection_id", "reason", "risk_level", "root_cause", "recommended_action"):
        assert payload.get(key) is not None, f"reviewer cannot decide without {key}"
    assert payload["cited_case_ids"] == ["bottle-broken_large-000"]


def test_approval_resumes_and_completes(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    state = pipeline.run("part.png", "bottle")

    resumed = pipeline.resume(
        state.inspection_id, approved=True, approver="qa.lead", rationale="confirmed"
    )

    assert resumed.approval.status == ApprovalStatus.APPROVED
    assert resumed.approval.approver == "qa.lead"
    assert resumed.approval.rationale == "confirmed"
    assert resumed.final_verdict == Verdict.FAIL
    assert not pipeline.is_suspended(state.inspection_id)


def test_rejection_makes_the_verdict_inconclusive(tmp_path: Path) -> None:
    """A rejected analysis must not be recorded as a confirmed verdict."""
    pipeline = _pipeline(tmp_path)
    state = pipeline.run("part.png", "bottle")

    resumed = pipeline.resume(state.inspection_id, approved=False, approver="qa.lead")

    assert resumed.approval.status == ApprovalStatus.REJECTED
    assert resumed.final_verdict == Verdict.INCONCLUSIVE


# --------------------------------------------------------------- durability


def test_pending_approval_survives_losing_the_pipeline_object(tmp_path: Path) -> None:
    """The property that separates an agent from a script.

    The first pipeline is discarded entirely after suspending - as if the process
    had exited. A completely new pipeline, sharing only the checkpoint database,
    must find the pending approval and resume it.
    """
    connection = sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False)

    first = _pipeline(tmp_path, connection=connection)
    state = first.run("part.png", "bottle")
    inspection_id = state.inspection_id
    assert first.is_suspended(inspection_id)
    del first

    second = _pipeline(tmp_path, connection=connection)
    assert second.is_suspended(inspection_id), "the pause did not survive"

    payload = second.pending_approval(inspection_id)
    assert payload["root_cause"] == "Thermal shock during annealing"

    resumed = second.resume(inspection_id, approved=True, approver="late.reviewer")
    assert resumed.approval.status == ApprovalStatus.APPROVED
    assert resumed.approval.approver == "late.reviewer"


# ------------------------------------------------------------ fail-closed


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "APPROVED", "approver": "a"}, ApprovalStatus.APPROVED),
        ({"approved": True}, ApprovalStatus.APPROVED),
        (True, ApprovalStatus.APPROVED),
        ("APPROVE", ApprovalStatus.APPROVED),
        ({"status": "REJECTED"}, ApprovalStatus.REJECTED),
        (False, ApprovalStatus.REJECTED),
        (None, ApprovalStatus.REJECTED),
        ({}, ApprovalStatus.REJECTED),
        ("gibberish", ApprovalStatus.REJECTED),
        (42, ApprovalStatus.REJECTED),
    ],
)
def test_unrecognised_decisions_fail_closed(payload: Any, expected: ApprovalStatus) -> None:
    """A gate that fails open is not a gate."""
    assert _coerce_decision(payload, "reason").status == expected


# ------------------------------------------------------------------- audit


def test_every_stage_is_audited(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, score=0.55, risk=RiskLevel.LOW)
    state = pipeline.run("part.png", "bottle")

    actions = [e.action for e in pipeline.audit.events_for(state.inspection_id)]
    agents = {str(e.agent) for e in pipeline.audit.events_for(state.inspection_id)}

    assert actions[0] == "inspection_started"
    assert actions[-1] == "finalised"
    assert actions.count("completed") == 4, (
        "vision, retrieval, analysis and report must each complete"
    )
    assert {
        "vision_inspector",
        "history_retriever",
        "root_cause_analyst",
        "report_writer",
    } <= agents
    assert "report_written" in actions, "the QA record must be produced and audited"


def test_approval_decision_is_audited(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    state = pipeline.run("part.png", "bottle")
    pipeline.resume(state.inspection_id, approved=True, approver="qa.lead", rationale="ok")

    approvals = [
        e for e in pipeline.audit.events_for(state.inspection_id) if "approval" in e.action
    ]
    assert approvals, "the human decision must be recorded"
    assert approvals[0].action == "approval_approved"
    assert approvals[0].payload["approver"] == "qa.lead"


def test_audit_chain_is_intact_after_a_full_run(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    state = pipeline.run("part.png", "bottle")
    pipeline.resume(state.inspection_id, approved=True, approver="qa.lead")

    result = verify_chain(tmp_path / "audit.jsonl")
    assert result
    assert result.checked >= 9


def test_a_failing_node_is_recorded_and_does_not_crash(tmp_path: Path) -> None:
    """A camera fault must produce an audited, degraded inspection - not a traceback."""
    pipeline = _pipeline(tmp_path, inspector=FailingInspector())
    state = pipeline.run("part.png", "bottle")

    assert state.errors, "the failure must surface in the state"
    assert any("camera offline" in error for error in state.errors)

    failures = [e for e in pipeline.audit.events_for(state.inspection_id) if e.action == "failed"]
    assert failures
    assert "camera offline" in failures[0].payload["error"]
    assert verify_chain(tmp_path / "audit.jsonl")


def test_audit_payloads_exclude_bulky_fields(tmp_path: Path) -> None:
    """Hash payloads stay compact; the chain covers decisions, not pixels."""
    pipeline = _pipeline(tmp_path, score=0.55, risk=RiskLevel.LOW)
    state = pipeline.run("part.png", "bottle")

    for event in pipeline.audit.events_for(state.inspection_id):
        vision = event.payload.get("vision")
        if isinstance(vision, dict):
            assert "anomaly_map_path" not in vision
            assert "regions" not in vision
